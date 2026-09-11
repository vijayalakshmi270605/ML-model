"""The explainability service: the single entry point that ties a trained
checkpoint + one MRI scan together into a prediction plus LIME and/or SHAP
explanations, with every artifact traceable back to the exact model,
preprocessing, and explanation parameters that produced it.

Receives an image path + a trained model (checkpoint path) and returns the
prediction, class probabilities, and requested explanations — exactly the
Phase 8 contract. Deliberately independent of the MLflow-registry-backed
`inference.predictor.InferenceEngine` (a separate, unrelated stub pending the
model-registry phase — still true when this was written): this service loads
a checkpoint directly, the same way `evaluation.evaluate`/
`training.cross_validation` do, so explainability does not have to wait on an
unrelated phase. It *does* reuse the identical deterministic preprocessing
pipeline every other phase uses (`datasets.transforms.build_transforms(...,
train=False)`) — never a second, explainability-only preprocessing
implementation.

`inference.predictor.InferenceEngine` (Phase 9) is a different caller of this
same class: its optional-XAI path builds a service via
`ExplainabilityService.from_loaded_model` — bypassing `__init__`'s own
checkpoint load — around the checkpoint its own `inference.model_loader`
cache already holds, so a request asking for both a prediction and an
explanation loads the checkpoint from disk once, not twice.

See docs/TECHNICAL_SPECIFICATION.md §7, docs/explainability.md.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ad_mri.core.exceptions import ConfigurationError
from ad_mri.core.logging import get_logger
from ad_mri.core.seeding import set_global_seed
from ad_mri.datasets.preprocessing_config import PreprocessingConfig, load_preprocessing_config
from ad_mri.datasets.validation import validate_image_for_pipeline
from ad_mri.explainability.config import LimeConfig, ShapConfig
from ad_mri.explainability.lime_explainer import LimeExplanationResult, explain_with_lime
from ad_mri.explainability.sampling import load_volume_batch
from ad_mri.explainability.shap_explainer import ShapExplanationResult, explain_with_shap
from ad_mri.modeling.checkpoint import CheckpointMetadata, load_checkpoint
from ad_mri.modeling.model_config import ModelConfig

logger = get_logger(__name__)

_SUPPORTED_METHODS = ("lime", "shap")


@dataclass(frozen=True)
class PredictionSummary:
    predicted_class: str
    confidence: float
    class_probabilities: dict[str, float]


@dataclass(frozen=True)
class ExplanationBundle:
    """Everything Phase 8 requires to be stored: model version, input ID,
    preprocessing version, explanation method(s), their parameters, and every
    generated artifact — see `write_manifest`."""

    input_id: str
    image_path: str
    model_version: str
    model_config_hash: str
    checkpoint_path: str
    preprocessing_version: str
    preprocessing_config_hash: str
    class_names: list[str]
    prediction: PredictionSummary
    target_class_index: int
    lime: LimeExplanationResult | None
    shap: ShapExplanationResult | None
    generated_at: str
    output_dir: Path
    artifacts: dict[str, Path] = field(default_factory=dict)


def _resolve_class_names(metadata_extra: dict[str, Any], checkpoint_path: Path) -> list[str]:
    class_names = metadata_extra.get("class_names")
    if not class_names:
        raise ConfigurationError(
            f"Checkpoint at {checkpoint_path} does not carry a class_names mapping "
            "(expected in its training-provenance metadata) — cannot reliably map "
            "predicted indices back to label names. Re-train with the current "
            "training pipeline (Phase 5+), which always records this."
        )
    return list(class_names)


class ExplainabilityService:
    """Loads one checkpoint once, then explains as many images as asked."""

    def __init__(
        self,
        model_path: Path,
        preprocessing_config_path: Path | None = None,
        device_choice: str = "auto",
    ) -> None:
        import torch

        self.model_path = Path(model_path)
        self.model, self.model_config, self._metadata = load_checkpoint(self.model_path)
        self.class_names = _resolve_class_names(self._metadata.extra, self.model_path)
        if len(self.class_names) != self.model_config.num_classes:
            raise ConfigurationError(
                f"Checkpoint class_names {self.class_names} has {len(self.class_names)} "
                f"entries but model_config.num_classes={self.model_config.num_classes}."
            )

        resolved_preprocessing_path = preprocessing_config_path or Path(
            "configs/preprocessing/default.yaml"
        )
        self.preprocessing_config: PreprocessingConfig = load_preprocessing_config(
            resolved_preprocessing_path
        )
        if tuple(self.preprocessing_config.target_shape) != tuple(self.model_config.input_shape):
            raise ConfigurationError(
                f"preprocessing_config.target_shape {self.preprocessing_config.target_shape} "
                f"!= model_config.input_shape {self.model_config.input_shape}."
            )
        stored_hash = self._metadata.extra.get("preprocessing_config_hash")
        if stored_hash and stored_hash != self.preprocessing_config.compute_hash():
            raise ConfigurationError(
                f"preprocessing config at {resolved_preprocessing_path} (hash "
                f"{self.preprocessing_config.compute_hash()[:12]}) does not match the one "
                f"the checkpoint was trained with (hash {stored_hash[:12]}) — explaining "
                "with mismatched preprocessing would explain a different model than the "
                "one that produced this checkpoint's weights."
            )

        if device_choice == "cpu":
            self.device = torch.device("cpu")
        elif device_choice == "cuda":
            if not torch.cuda.is_available():
                raise ConfigurationError("device='cuda' was requested but CUDA is not available.")
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_loaded_model(
        cls,
        model: Any,
        model_config: ModelConfig,
        metadata: CheckpointMetadata,
        preprocessing_config: PreprocessingConfig,
        checkpoint_path: Path,
        device: Any,
    ) -> ExplainabilityService:
        """Build a service around an **already-loaded, already-validated**
        checkpoint — e.g. a `inference.model_loader.CachedModel`'s fields —
        instead of loading one from disk via `__init__`. The caller is
        responsible for having already performed the validation `__init__`
        would otherwise do (class-names presence, `num_classes` match,
        preprocessing shape/hash match) — true by construction for anything
        that came out of `ModelLoader`, which performs exactly these checks
        before caching. `model` is expected to already be on `device` and in
        eval mode.
        """
        service = cls.__new__(cls)
        service.model_path = Path(checkpoint_path)
        service.model = model
        service.model_config = model_config
        service._metadata = metadata
        service.class_names = _resolve_class_names(metadata.extra, service.model_path)
        service.preprocessing_config = preprocessing_config
        service.device = device
        return service

    def _preprocess(self, image_path: Path) -> Any:
        """Identical deterministic pipeline used by training/evaluation/inference
        (`train=False`: no augmentation) — see module docstring."""
        from ad_mri.datasets.transforms import apply_transforms, build_transforms

        validate_image_for_pipeline(image_path, self.preprocessing_config)
        transform = build_transforms(self.preprocessing_config, train=False)
        sample = apply_transforms(transform, {"image": str(image_path)})
        return sample["image"]

    def _predict(self, volume: Any) -> tuple[PredictionSummary, int]:
        import torch

        with torch.no_grad():
            logits = self.model(volume.unsqueeze(0).to(self.device))
            probs = torch.softmax(logits.float(), dim=1)[0].cpu().tolist()
        predicted_index = max(range(len(probs)), key=lambda i: probs[i])
        summary = PredictionSummary(
            predicted_class=self.class_names[predicted_index],
            confidence=float(probs[predicted_index]),
            class_probabilities=dict(zip(self.class_names, (float(p) for p in probs), strict=True)),
        )
        return summary, predicted_index

    def explain(
        self,
        image_path: Path,
        input_id: str | None = None,
        methods: tuple[str, ...] = _SUPPORTED_METHODS,
        lime_config: LimeConfig | None = None,
        shap_config: ShapConfig | None = None,
        background_manifest_path: Path | None = None,
        target_class_index: int | None = None,
        output_dir: Path | None = None,
        seed: int = 42,
    ) -> ExplanationBundle:
        """Run inference on `image_path`, then generate the requested
        explanation(s) for `target_class_index` (default: the predicted class).

        Raises:
            ConfigurationError: an unsupported method is requested, "shap" is
                requested without `background_manifest_path`, or
                `target_class_index` is out of range.
        """
        unknown = set(methods) - set(_SUPPORTED_METHODS)
        if unknown:
            raise ConfigurationError(f"Unsupported explanation method(s): {sorted(unknown)}")
        if "shap" in methods and background_manifest_path is None:
            raise ConfigurationError(
                "background_manifest_path is required to generate a SHAP explanation "
                "(the reference/baseline sample set — see docs/explainability.md)."
            )

        image_path = Path(image_path)
        resolved_input_id = input_id or image_path.stem
        resolved_output_dir = output_dir or (
            Path("artifacts/explanations") / f"{resolved_input_id}-{time.strftime('%Y%m%d-%H%M%S')}"
        )
        resolved_output_dir.mkdir(parents=True, exist_ok=True)

        set_global_seed(seed)
        volume = self._preprocess(image_path)
        prediction, predicted_index = self._predict(volume)
        resolved_target_index = (
            target_class_index if target_class_index is not None else predicted_index
        )
        if not (0 <= resolved_target_index < len(self.class_names)):
            raise ConfigurationError(
                f"target_class_index {resolved_target_index} is out of range for "
                f"{len(self.class_names)} class(es)."
            )

        logger.info(
            "explanation_requested",
            input_id=resolved_input_id,
            image_path=str(image_path),
            methods=list(methods),
            predicted_class=prediction.predicted_class,
            target_class=self.class_names[resolved_target_index],
        )

        artifacts: dict[str, Path] = {}
        lime_result: LimeExplanationResult | None = None
        if "lime" in methods:
            lime_result = explain_with_lime(
                self.model,
                volume,
                self.class_names,
                resolved_target_index,
                lime_config or LimeConfig(seed=seed),
                resolved_output_dir,
                self.device,
            )
            artifacts.update(lime_result.artifacts)

        shap_result: ShapExplanationResult | None = None
        if "shap" in methods:
            assert background_manifest_path is not None  # checked above
            resolved_shap_config = shap_config or ShapConfig(seed=seed)
            background, background_subject_ids = load_volume_batch(
                background_manifest_path,
                self.preprocessing_config,
                resolved_shap_config.n_background_samples,
                seed=resolved_shap_config.seed,
            )
            shap_result = explain_with_shap(
                self.model,
                volume,
                self.class_names,
                resolved_target_index,
                background,
                resolved_shap_config,
                resolved_output_dir,
                self.device,
            )
            artifacts.update(shap_result.artifacts)
            logger.info(
                "shap_background_sampled",
                manifest_path=str(background_manifest_path),
                subject_ids=background_subject_ids,
            )

        prediction_path = resolved_output_dir / "prediction.json"
        prediction_path.write_text(json.dumps(asdict(prediction), indent=2), encoding="utf-8")
        artifacts["prediction"] = prediction_path

        bundle = ExplanationBundle(
            input_id=resolved_input_id,
            image_path=str(image_path),
            model_version=self.model_config.version,
            model_config_hash=self.model_config.compute_hash(),
            checkpoint_path=str(self.model_path),
            preprocessing_version=self.preprocessing_config.version,
            preprocessing_config_hash=self.preprocessing_config.compute_hash(),
            class_names=self.class_names,
            prediction=prediction,
            target_class_index=resolved_target_index,
            lime=lime_result,
            shap=shap_result,
            generated_at=datetime.now(UTC).isoformat(),
            output_dir=resolved_output_dir,
            artifacts=artifacts,
        )
        manifest_path = write_manifest(bundle)
        bundle.artifacts["manifest"] = manifest_path
        return bundle


def write_manifest(bundle: ExplanationBundle) -> Path:
    """Persist `bundle` as `manifest.json` under `bundle.output_dir` — the
    durable record of "model version, input ID, preprocessing version,
    explanation method, explanation parameters, generated artifacts"."""

    def _artifact_dict(artifacts: dict[str, Path]) -> dict[str, str]:
        return {name: str(path) for name, path in artifacts.items()}

    payload: dict[str, Any] = {
        "input_id": bundle.input_id,
        "image_path": bundle.image_path,
        "model_version": bundle.model_version,
        "model_config_hash": bundle.model_config_hash,
        "checkpoint_path": bundle.checkpoint_path,
        "preprocessing_version": bundle.preprocessing_version,
        "preprocessing_config_hash": bundle.preprocessing_config_hash,
        "class_names": bundle.class_names,
        "target_class_index": bundle.target_class_index,
        "target_class_name": bundle.class_names[bundle.target_class_index],
        "prediction": asdict(bundle.prediction),
        "generated_at": bundle.generated_at,
        "methods": {},
        "artifacts": _artifact_dict(bundle.artifacts),
    }
    if bundle.lime is not None:
        lime_payload = asdict(bundle.lime)
        lime_payload["artifacts"] = _artifact_dict(bundle.lime.artifacts)
        payload["methods"]["lime"] = lime_payload
    if bundle.shap is not None:
        shap_payload = asdict(bundle.shap)
        shap_payload["artifacts"] = _artifact_dict(bundle.shap.artifacts)
        payload["methods"]["shap"] = shap_payload

    manifest_path = bundle.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path
