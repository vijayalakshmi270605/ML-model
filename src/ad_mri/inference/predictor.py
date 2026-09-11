"""The production inference engine: the single entry point that turns one MRI
file into a prediction.

Pipeline: MRI file -> validation -> preprocessing -> (cached) model loading ->
inference (`torch.inference_mode()`, no gradients) -> probability calibration
(if configured) -> prediction + confidence -> optional LIME/SHAP explanation.
Runnable as a library call, `ad-mri predict`, or `python -m
ad_mri.inference.predict`; the FastAPI `POST /api/v1/predictions` endpoint
(`api.routers.predictions`) is a thin wrapper around the exact same
`InferenceEngine.predict()` call — there is no second, HTTP-only copy of this
logic.

**Never retrains during inference**: the cached model is loaded once
(`inference.model_loader.ModelLoader`), moved to `.eval()` mode exactly once,
and nothing in this module ever calls `.train()`, `.backward()`, or touches an
optimizer — the model's weights are as fixed as the checkpoint they came from.

**Never logs sensitive patient information**: an MRI's file path (and
certainly its pixel data) can encode a patient identifier in a way a bare
`subject_id` in this project's already-de-identified manifests does not — a
clinical PACS export's filename might be `john_doe_1957.nii.gz`. Every log
line in this module identifies a request by `request_id` and, where an image
needs to be referenced, by `redact_image_reference()` (a one-way hash of the
resolved path — never the path itself). See docs/inference.md.

`torch` is a training-phase dependency (requirements/train.txt), imported
lazily so this module stays importable without it.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ad_mri.core.exceptions import (
    ConfigurationError,
    DataValidationError,
    InferenceError,
    PreprocessingError,
)
from ad_mri.core.logging import get_logger
from ad_mri.inference.calibration import CalibrationConfig, apply_calibration
from ad_mri.inference.model_loader import CachedModel, ModelLoader
from ad_mri.monitoring.drift import ImageStatistics, compute_image_statistics
from ad_mri.monitoring.metrics import (
    CLASS_PROBABILITY,
    INFERENCE_DURATION_SECONDS,
    INFERENCE_REQUESTS_TOTAL,
    INPUT_MEAN_INTENSITY,
    INPUT_NONZERO_FRACTION,
    INPUT_STD_INTENSITY,
    PREDICTED_CLASS_TOTAL,
    PREDICTION_CONFIDENCE,
)

if TYPE_CHECKING:
    from ad_mri.explainability.service import ExplanationBundle

logger = get_logger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 30.0
_DEFAULT_MAX_WORKERS = 4


@dataclass(frozen=True)
class PredictionResult:
    """The deterministic response structure every prediction returns —
    identical shape whether called from the CLI, a script, or the API
    (`api.schemas.predict.PredictionResponse` mirrors every field above
    `image_statistics` exactly, for the HTTP boundary). `to_dict()` is what
    gets JSON-serialized.

    `image_statistics` is the one field `PredictionResponse` does **not**
    mirror (pydantic's default `extra="ignore"` silently drops it rather
    than erroring) — it exists for `api.monitoring.record_input_statistics`'s
    best-effort persistence (Phase 13 data-drift monitoring), not for an API
    caller, and was never part of this response's documented contract.
    `to_dict()` still serializes it (as a plain nested dict) via
    `dataclasses.asdict`'s normal recursion — nothing downstream that calls
    `to_dict()` needs to special-case it.
    """

    request_id: str
    model_version: str
    preprocessing_version: str
    prediction: str
    confidence: float
    probabilities: dict[str, float]
    processing_time_ms: float
    explanation_available: bool
    explanation_id: str | None
    timestamp: str
    image_statistics: ImageStatistics | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelVersionInfo:
    """Backs the "model version endpoint" requirement — what's currently
    loaded and serving, independent of the full (not-yet-implemented) model
    registry listing."""

    architecture: str
    model_version: str
    model_config_hash: str
    preprocessing_version: str
    preprocessing_config_hash: str
    calibration_method: str
    device: str
    checkpoint_path: str
    loaded_at: str


def redact_image_reference(image_path: Path) -> str:
    """A stable, one-way reference to `image_path` safe to put in logs —
    never the path itself, which can encode a patient-identifying filename."""
    digest = hashlib.sha256(str(Path(image_path).resolve()).encode("utf-8")).hexdigest()
    return digest[:16]


class InferenceEngine:
    """Loads and caches one model (via `ModelLoader`), then serves
    `predict()` calls against it. One instance is meant to be long-lived —
    constructed once per process (the API's DI provider does exactly this)
    and reused across every request, satisfying "cache model in memory" for
    the model *and* for the thread pool `predict()` uses to enforce its
    timeout.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        preprocessing_config_path: Path | None = None,
        calibration_config: CalibrationConfig | None = None,
        device_choice: str = "auto",
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        explanation_background_manifest_path: Path | None = None,
        explanation_output_dir: Path = Path("artifacts/explanations"),
        model_loader: ModelLoader | None = None,
        max_workers: int = _DEFAULT_MAX_WORKERS,
    ) -> None:
        if timeout_seconds <= 0:
            raise ConfigurationError(f"timeout_seconds must be > 0, got {timeout_seconds!r}.")

        self._checkpoint_path = Path(checkpoint_path)
        self._preprocessing_config_path = Path(
            preprocessing_config_path or "configs/preprocessing/default.yaml"
        )
        self._calibration_config = calibration_config or CalibrationConfig()
        self._device_choice = device_choice
        self._timeout_seconds = timeout_seconds
        self._explanation_background_manifest_path = explanation_background_manifest_path
        self._explanation_output_dir = Path(explanation_output_dir)
        self._model_loader = model_loader or ModelLoader()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="ad-mri-inference"
        )

        # Eager, fail-fast load at construction — a misconfigured checkpoint/
        # preprocessing pair is caught at service startup, not on a request.
        self._cached: CachedModel = self._model_loader.load(
            self._checkpoint_path, self._preprocessing_config_path, self._device_choice
        )
        logger.info(
            "inference_engine_ready",
            model_version=self._cached.model_config.version,
            model_config_hash=self._cached.model_config.compute_hash(),
            calibration_method=self._calibration_config.method,
            device=str(self._cached.device),
            timeout_seconds=self._timeout_seconds,
        )

    # --- public API -----------------------------------------------------------

    @property
    def class_names(self) -> list[str]:
        return list(self._cached.class_names)

    def model_version_info(self) -> ModelVersionInfo:
        """Backs `GET /api/v1/models` (`api.routers.models`)."""
        return ModelVersionInfo(
            architecture=self._cached.model_config.architecture,
            model_version=self._cached.model_config.version,
            model_config_hash=self._cached.model_config.compute_hash(),
            preprocessing_version=self._cached.preprocessing_config.version,
            preprocessing_config_hash=self._cached.preprocessing_config.compute_hash(),
            calibration_method=self._calibration_config.method,
            device=str(self._cached.device),
            checkpoint_path=str(self._cached.checkpoint_path),
            loaded_at=self._cached.loaded_at,
        )

    def health_check(self, run_dummy_forward_pass: bool = True) -> dict[str, Any]:
        """Never raises — a failed health check is a signal, not an
        application error (mirrors `monitoring.health_checks.check_database`).
        `run_dummy_forward_pass=True` additionally runs one real forward pass
        on a zero tensor, catching a broken model (e.g. corrupted weights) a
        "model object exists" check alone would miss.
        """
        status: dict[str, Any] = {
            "model_loaded": True,
            "model_version": self._cached.model_config.version,
            "device": str(self._cached.device),
        }
        if not run_dummy_forward_pass:
            status["forward_pass_ok"] = None
            return status
        try:
            import torch

            dummy_shape = (
                1,
                self._cached.model_config.in_channels,
                *self._cached.model_config.input_shape,
            )
            with torch.inference_mode():
                logits = self._cached.model(torch.zeros(dummy_shape, device=self._cached.device))
            status["forward_pass_ok"] = bool(torch.isfinite(logits).all())
        except Exception as exc:  # noqa: BLE001 - health checks must never raise
            logger.warning(
                "inference_health_check_forward_pass_failed", error_type=type(exc).__name__
            )
            status["forward_pass_ok"] = False
        return status

    def predict(
        self,
        image_path: Path,
        request_id: str | None = None,
        include_explanation: bool = False,
        explanation_methods: tuple[str, ...] | None = None,
        explanation_target_class: str | None = None,
    ) -> PredictionResult:
        """Run the full pipeline for one MRI file and return a
        `PredictionResult`. Deterministic given identical model/calibration/
        image bytes (the same guarantee `evaluation.evaluate` gives, minus
        any GPU non-determinism outside this project's control).

        Raises:
            DataValidationError: the file is missing/corrupted/out of the
                configured dimension or voxel-count bounds — "handle invalid
                files gracefully" means raising this well-typed exception
                rather than crashing uninformatively; the API layer maps it
                to HTTP 422.
            PreprocessingError: the deterministic pipeline itself fails, or an
                enabled quality-control check fails.
            InferenceError: the pipeline did not finish within
                `timeout_seconds`, or the model produced a non-finite output.
            ConfigurationError: `explanation_target_class` is not one of this
                model's known classes.
        """
        resolved_request_id = request_id or str(uuid.uuid4())
        image_ref = redact_image_reference(image_path)
        start = time.perf_counter()

        logger.info(
            "inference_request_received",
            request_id=resolved_request_id,
            model_version=self._cached.model_config.version,
            image_ref=image_ref,
            include_explanation=include_explanation,
        )

        # The *prediction* is always the model's own argmax — never overridden
        # by a target-class request; only which class the *explanation* (if
        # any) covers is choosable, and only validated once actually needed.
        if explanation_target_class is not None and explanation_target_class not in (
            self._cached.class_names
        ):
            raise ConfigurationError(
                f"Unknown class {explanation_target_class!r}; "
                f"known classes: {self._cached.class_names}"
            )

        future: Future[tuple[int, list[float], ImageStatistics]] = self._executor.submit(
            self._run_pipeline, image_path
        )
        try:
            predicted_index, probabilities_list, image_stats = future.result(
                timeout=self._timeout_seconds
            )
        except FutureTimeoutError as exc:
            INFERENCE_REQUESTS_TOTAL.labels(outcome="timeout").inc()
            logger.warning(
                "inference_timed_out",
                request_id=resolved_request_id,
                image_ref=image_ref,
                timeout_seconds=self._timeout_seconds,
            )
            raise InferenceError(
                f"Inference timed out after {self._timeout_seconds}s "
                f"(request_id={resolved_request_id})."
            ) from exc
        except DataValidationError as exc:
            # Never log str(exc) here: these exceptions' messages legitimately
            # include the raw file path (needed for the *caller* to debug),
            # which is exactly what must never reach a log line — only the
            # exception type and the already-redacted image_ref are logged.
            # A distinct outcome from PreprocessingError below (Phase 13:
            # "input validation failures" and "preprocessing failures" are
            # two separately-monitored things, not one conflated bucket).
            INFERENCE_REQUESTS_TOTAL.labels(outcome="validation_failed").inc()
            logger.warning(
                "inference_input_rejected",
                request_id=resolved_request_id,
                image_ref=image_ref,
                error_type=type(exc).__name__,
            )
            raise
        except PreprocessingError as exc:
            INFERENCE_REQUESTS_TOTAL.labels(outcome="preprocessing_failed").inc()
            logger.warning(
                "inference_preprocessing_failed",
                request_id=resolved_request_id,
                image_ref=image_ref,
                error_type=type(exc).__name__,
            )
            raise
        except Exception:
            INFERENCE_REQUESTS_TOTAL.labels(outcome="error").inc()
            raise

        # Core pipeline (validation -> preprocessing -> inference -> calibration)
        # finished — record its outcome/duration before the optional,
        # separately-timed explanation step runs.
        core_elapsed_seconds = time.perf_counter() - start
        INFERENCE_REQUESTS_TOTAL.labels(outcome="success").inc()
        INFERENCE_DURATION_SECONDS.observe(core_elapsed_seconds)

        class_names = self._cached.class_names
        probabilities = dict(zip(class_names, probabilities_list, strict=True))
        predicted_class = class_names[predicted_index]
        confidence = probabilities_list[predicted_index]

        # Model monitoring (Phase 13): class/confidence/prediction
        # distributions, and the real-time half of data-drift monitoring
        # (see monitoring.drift's module docstring for the offline half).
        PREDICTED_CLASS_TOTAL.labels(predicted_class).inc()
        PREDICTION_CONFIDENCE.observe(confidence)
        for class_name, probability in probabilities.items():
            CLASS_PROBABILITY.labels(class_name).observe(probability)
        INPUT_MEAN_INTENSITY.observe(image_stats.mean_intensity)
        INPUT_STD_INTENSITY.observe(image_stats.std_intensity)
        INPUT_NONZERO_FRACTION.observe(image_stats.nonzero_fraction)

        explanation_id: str | None = None
        explanation_available = False
        if include_explanation:
            explanation_id = self._try_generate_explanation(
                image_path, resolved_request_id, explanation_methods, explanation_target_class
            )
            explanation_available = explanation_id is not None

        # processing_time_ms is the full request wall-clock time (including
        # the optional explanation step, if requested) — what a caller
        # actually experienced; INFERENCE_DURATION_SECONDS above measures
        # only the core prediction pipeline, so XAI cost doesn't skew it.
        processing_time_ms = round((time.perf_counter() - start) * 1000, 3)

        result = PredictionResult(
            request_id=resolved_request_id,
            model_version=self._cached.model_config.version,
            preprocessing_version=self._cached.preprocessing_config.version,
            prediction=predicted_class,
            confidence=confidence,
            probabilities=probabilities,
            processing_time_ms=processing_time_ms,
            explanation_available=explanation_available,
            explanation_id=explanation_id,
            timestamp=datetime.now(UTC).isoformat(),
            image_statistics=image_stats,
        )

        logger.info(
            "inference_request_completed",
            request_id=resolved_request_id,
            image_ref=image_ref,
            prediction=predicted_class,
            confidence=round(confidence, 4),
            processing_time_ms=processing_time_ms,
            explanation_available=explanation_available,
        )
        return result

    # --- internals --------------------------------------------------------------

    def _run_pipeline(self, image_path: Path) -> tuple[int, list[float], ImageStatistics]:
        """Runs in a worker thread so `predict()` can enforce
        `timeout_seconds` via `Future.result(timeout=...)` — a pathological
        input (e.g. a huge malformed file) that hangs preprocessing cannot
        hang the calling thread indefinitely.

        Validation -> preprocessing -> `torch.inference_mode()` forward pass
        -> calibration, in that order. Never touches gradients, `.train()`,
        or an optimizer — see the module docstring's "never retrains"
        guarantee.

        Also computes `ImageStatistics` (Phase 13 data-drift monitoring) on
        the preprocessed `volume` right here - the exact array the model is
        about to see, already in memory, so this costs nothing extra to
        load and measures what actually matters (the model's real input,
        not the raw uploaded file, which may have a different resolution/
        orientation/intensity range entirely before preprocessing).
        """
        import torch

        from ad_mri.datasets.transforms import apply_transforms, build_transforms
        from ad_mri.datasets.validation import validate_image_for_pipeline

        validate_image_for_pipeline(image_path, self._cached.preprocessing_config)
        transform = build_transforms(self._cached.preprocessing_config, train=False)
        sample = apply_transforms(transform, {"image": str(image_path)})
        volume = sample["image"]
        image_stats = compute_image_statistics(volume)

        with torch.inference_mode():
            logits = self._cached.model(volume.unsqueeze(0).to(self._cached.device))
            probabilities = apply_calibration(logits.float(), self._calibration_config)[0]

        if not bool(torch.isfinite(probabilities).all()):
            raise InferenceError("Model produced non-finite probabilities for this input.")

        return int(torch.argmax(probabilities).item()), probabilities.cpu().tolist(), image_stats

    def generate_explanation(
        self,
        image_path: Path,
        request_id: str | None = None,
        methods: tuple[str, ...] | None = None,
        target_class: str | None = None,
    ) -> ExplanationBundle:
        """Generate a LIME/SHAP explanation against the *already-loaded,
        already-cached* checkpoint — via `ExplainabilityService.from_loaded_model`,
        so a caller that wants both a prediction and an explanation never
        pays for loading the checkpoint twice.

        Unlike `predict(include_explanation=True)` (best-effort — a bonus
        explanation must never fail an otherwise-valid prediction), this is
        the **loud** path: appropriate for a caller (e.g. a dedicated
        `POST /explanations` endpoint) whose entire purpose is producing an
        explanation, where "silently unavailable" is not a sensible outcome —
        every exception `explainability.service.ExplainabilityService.explain`
        can raise propagates unchanged.

        `target_class` defaults to the model's own predicted class when
        omitted (resolved by `ExplainabilityService.explain` itself, from its
        own — identical, deterministic — forward pass).

        Raises:
            ConfigurationError: `target_class` is not one of this model's
                known classes, an unsupported method is requested, or "shap"
                is requested without a configured background manifest.
        """
        from ad_mri.explainability.service import ExplainabilityService

        resolved_request_id = request_id or str(uuid.uuid4())
        resolved_methods = self._resolve_explanation_methods(methods)
        target_class_index = self._resolve_class_index(target_class)

        service = ExplainabilityService.from_loaded_model(
            self._cached.model,
            self._cached.model_config,
            self._cached.metadata,
            self._cached.preprocessing_config,
            self._cached.checkpoint_path,
            self._cached.device,
        )
        return service.explain(
            image_path,
            input_id=resolved_request_id,
            methods=resolved_methods,
            background_manifest_path=self._explanation_background_manifest_path,
            target_class_index=target_class_index,
            output_dir=self._explanation_output_dir / resolved_request_id,
        )

    def _resolve_explanation_methods(self, methods: tuple[str, ...] | None) -> tuple[str, ...]:
        return methods or (
            ("lime", "shap") if self._explanation_background_manifest_path else ("lime",)
        )

    def _resolve_class_index(self, class_name: str | None) -> int | None:
        if class_name is None:
            return None
        if class_name not in self._cached.class_names:
            raise ConfigurationError(
                f"Unknown class {class_name!r}; known classes: {self._cached.class_names}"
            )
        return self._cached.class_names.index(class_name)

    def _try_generate_explanation(
        self,
        image_path: Path,
        request_id: str,
        methods: tuple[str, ...] | None,
        target_class: str | None,
    ) -> str | None:
        """Best-effort wrapper around `generate_explanation` for
        `predict(include_explanation=True)`: a failed explanation never fails
        the underlying prediction (the caller already has a valid prediction
        by this point) — logged and reported as unavailable instead."""
        try:
            self.generate_explanation(image_path, request_id, methods, target_class)
            return request_id
        except Exception as exc:  # noqa: BLE001 - explanation is best-effort, never fatal here
            logger.warning(
                "inference_explanation_failed",
                request_id=request_id,
                error_type=type(exc).__name__,
            )
            return None

    def shutdown(self) -> None:
        """Release the worker thread pool. Call on process shutdown (the
        API's lifespan handler does); not required for a short-lived script."""
        self._executor.shutdown(wait=True)
