"""End-to-end tests for explainability.service.ExplainabilityService — a real
checkpoint, real preprocessing, real LIME + SHAP, real manifest.json (no mocks).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lime")
pytest.importorskip("shap")

from ad_mri.core.exceptions import ConfigurationError  # noqa: E402
from ad_mri.explainability.config import LimeConfig, ShapConfig  # noqa: E402
from ad_mri.explainability.service import ExplainabilityService  # noqa: E402

_LIME_CONFIG = LimeConfig(
    num_samples=30,
    quickshift_kernel_size=2.0,
    quickshift_max_dist=10.0,
    quickshift_ratio=0.2,
    seed=42,
)
_SHAP_CONFIG = ShapConfig(n_background_samples=3, nsamples=10, seed=42)


def _first_image_path(manifest: Path) -> Path:
    import pandas as pd

    df = pd.read_csv(manifest)
    return Path(df.iloc[0]["image_path"])


def test_explain_returns_prediction_and_both_explanations(
    tmp_path: Path, checkpoint: tuple[Path, Path], manifest: Path
) -> None:
    checkpoint_path, preprocessing_path = checkpoint
    service = ExplainabilityService(checkpoint_path, preprocessing_path)

    bundle = service.explain(
        _first_image_path(manifest),
        methods=("lime", "shap"),
        lime_config=_LIME_CONFIG,
        shap_config=_SHAP_CONFIG,
        background_manifest_path=manifest,
        output_dir=tmp_path / "out",
    )

    assert bundle.prediction.predicted_class in {"AD", "CN"}
    assert 0.0 <= bundle.prediction.confidence <= 1.0
    assert set(bundle.prediction.class_probabilities) == {"AD", "CN"}
    assert sum(bundle.prediction.class_probabilities.values()) == pytest.approx(1.0, abs=1e-4)
    assert bundle.lime is not None
    assert bundle.shap is not None


def test_explain_defaults_to_explaining_the_predicted_class(
    tmp_path: Path, checkpoint: tuple[Path, Path], manifest: Path
) -> None:
    checkpoint_path, preprocessing_path = checkpoint
    service = ExplainabilityService(checkpoint_path, preprocessing_path)
    bundle = service.explain(
        _first_image_path(manifest),
        methods=("lime",),
        lime_config=_LIME_CONFIG,
        output_dir=tmp_path / "out",
    )
    predicted_index = service.class_names.index(bundle.prediction.predicted_class)
    assert bundle.target_class_index == predicted_index


def test_explain_target_class_can_be_overridden(
    tmp_path: Path, checkpoint: tuple[Path, Path], manifest: Path
) -> None:
    checkpoint_path, preprocessing_path = checkpoint
    service = ExplainabilityService(checkpoint_path, preprocessing_path)
    non_predicted_index = 1  # explicit override, regardless of what was predicted
    bundle = service.explain(
        _first_image_path(manifest),
        methods=("lime",),
        lime_config=_LIME_CONFIG,
        target_class_index=non_predicted_index,
        output_dir=tmp_path / "out",
    )
    assert bundle.target_class_index == non_predicted_index
    assert bundle.lime.target_class_index == non_predicted_index


def test_explain_records_model_and_preprocessing_provenance(
    tmp_path: Path, checkpoint: tuple[Path, Path], manifest: Path
) -> None:
    from ad_mri.datasets.preprocessing_config import load_preprocessing_config
    from ad_mri.modeling.checkpoint import load_checkpoint

    checkpoint_path, preprocessing_path = checkpoint
    _model, model_config, _metadata = load_checkpoint(checkpoint_path)
    preprocessing_config = load_preprocessing_config(preprocessing_path)

    service = ExplainabilityService(checkpoint_path, preprocessing_path)
    bundle = service.explain(
        _first_image_path(manifest),
        methods=("lime",),
        lime_config=_LIME_CONFIG,
        output_dir=tmp_path / "out",
    )
    assert bundle.model_version == model_config.version
    assert bundle.model_config_hash == model_config.compute_hash()
    assert bundle.checkpoint_path == str(checkpoint_path)
    assert bundle.preprocessing_version == preprocessing_config.version
    assert bundle.preprocessing_config_hash == preprocessing_config.compute_hash()


def test_explain_input_id_defaults_to_image_stem(
    tmp_path: Path, checkpoint: tuple[Path, Path], manifest: Path
) -> None:
    checkpoint_path, preprocessing_path = checkpoint
    service = ExplainabilityService(checkpoint_path, preprocessing_path)
    image_path = _first_image_path(manifest)
    bundle = service.explain(
        image_path, methods=("lime",), lime_config=_LIME_CONFIG, output_dir=tmp_path / "out"
    )
    assert bundle.input_id == image_path.stem


def test_explain_input_id_can_be_overridden(
    tmp_path: Path, checkpoint: tuple[Path, Path], manifest: Path
) -> None:
    checkpoint_path, preprocessing_path = checkpoint
    service = ExplainabilityService(checkpoint_path, preprocessing_path)
    bundle = service.explain(
        _first_image_path(manifest),
        input_id="patient-42-visit-1",
        methods=("lime",),
        lime_config=_LIME_CONFIG,
        output_dir=tmp_path / "out",
    )
    assert bundle.input_id == "patient-42-visit-1"


def test_shap_without_background_manifest_raises(
    tmp_path: Path, checkpoint: tuple[Path, Path], manifest: Path
) -> None:
    checkpoint_path, preprocessing_path = checkpoint
    service = ExplainabilityService(checkpoint_path, preprocessing_path)
    with pytest.raises(ConfigurationError, match="background_manifest_path"):
        service.explain(
            _first_image_path(manifest),
            methods=("shap",),
            shap_config=_SHAP_CONFIG,
            output_dir=tmp_path / "out",
        )


def test_unsupported_method_raises(
    tmp_path: Path, checkpoint: tuple[Path, Path], manifest: Path
) -> None:
    checkpoint_path, preprocessing_path = checkpoint
    service = ExplainabilityService(checkpoint_path, preprocessing_path)
    with pytest.raises(ConfigurationError, match="Unsupported explanation method"):
        service.explain(
            _first_image_path(manifest), methods=("gradcam",), output_dir=tmp_path / "out"
        )


def test_out_of_range_target_class_index_raises(
    tmp_path: Path, checkpoint: tuple[Path, Path], manifest: Path
) -> None:
    checkpoint_path, preprocessing_path = checkpoint
    service = ExplainabilityService(checkpoint_path, preprocessing_path)
    with pytest.raises(ConfigurationError, match="out of range"):
        service.explain(
            _first_image_path(manifest),
            methods=("lime",),
            lime_config=_LIME_CONFIG,
            target_class_index=5,
            output_dir=tmp_path / "out",
        )


def test_checkpoint_without_class_names_is_rejected(tmp_path: Path) -> None:
    from ad_mri.datasets.preprocessing_config import PreprocessingConfig, save_preprocessing_config
    from ad_mri.modeling.checkpoint import save_checkpoint
    from ad_mri.modeling.factory import build_model
    from ad_mri.modeling.model_config import ModelConfig

    model_config = ModelConfig(
        num_classes=2,
        spatial_dims=3,
        input_shape=(16, 16, 16),
        num_blocks=1,
        base_channels=4,
        fc_hidden_dim=8,
    )
    model = build_model(model_config)
    checkpoint_path = tmp_path / "bare.pt"
    save_checkpoint(model, model_config, checkpoint_path, extra={})  # no class_names

    preprocessing_path = save_preprocessing_config(
        PreprocessingConfig(
            target_shape=(16, 16, 16), target_spacing=(1.0, 1.0, 1.0), min_voxel_count=1000
        ),
        tmp_path / "preprocessing.yaml",
    )
    with pytest.raises(ConfigurationError, match="class_names"):
        ExplainabilityService(checkpoint_path, preprocessing_path)


def test_mismatched_preprocessing_is_rejected(
    tmp_path: Path, checkpoint: tuple[Path, Path]
) -> None:
    from ad_mri.datasets.preprocessing_config import PreprocessingConfig, save_preprocessing_config

    checkpoint_path, _correct_preprocessing_path = checkpoint
    wrong_preprocessing_path = save_preprocessing_config(
        PreprocessingConfig(
            target_shape=(16, 16, 16), target_spacing=(2.0, 2.0, 2.0), min_voxel_count=1000
        ),
        tmp_path / "wrong_preprocessing.yaml",
    )
    with pytest.raises(ConfigurationError, match="does not match"):
        ExplainabilityService(checkpoint_path, wrong_preprocessing_path)


def test_manifest_json_contains_every_required_field(
    tmp_path: Path, checkpoint: tuple[Path, Path], manifest: Path
) -> None:
    """Requirement: store model version, input ID, preprocessing version,
    explanation method, explanation parameters, generated artifacts."""
    checkpoint_path, preprocessing_path = checkpoint
    service = ExplainabilityService(checkpoint_path, preprocessing_path)
    bundle = service.explain(
        _first_image_path(manifest),
        input_id="subject-007",
        methods=("lime", "shap"),
        lime_config=_LIME_CONFIG,
        shap_config=_SHAP_CONFIG,
        background_manifest_path=manifest,
        output_dir=tmp_path / "out",
    )

    manifest_path = bundle.artifacts["manifest"]
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["input_id"] == "subject-007"
    assert payload["model_version"] == bundle.model_version
    assert payload["preprocessing_version"] == bundle.preprocessing_version
    assert "lime" in payload["methods"]
    assert "shap" in payload["methods"]
    # explanation parameters: the config content hash is present per method
    assert payload["methods"]["lime"]["config_hash"] == _LIME_CONFIG.compute_hash()
    assert payload["methods"]["shap"]["config_hash"] == _SHAP_CONFIG.compute_hash()
    # generated artifacts: every artifact path referenced actually exists on disk
    for artifact_path in payload["artifacts"].values():
        assert Path(artifact_path).exists()
    for method_payload in payload["methods"].values():
        for artifact_path in method_payload["artifacts"].values():
            assert Path(artifact_path).exists()


def test_prediction_json_artifact_matches_the_bundle(
    tmp_path: Path, checkpoint: tuple[Path, Path], manifest: Path
) -> None:
    checkpoint_path, preprocessing_path = checkpoint
    service = ExplainabilityService(checkpoint_path, preprocessing_path)
    bundle = service.explain(
        _first_image_path(manifest),
        methods=("lime",),
        lime_config=_LIME_CONFIG,
        output_dir=tmp_path / "out",
    )
    prediction_payload = json.loads(bundle.artifacts["prediction"].read_text(encoding="utf-8"))
    assert prediction_payload["predicted_class"] == bundle.prediction.predicted_class
    assert prediction_payload["confidence"] == pytest.approx(bundle.prediction.confidence)


def test_explain_with_lime_only_leaves_shap_none(
    tmp_path: Path, checkpoint: tuple[Path, Path], manifest: Path
) -> None:
    checkpoint_path, preprocessing_path = checkpoint
    service = ExplainabilityService(checkpoint_path, preprocessing_path)
    bundle = service.explain(
        _first_image_path(manifest),
        methods=("lime",),
        lime_config=_LIME_CONFIG,
        output_dir=tmp_path / "out",
    )
    assert bundle.lime is not None
    assert bundle.shap is None
