"""Integration tests for inference.predictor.InferenceEngine — a real, tiny
checkpoint, real preprocessing, real inference (no mocks).
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")
pytest.importorskip("nibabel")

from ad_mri.core.exceptions import (  # noqa: E402
    ConfigurationError,
    DataValidationError,
    InferenceError,
)
from ad_mri.datasets.transforms import apply_transforms, build_transforms  # noqa: E402
from ad_mri.inference.calibration import CalibrationConfig  # noqa: E402
from ad_mri.inference.model_loader import ModelLoader  # noqa: E402
from ad_mri.inference.predictor import InferenceEngine, redact_image_reference  # noqa: E402
from tests.helpers import write_synthetic_nifti  # noqa: E402


@pytest.fixture
def checkpoint(tmp_path: Path) -> tuple[Path, Path]:
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
    preprocessing_config = PreprocessingConfig(
        target_shape=(16, 16, 16), target_spacing=(1.0, 1.0, 1.0), min_voxel_count=1000
    )
    preprocessing_path = save_preprocessing_config(
        preprocessing_config, tmp_path / "preprocessing.yaml"
    )

    model = build_model(model_config)
    checkpoint_path = tmp_path / "model.pt"
    save_checkpoint(
        model,
        model_config,
        checkpoint_path,
        extra={
            "class_names": ["AD", "CN"],
            "preprocessing_config_hash": preprocessing_config.compute_hash(),
        },
    )
    return checkpoint_path, preprocessing_path


@pytest.fixture
def image(tmp_path: Path) -> Path:
    return write_synthetic_nifti(tmp_path / "scan.nii.gz", shape=(32, 32, 32))


@pytest.fixture
def engine(checkpoint: tuple[Path, Path], tmp_path: Path) -> InferenceEngine:
    checkpoint_path, preprocessing_path = checkpoint
    built = InferenceEngine(
        checkpoint_path,
        preprocessing_path,
        timeout_seconds=30.0,
        explanation_output_dir=tmp_path / "explanations",
    )
    yield built
    built.shutdown()


def test_predict_returns_the_deterministic_response_structure(
    engine: InferenceEngine, image: Path
) -> None:
    result = engine.predict(image)
    assert set(result.to_dict()) == {
        "request_id",
        "model_version",
        "preprocessing_version",
        "prediction",
        "confidence",
        "probabilities",
        "processing_time_ms",
        "explanation_available",
        "explanation_id",
        "timestamp",
        "image_statistics",
    }
    assert result.prediction in {"AD", "CN"}
    assert set(result.probabilities) == {"AD", "CN"}
    assert result.confidence == pytest.approx(max(result.probabilities.values()))
    assert result.confidence == pytest.approx(result.probabilities[result.prediction])
    assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-5)
    # Phase 13: real per-request image statistics, computed on the exact
    # preprocessed volume the model saw - not exposed via the HTTP response
    # (api.schemas.predict.PredictionResponse), but present on the engine's
    # own return value for api.monitoring's best-effort persistence.
    assert result.image_statistics is not None
    assert result.image_statistics.voxel_count > 0
    # `to_dict()` (dataclasses.asdict) recurses into the nested
    # ImageStatistics dataclass too, producing a plain dict either way.
    serialized = result.to_dict()["image_statistics"]
    assert set(serialized) == {
        "mean_intensity",
        "std_intensity",
        "p01_intensity",
        "p50_intensity",
        "p99_intensity",
        "nonzero_fraction",
        "voxel_count",
    }
    assert result.processing_time_ms > 0
    assert result.explanation_available is False
    assert result.explanation_id is None


def test_request_id_defaults_to_a_uuid(engine: InferenceEngine, image: Path) -> None:
    result = engine.predict(image)
    uuid.UUID(result.request_id)  # must not raise


def test_request_id_is_honored_when_given(engine: InferenceEngine, image: Path) -> None:
    result = engine.predict(image, request_id="my-custom-id-42")
    assert result.request_id == "my-custom-id-42"


def test_predict_is_deterministic_across_calls(engine: InferenceEngine, image: Path) -> None:
    first = engine.predict(image, request_id="a")
    second = engine.predict(image, request_id="b")
    assert first.prediction == second.prediction
    assert first.probabilities == second.probabilities


def test_missing_file_raises_data_validation_error(engine: InferenceEngine, tmp_path: Path) -> None:
    with pytest.raises(DataValidationError):
        engine.predict(tmp_path / "does-not-exist.nii.gz")


def test_corrupted_file_raises_gracefully_not_a_crash(
    engine: InferenceEngine, tmp_path: Path
) -> None:
    corrupted = tmp_path / "corrupted.nii.gz"
    corrupted.write_bytes(b"this is not a real nifti file")
    with pytest.raises(DataValidationError):
        engine.predict(corrupted)


def test_timeout_raises_inference_error(checkpoint: tuple[Path, Path], image: Path) -> None:
    checkpoint_path, preprocessing_path = checkpoint
    engine = InferenceEngine(checkpoint_path, preprocessing_path, timeout_seconds=1e-6)
    try:
        with pytest.raises(InferenceError, match="timed out"):
            engine.predict(image)
    finally:
        engine.shutdown()


def test_construction_rejects_non_positive_timeout(checkpoint: tuple[Path, Path]) -> None:
    checkpoint_path, preprocessing_path = checkpoint
    with pytest.raises(ConfigurationError, match="timeout_seconds"):
        InferenceEngine(checkpoint_path, preprocessing_path, timeout_seconds=0.0)


def test_no_gradients_are_tracked_during_inference(engine: InferenceEngine, image: Path) -> None:
    """'Use inference mode/no gradients': every model parameter's .grad stays
    untouched by a predict() call."""
    for param in engine._cached.model.parameters():
        assert param.grad is None
    engine.predict(image)
    for param in engine._cached.model.parameters():
        assert param.grad is None


def test_model_never_leaves_eval_mode(engine: InferenceEngine, image: Path) -> None:
    """'Do not retrain during inference': .training stays False across calls."""
    assert engine._cached.model.training is False
    engine.predict(image)
    assert engine._cached.model.training is False


def test_preprocessing_matches_build_transforms_directly(
    engine: InferenceEngine, image: Path
) -> None:
    """The engine's internal pipeline produces the exact same preprocessed
    tensor (and therefore the exact same logits) as calling
    datasets.transforms.build_transforms directly — no second, inference-only
    preprocessing implementation."""
    predicted_index, probabilities, image_stats = engine._run_pipeline(image)

    preprocessing_config = engine._cached.preprocessing_config
    transform = build_transforms(preprocessing_config, train=False)
    volume = apply_transforms(transform, {"image": str(image)})["image"]
    with torch.inference_mode():
        logits = engine._cached.model(volume.unsqueeze(0))
        direct_probabilities = torch.softmax(logits, dim=-1)[0].tolist()

    assert probabilities == pytest.approx(direct_probabilities, abs=1e-6)
    assert predicted_index == int(torch.argmax(torch.tensor(direct_probabilities)))
    assert image_stats.voxel_count == volume.numel()


def test_temperature_calibration_changes_probabilities_not_the_prediction(
    checkpoint: tuple[Path, Path], image: Path
) -> None:
    checkpoint_path, preprocessing_path = checkpoint
    uncalibrated_engine = InferenceEngine(checkpoint_path, preprocessing_path)
    calibrated_engine = InferenceEngine(
        checkpoint_path,
        preprocessing_path,
        calibration_config=CalibrationConfig(method="temperature", temperature=5.0),
    )
    try:
        uncalibrated = uncalibrated_engine.predict(image)
        calibrated = calibrated_engine.predict(image)
        assert uncalibrated.prediction == calibrated.prediction  # argmax is temperature-invariant
        assert uncalibrated.probabilities != calibrated.probabilities
        # a temperature > 1 flattens the distribution toward uniform
        assert calibrated.confidence < uncalibrated.confidence
    finally:
        uncalibrated_engine.shutdown()
        calibrated_engine.shutdown()


def test_model_version_info_matches_the_loaded_checkpoint(
    engine: InferenceEngine, checkpoint: tuple[Path, Path]
) -> None:
    from ad_mri.modeling.checkpoint import load_checkpoint

    checkpoint_path, _preprocessing_path = checkpoint
    _model, model_config, _metadata = load_checkpoint(checkpoint_path)

    info = engine.model_version_info()
    assert info.architecture == model_config.architecture
    assert info.model_version == model_config.version
    assert info.model_config_hash == model_config.compute_hash()
    assert info.calibration_method == "none"
    assert info.device == "cpu"


def test_health_check_reports_a_working_model(engine: InferenceEngine) -> None:
    status = engine.health_check(run_dummy_forward_pass=True)
    assert status["model_loaded"] is True
    assert status["forward_pass_ok"] is True


def test_health_check_can_skip_the_forward_pass(engine: InferenceEngine) -> None:
    status = engine.health_check(run_dummy_forward_pass=False)
    assert status["model_loaded"] is True
    assert status["forward_pass_ok"] is None


def test_unknown_explanation_target_class_raises(engine: InferenceEngine, image: Path) -> None:
    with pytest.raises(ConfigurationError, match="Unknown class"):
        engine.predict(image, include_explanation=True, explanation_target_class="not-a-real-class")


def test_explanation_available_when_requested(
    engine: InferenceEngine, image: Path, tmp_path: Path
) -> None:
    pytest.importorskip("lime")
    result = engine.predict(
        image, include_explanation=True, explanation_methods=("lime",), request_id="explain-test-1"
    )
    assert result.explanation_available is True
    assert result.explanation_id == "explain-test-1"
    manifest_path = tmp_path / "explanations" / "explain-test-1" / "manifest.json"
    assert manifest_path.exists()


def test_explanation_failure_does_not_fail_the_prediction(
    checkpoint: tuple[Path, Path], image: Path, tmp_path: Path
) -> None:
    """A best-effort explanation: requesting an impossible method combination
    (shap with no configured background manifest) must not take down an
    otherwise-valid prediction."""
    checkpoint_path, preprocessing_path = checkpoint
    engine = InferenceEngine(  # no background manifest configured
        checkpoint_path, preprocessing_path, explanation_output_dir=tmp_path / "explanations"
    )
    try:
        result = engine.predict(
            image,
            include_explanation=True,
            explanation_methods=("shap",),
            request_id="explain-fail-1",
        )
        assert result.prediction in {"AD", "CN"}  # the prediction itself is still valid
        assert result.explanation_available is False
        assert result.explanation_id is None
    finally:
        engine.shutdown()


def test_generate_explanation_is_loud_on_failure(
    checkpoint: tuple[Path, Path], image: Path, tmp_path: Path
) -> None:
    """Unlike predict(include_explanation=True) (best-effort),
    generate_explanation() is the dedicated, loud path (backs POST
    /explanations): the same impossible request (shap, no background
    manifest configured) must raise, not silently report "unavailable"."""
    from ad_mri.core.exceptions import ConfigurationError

    checkpoint_path, preprocessing_path = checkpoint
    engine = InferenceEngine(
        checkpoint_path, preprocessing_path, explanation_output_dir=tmp_path / "explanations"
    )
    try:
        with pytest.raises(ConfigurationError):
            engine.generate_explanation(image, methods=("shap",))
    finally:
        engine.shutdown()


def test_generate_explanation_succeeds_and_returns_a_real_bundle(
    engine: InferenceEngine, image: Path
) -> None:
    pytest.importorskip("lime")
    bundle = engine.generate_explanation(image, request_id="bundle-test-1", methods=("lime",))
    assert bundle.input_id == "bundle-test-1"
    assert bundle.lime is not None
    assert bundle.shap is None
    assert bundle.prediction.predicted_class in {"AD", "CN"}


def test_generate_explanation_rejects_an_unknown_target_class(
    engine: InferenceEngine, image: Path
) -> None:
    with pytest.raises(ConfigurationError, match="Unknown class"):
        engine.generate_explanation(image, target_class="not-a-real-class")


def test_shared_model_loader_caches_across_engine_instances(checkpoint: tuple[Path, Path]) -> None:
    checkpoint_path, preprocessing_path = checkpoint
    shared_loader = ModelLoader()
    engine_a = InferenceEngine(checkpoint_path, preprocessing_path, model_loader=shared_loader)
    engine_b = InferenceEngine(checkpoint_path, preprocessing_path, model_loader=shared_loader)
    try:
        assert engine_a._cached is engine_b._cached
    finally:
        engine_a.shutdown()
        engine_b.shutdown()


def test_redact_image_reference_is_stable_and_does_not_leak_the_filename(tmp_path: Path) -> None:
    path = tmp_path / "john_doe_1957.nii.gz"
    path.write_bytes(b"x")
    ref_a = redact_image_reference(path)
    ref_b = redact_image_reference(path)
    assert ref_a == ref_b
    assert "john_doe" not in ref_a
    assert "1957" not in ref_a
    assert len(ref_a) == 16


def test_redact_image_reference_differs_for_different_paths(tmp_path: Path) -> None:
    a = tmp_path / "a.nii.gz"
    b = tmp_path / "b.nii.gz"
    assert redact_image_reference(a) != redact_image_reference(b)
