"""End-to-end test for evaluation.evaluate.run_evaluation — a real, tiny
Trainer run (no mocks) producing a real checkpoint, then a real standalone
evaluation pass over a held-out synthetic test manifest.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")
pytest.importorskip("sklearn")
pytest.importorskip("scipy")

from ad_mri.core.exceptions import ConfigurationError  # noqa: E402
from ad_mri.datasets.preprocessing_config import PreprocessingConfig  # noqa: E402
from ad_mri.evaluation.evaluate import run_evaluation  # noqa: E402
from ad_mri.modeling.model_config import ModelConfig  # noqa: E402
from ad_mri.training.trainer import Trainer  # noqa: E402
from ad_mri.training.training_config import TrainingConfig  # noqa: E402
from tests.helpers import write_synthetic_manifest  # noqa: E402


def _model_config(**overrides: object) -> ModelConfig:
    defaults: dict[str, object] = {
        "num_classes": 2,
        "spatial_dims": 3,
        "input_shape": (16, 16, 16),
        "num_blocks": 1,
        "base_channels": 4,
        "fc_hidden_dim": 8,
    }
    defaults.update(overrides)
    return ModelConfig(**defaults)  # type: ignore[arg-type]


def _preprocessing_config(**overrides: object) -> PreprocessingConfig:
    defaults: dict[str, object] = {
        "target_shape": (16, 16, 16),
        "target_spacing": (1.0, 1.0, 1.0),
        "min_voxel_count": 1000,
    }
    defaults.update(overrides)
    return PreprocessingConfig(**defaults)  # type: ignore[arg-type]


def _training_config(tmp_path: Path, **overrides: object) -> TrainingConfig:
    defaults: dict[str, object] = {
        "batch_size": 2,
        "epochs": 2,
        "num_workers": 0,
        "scheduler": "none",
        "mixed_precision": False,
        "class_imbalance_strategy": "none",
        "mlflow_enabled": False,
        "checkpoint_dir": tmp_path / "checkpoints",
        "early_stopping_patience": 100,
    }
    defaults.update(overrides)
    return TrainingConfig(**defaults)  # type: ignore[arg-type]


@pytest.fixture
def trained_checkpoint(tmp_path: Path) -> tuple[Path, PreprocessingConfig]:
    """Train a real (tiny) model and return its best checkpoint path."""
    train_manifest = write_synthetic_manifest(
        tmp_path / "train", num_per_class=4, shape=(16, 16, 16)
    )
    val_manifest = write_synthetic_manifest(tmp_path / "val", num_per_class=2, shape=(16, 16, 16))
    preprocessing_config = _preprocessing_config()

    trainer = Trainer(
        model_config=_model_config(),
        training_config=_training_config(tmp_path),
        train_manifest_path=train_manifest,
        val_manifest_path=val_manifest,
        preprocessing_config=preprocessing_config,
        run_name="eval-test-run",
    )
    result = trainer.fit()
    return result.best_checkpoint_path, preprocessing_config


def _write_preprocessing_yaml(path: Path, config: PreprocessingConfig) -> Path:
    from ad_mri.datasets.preprocessing_config import save_preprocessing_config

    return save_preprocessing_config(config, path)


def test_run_evaluation_produces_a_full_report(
    tmp_path: Path, trained_checkpoint: tuple[Path, PreprocessingConfig]
) -> None:
    checkpoint_path, preprocessing_config = trained_checkpoint
    preprocessing_path = _write_preprocessing_yaml(
        tmp_path / "preprocessing.yaml", preprocessing_config
    )
    test_manifest = write_synthetic_manifest(tmp_path / "test", num_per_class=3, shape=(16, 16, 16))

    result = run_evaluation(
        model_path=checkpoint_path,
        dataset_manifest_path=test_manifest,
        preprocessing_config_path=preprocessing_path,
        output_dir=tmp_path / "eval_out",
        n_bootstrap=50,
    )

    assert result.report.n_samples == 6
    assert result.report.class_names == ["AD", "CN"]
    assert 0.0 <= result.report.metrics["accuracy"] <= 1.0
    assert "f1_macro" in result.report.metrics
    assert result.output_dir == tmp_path / "eval_out"
    for key in (
        "confusion_matrix",
        "json",
        "csv",
        "predictions",
        "markdown",
        "classification_report",
    ):
        assert result.written[key].exists()


def test_run_evaluation_rejects_mismatched_preprocessing(
    tmp_path: Path, trained_checkpoint: tuple[Path, PreprocessingConfig]
) -> None:
    checkpoint_path, _correct_preprocessing = trained_checkpoint
    wrong_preprocessing = _preprocessing_config(target_spacing=(2.0, 2.0, 2.0))
    wrong_path = _write_preprocessing_yaml(
        tmp_path / "wrong_preprocessing.yaml", wrong_preprocessing
    )
    test_manifest = write_synthetic_manifest(tmp_path / "test", num_per_class=2, shape=(16, 16, 16))

    with pytest.raises(ConfigurationError, match="does not match"):
        run_evaluation(
            model_path=checkpoint_path,
            dataset_manifest_path=test_manifest,
            preprocessing_config_path=wrong_path,
        )


def test_run_evaluation_rejects_threshold_for_multiclass(tmp_path: Path) -> None:
    train_manifest = write_synthetic_manifest(
        tmp_path / "train", classes=("CN", "MCI", "AD"), num_per_class=3, shape=(16, 16, 16)
    )
    val_manifest = write_synthetic_manifest(
        tmp_path / "val", classes=("CN", "MCI", "AD"), num_per_class=2, shape=(16, 16, 16)
    )
    preprocessing_config = _preprocessing_config()
    trainer = Trainer(
        model_config=_model_config(num_classes=3),
        training_config=_training_config(tmp_path, epochs=1),
        train_manifest_path=train_manifest,
        val_manifest_path=val_manifest,
        preprocessing_config=preprocessing_config,
    )
    result = trainer.fit()
    preprocessing_path = _write_preprocessing_yaml(
        tmp_path / "preprocessing.yaml", preprocessing_config
    )
    test_manifest = write_synthetic_manifest(
        tmp_path / "test", classes=("CN", "MCI", "AD"), num_per_class=2, shape=(16, 16, 16)
    )

    with pytest.raises(ConfigurationError, match="binary classification"):
        run_evaluation(
            model_path=result.best_checkpoint_path,
            dataset_manifest_path=test_manifest,
            preprocessing_config_path=preprocessing_path,
            threshold=0.6,
        )


def test_binary_threshold_changes_predictions(
    tmp_path: Path, trained_checkpoint: tuple[Path, PreprocessingConfig]
) -> None:
    checkpoint_path, preprocessing_config = trained_checkpoint
    preprocessing_path = _write_preprocessing_yaml(
        tmp_path / "preprocessing.yaml", preprocessing_config
    )
    test_manifest = write_synthetic_manifest(tmp_path / "test", num_per_class=4, shape=(16, 16, 16))

    default_result = run_evaluation(
        model_path=checkpoint_path,
        dataset_manifest_path=test_manifest,
        preprocessing_config_path=preprocessing_path,
        output_dir=tmp_path / "eval_default",
        n_bootstrap=20,
    )
    extreme_result = run_evaluation(
        model_path=checkpoint_path,
        dataset_manifest_path=test_manifest,
        preprocessing_config_path=preprocessing_path,
        output_dir=tmp_path / "eval_extreme",
        threshold=0.999999,  # forces every prediction to the negative class
        n_bootstrap=20,
    )
    assert default_result.report.threshold is None
    assert extreme_result.report.threshold == pytest.approx(0.999999)
    predicted_labels = {row["predicted_label"] for row in extreme_result.report.predictions}
    assert predicted_labels == {extreme_result.report.class_names[0]}


def test_checkpoint_without_class_names_is_rejected(tmp_path: Path) -> None:
    """A checkpoint saved without the Trainer's provenance metadata (e.g. hand-built)
    must fail loudly rather than silently guessing a label mapping."""
    from ad_mri.modeling.checkpoint import save_checkpoint
    from ad_mri.modeling.factory import build_model

    model_config = _model_config()
    model = build_model(model_config)
    checkpoint_path = tmp_path / "bare.pt"
    save_checkpoint(model, model_config, checkpoint_path, extra={})  # no class_names

    test_manifest = write_synthetic_manifest(tmp_path / "test", num_per_class=2, shape=(16, 16, 16))
    preprocessing_path = _write_preprocessing_yaml(
        tmp_path / "preprocessing.yaml", _preprocessing_config()
    )

    with pytest.raises(ConfigurationError, match="class_names"):
        run_evaluation(
            model_path=checkpoint_path,
            dataset_manifest_path=test_manifest,
            preprocessing_config_path=preprocessing_path,
        )
