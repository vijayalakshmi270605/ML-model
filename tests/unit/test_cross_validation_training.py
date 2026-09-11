"""Integration test for training.cross_validation.run_cross_validation — a
real, tiny multi-fold training run (no mocks), reusing the exact Trainer
verified in Phase 5.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")
pytest.importorskip("sklearn")
pytest.importorskip("scipy")
pytest.importorskip("mlflow")

from ad_mri.data_pipeline.cross_validation import create_cross_validation_folds  # noqa: E402
from ad_mri.datasets.preprocessing_config import PreprocessingConfig  # noqa: E402
from ad_mri.modeling.model_config import ModelConfig  # noqa: E402
from ad_mri.training.cross_validation import run_cross_validation  # noqa: E402
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


def _preprocessing_config() -> PreprocessingConfig:
    return PreprocessingConfig(
        target_shape=(16, 16, 16), target_spacing=(1.0, 1.0, 1.0), min_voxel_count=1000
    )


def _training_config(tmp_path: Path, **overrides: object) -> TrainingConfig:
    defaults: dict[str, object] = {
        "batch_size": 2,
        "epochs": 1,
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
def cv_folds(tmp_path: Path) -> list[dict[str, Path]]:
    manifest = write_synthetic_manifest(tmp_path / "pool", num_per_class=4, shape=(16, 16, 16))
    # write_synthetic_manifest already writes one manifest for the whole pool;
    # cross-validation folding just needs it as a single-element manifest list.
    return create_cross_validation_folds([manifest], tmp_path / "cv", n_folds=2, seed=42)


def test_run_cross_validation_trains_one_model_per_fold(
    tmp_path: Path, cv_folds: list[dict[str, Path]]
) -> None:
    result = run_cross_validation(
        model_config=_model_config(),
        training_config=_training_config(tmp_path),
        folds=cv_folds,
        preprocessing_config=_preprocessing_config(),
        run_name_prefix="test-cv",
    )

    assert len(result.fold_results) == 2
    for fold in result.fold_results:
        assert fold.best_checkpoint_path.exists()
        assert "loss" in fold.final_val_metrics
        assert "accuracy" in fold.final_val_metrics


def test_run_cross_validation_summarizes_mean_std_ci(
    tmp_path: Path, cv_folds: list[dict[str, Path]]
) -> None:
    result = run_cross_validation(
        model_config=_model_config(),
        training_config=_training_config(tmp_path),
        folds=cv_folds,
        preprocessing_config=_preprocessing_config(),
    )

    assert "accuracy" in result.metric_summary
    summary = result.metric_summary["accuracy"]
    assert summary.n == 2
    # exact mean check: manually average the two folds' reported accuracy
    manual_mean = sum(fr.final_val_metrics["accuracy"] for fr in result.fold_results) / 2
    assert summary.mean == pytest.approx(manual_mean)
    # n=2 is enough for a std/CI (unlike n=1, tested elsewhere)
    assert summary.std is not None
    assert summary.ci_lower is not None and summary.ci_upper is not None
    assert summary.ci_lower <= summary.mean <= summary.ci_upper


def test_cross_validation_folds_never_reference_a_test_manifest(
    cv_folds: list[dict[str, Path]],
) -> None:
    """Structural guarantee: the fold dict only ever has train/val keys."""
    for fold in cv_folds:
        assert set(fold) == {"train", "val"}


def test_cross_validation_mlflow_nesting_produces_a_parent_and_fold_runs(
    tmp_path: Path, cv_folds: list[dict[str, Path]]
) -> None:
    from mlflow.tracking import MlflowClient

    tracking_uri = f"file:{tmp_path / 'mlruns'}"
    training_config = _training_config(
        tmp_path,
        mlflow_enabled=True,
        mlflow_tracking_uri=tracking_uri,
        mlflow_experiment_name="test-cv-experiment",
    )

    result = run_cross_validation(
        model_config=_model_config(),
        training_config=training_config,
        folds=cv_folds,
        preprocessing_config=_preprocessing_config(),
        run_name_prefix="nested-test",
    )

    assert result.mlflow_run_id is not None
    client = MlflowClient(tracking_uri=tracking_uri)

    parent_run = client.get_run(result.mlflow_run_id)
    assert parent_run.data.params.get("n_folds") == "2"
    assert "accuracy_mean" in parent_run.data.metrics

    child_runs = [fr for fr in result.fold_results if fr.mlflow_run_id is not None]
    assert len(child_runs) == 2
    for fold_result in child_runs:
        child_run = client.get_run(fold_result.mlflow_run_id)  # type: ignore[arg-type]
        assert child_run.data.tags.get("mlflow.parentRunId") == result.mlflow_run_id
