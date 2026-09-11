"""Integration tests for training.trainer.Trainer — real, tiny end-to-end
training runs (no mocks): synthetic data -> real forward/backward passes ->
real checkpoints -> real (local-file) MLflow tracking.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")
pytest.importorskip("sklearn")
pytest.importorskip("mlflow")

from ad_mri.core.exceptions import ConfigurationError, TrainingError  # noqa: E402
from ad_mri.datasets.preprocessing_config import PreprocessingConfig  # noqa: E402
from ad_mri.modeling.checkpoint import load_checkpoint  # noqa: E402
from ad_mri.modeling.factory import build_model  # noqa: E402
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
def dataset(tmp_path: Path) -> tuple[Path, Path]:
    train_manifest = write_synthetic_manifest(
        tmp_path / "train", num_per_class=3, shape=(16, 16, 16)
    )
    val_manifest = write_synthetic_manifest(tmp_path / "val", num_per_class=2, shape=(16, 16, 16))
    return train_manifest, val_manifest


def test_fit_runs_end_to_end_and_produces_checkpoints(
    tmp_path: Path, dataset: tuple[Path, Path]
) -> None:
    train_manifest, val_manifest = dataset
    trainer = Trainer(
        model_config=_model_config(),
        training_config=_training_config(tmp_path),
        train_manifest_path=train_manifest,
        val_manifest_path=val_manifest,
        preprocessing_config=_preprocessing_config(),
        run_name="test-run",
    )

    result = trainer.fit()

    assert result.epochs_run == 2
    assert len(result.history) == 2
    assert result.best_checkpoint_path.exists()
    assert result.last_checkpoint_path.exists()
    assert "loss" in result.final_train_metrics
    assert "loss" in result.final_val_metrics
    assert "accuracy" in result.final_val_metrics


def test_history_rows_are_in_epoch_order(tmp_path: Path, dataset: tuple[Path, Path]) -> None:
    train_manifest, val_manifest = dataset
    trainer = Trainer(
        model_config=_model_config(),
        training_config=_training_config(tmp_path, epochs=3),
        train_manifest_path=train_manifest,
        val_manifest_path=val_manifest,
        preprocessing_config=_preprocessing_config(),
    )
    result = trainer.fit()
    assert [row["epoch"] for row in result.history] == [1.0, 2.0, 3.0]


def test_num_classes_mismatch_raises_configuration_error(
    tmp_path: Path, dataset: tuple[Path, Path]
) -> None:
    train_manifest, val_manifest = dataset
    trainer = Trainer(
        model_config=_model_config(num_classes=5),  # dataset actually has 2 classes
        training_config=_training_config(tmp_path),
        train_manifest_path=train_manifest,
        val_manifest_path=val_manifest,
        preprocessing_config=_preprocessing_config(),
    )
    with pytest.raises(ConfigurationError):
        trainer.fit()


def test_mismatched_preprocessing_and_model_shapes_rejected_at_construction(
    tmp_path: Path, dataset: tuple[Path, Path]
) -> None:
    train_manifest, val_manifest = dataset
    with pytest.raises(ConfigurationError):
        Trainer(
            model_config=_model_config(input_shape=(16, 16, 16)),
            training_config=_training_config(tmp_path),
            train_manifest_path=train_manifest,
            val_manifest_path=val_manifest,
            preprocessing_config=_preprocessing_config(target_shape=(32, 32, 32)),
        )


@pytest.mark.parametrize("strategy", ["none", "sampler", "loss_weights"])
def test_every_class_imbalance_strategy_runs_without_error(
    tmp_path: Path, dataset: tuple[Path, Path], strategy: str
) -> None:
    train_manifest, val_manifest = dataset
    trainer = Trainer(
        model_config=_model_config(),
        training_config=_training_config(tmp_path, epochs=1, class_imbalance_strategy=strategy),
        train_manifest_path=train_manifest,
        val_manifest_path=val_manifest,
        preprocessing_config=_preprocessing_config(),
    )
    result = trainer.fit()
    assert result.epochs_run == 1


@pytest.mark.parametrize("scheduler", ["none", "step", "cosine", "plateau"])
def test_every_scheduler_runs_without_error(
    tmp_path: Path, dataset: tuple[Path, Path], scheduler: str
) -> None:
    train_manifest, val_manifest = dataset
    trainer = Trainer(
        model_config=_model_config(),
        training_config=_training_config(tmp_path, epochs=2, scheduler=scheduler),
        train_manifest_path=train_manifest,
        val_manifest_path=val_manifest,
        preprocessing_config=_preprocessing_config(),
    )
    result = trainer.fit()
    assert result.epochs_run == 2


def test_resume_from_checkpoint_continues_from_the_correct_epoch(
    tmp_path: Path, dataset: tuple[Path, Path]
) -> None:
    train_manifest, val_manifest = dataset
    model_config = _model_config()

    first_trainer = Trainer(
        model_config=model_config,
        training_config=_training_config(tmp_path, epochs=1),
        train_manifest_path=train_manifest,
        val_manifest_path=val_manifest,
        preprocessing_config=_preprocessing_config(),
        run_name="resume-test",
    )
    first_result = first_trainer.fit()
    assert first_result.epochs_run == 1

    second_trainer = Trainer(
        model_config=model_config,
        training_config=_training_config(tmp_path, epochs=3),
        train_manifest_path=train_manifest,
        val_manifest_path=val_manifest,
        preprocessing_config=_preprocessing_config(),
        resume_from=first_result.last_checkpoint_path,
    )
    second_result = second_trainer.fit()

    assert second_result.epochs_run == 3
    assert [row["epoch"] for row in second_result.history] == [1.0, 2.0, 3.0]
    # continued writing into the same run directory (no run_name override)
    assert second_trainer.run_dir == first_trainer.run_dir


def test_resume_rejects_a_checkpoint_from_a_different_architecture(
    tmp_path: Path, dataset: tuple[Path, Path]
) -> None:
    train_manifest, val_manifest = dataset
    first_trainer = Trainer(
        model_config=_model_config(base_channels=4),
        training_config=_training_config(tmp_path, epochs=1),
        train_manifest_path=train_manifest,
        val_manifest_path=val_manifest,
        preprocessing_config=_preprocessing_config(),
        run_name="arch-mismatch",
    )
    first_result = first_trainer.fit()

    second_trainer = Trainer(
        model_config=_model_config(base_channels=8),  # different architecture
        training_config=_training_config(tmp_path, epochs=2),
        train_manifest_path=train_manifest,
        val_manifest_path=val_manifest,
        preprocessing_config=_preprocessing_config(),
        resume_from=first_result.last_checkpoint_path,
    )
    with pytest.raises(ConfigurationError):
        second_trainer.fit()


def test_checkpoint_bundles_full_training_provenance(
    tmp_path: Path, dataset: tuple[Path, Path]
) -> None:
    train_manifest, val_manifest = dataset
    preprocessing_config = _preprocessing_config()
    trainer = Trainer(
        model_config=_model_config(),
        training_config=_training_config(tmp_path, epochs=1, scheduler="step"),
        train_manifest_path=train_manifest,
        val_manifest_path=val_manifest,
        preprocessing_config=preprocessing_config,
    )
    result = trainer.fit()

    payload = torch.load(result.last_checkpoint_path, weights_only=False)
    assert payload["model_config_hash"]
    extra = payload["extra"]
    assert extra["optimizer_state_dict"] is not None
    assert extra["scheduler_state_dict"] is not None  # scheduler="step"
    assert extra["preprocessing_config_hash"] == preprocessing_config.compute_hash()
    assert extra["preprocessing_version"] == preprocessing_config.version
    assert extra["train_manifest_sha256"]
    assert extra["val_manifest_sha256"]
    assert extra["class_names"] == ["AD", "CN"]
    assert len(extra["history"]) == 1

    # also verify it loads back into a working model via the public API
    _model, model_config, metadata = load_checkpoint(result.last_checkpoint_path)
    assert model_config.compute_hash() == payload["model_config_hash"]
    assert metadata.extra["train_manifest_sha256"] == extra["train_manifest_sha256"]


def test_nan_loss_raises_training_error(tmp_path: Path, dataset: tuple[Path, Path]) -> None:
    train_manifest, val_manifest = dataset
    trainer = Trainer(
        model_config=_model_config(),
        training_config=_training_config(tmp_path, epochs=1),
        train_manifest_path=train_manifest,
        val_manifest_path=val_manifest,
        preprocessing_config=_preprocessing_config(),
    )

    import pandas as pd

    train_df = pd.read_csv(train_manifest)
    val_df = pd.read_csv(val_manifest)
    label_to_index, class_names = trainer._resolve_label_mapping(train_df, val_df)
    trainer._class_names = class_names
    device = trainer._resolve_device()
    train_loader, _ = trainer._build_dataloaders(train_df, label_to_index)
    model = build_model(trainer.model_config).to(device)
    optimizer = trainer._build_optimizer(model)
    scaler = trainer._build_grad_scaler(device)

    def _nan_loss(_logits: object, _labels: object) -> object:
        return torch.tensor(float("nan"))

    with pytest.raises(TrainingError):
        trainer._run_epoch(
            model, train_loader, _nan_loss, optimizer, scaler, device, class_names, training=True
        )


def test_bad_sample_is_skipped_not_fatal(tmp_path: Path, dataset: tuple[Path, Path]) -> None:
    """A corrupted/unreadable scan in the training set must not crash the whole epoch."""
    train_manifest, val_manifest = dataset

    import pandas as pd

    df = pd.read_csv(train_manifest)
    corrupted_path = tmp_path / "corrupted.nii.gz"
    corrupted_path.write_bytes(b"not a real nifti file")
    df.loc[0, "image_path"] = str(corrupted_path)
    df.to_csv(train_manifest, index=False)

    trainer = Trainer(
        model_config=_model_config(),
        training_config=_training_config(tmp_path, epochs=1),
        train_manifest_path=train_manifest,
        val_manifest_path=val_manifest,
        preprocessing_config=_preprocessing_config(),
    )
    result = trainer.fit()  # must not raise despite one bad sample
    assert result.epochs_run == 1


def test_mlflow_run_is_created_with_params_metrics_and_artifacts(
    tmp_path: Path, dataset: tuple[Path, Path]
) -> None:
    from mlflow.tracking import MlflowClient

    train_manifest, val_manifest = dataset
    tracking_uri = f"file:{tmp_path / 'mlruns'}"
    trainer = Trainer(
        model_config=_model_config(),
        training_config=_training_config(
            tmp_path,
            epochs=1,
            mlflow_enabled=True,
            mlflow_tracking_uri=tracking_uri,
            mlflow_experiment_name="test-experiment",
        ),
        train_manifest_path=train_manifest,
        val_manifest_path=val_manifest,
        preprocessing_config=_preprocessing_config(),
    )

    result = trainer.fit()
    assert result.mlflow_run_id is not None

    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(result.mlflow_run_id)

    assert run.data.params.get("training.batch_size") == "2"
    assert "model_config_hash" in run.data.params
    assert "train_loss" in run.data.metrics
    assert "val_accuracy" in run.data.metrics

    artifact_paths = {f.path for f in client.list_artifacts(result.mlflow_run_id)}
    assert "best.pt" in artifact_paths
    assert "last.pt" in artifact_paths
    assert "training_curves.png" in artifact_paths


def test_training_proceeds_when_mlflow_tracking_uri_is_unreachable(
    tmp_path: Path, dataset: tuple[Path, Path]
) -> None:
    """MLflow is supplementary tracking — an unreachable server must not abort training."""
    train_manifest, val_manifest = dataset
    trainer = Trainer(
        model_config=_model_config(),
        training_config=_training_config(
            tmp_path,
            epochs=1,
            mlflow_enabled=True,
            mlflow_tracking_uri="http://localhost:1/unreachable",
        ),
        train_manifest_path=train_manifest,
        val_manifest_path=val_manifest,
        preprocessing_config=_preprocessing_config(),
    )
    result = trainer.fit()  # must not raise
    assert result.epochs_run == 1
    assert result.best_checkpoint_path.exists()
