"""Integration test for training.hpo.run_hyperparameter_search — a real,
tiny Optuna study (no mocks), each trial scored by real cross-validation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")
pytest.importorskip("sklearn")
pytest.importorskip("scipy")
pytest.importorskip("mlflow")
pytest.importorskip("optuna")

from ad_mri.core.exceptions import ConfigurationError  # noqa: E402
from ad_mri.data_pipeline.cross_validation import create_cross_validation_folds  # noqa: E402
from ad_mri.datasets.preprocessing_config import PreprocessingConfig  # noqa: E402
from ad_mri.modeling.model_config import ModelConfig  # noqa: E402
from ad_mri.training.hpo import run_hyperparameter_search  # noqa: E402
from ad_mri.training.hpo_config import HPOConfig, HPOSearchSpace  # noqa: E402
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


def _training_config(tmp_path: Path) -> TrainingConfig:
    return TrainingConfig(
        batch_size=2,
        epochs=1,
        num_workers=0,
        scheduler="none",
        mixed_precision=False,
        class_imbalance_strategy="none",
        mlflow_enabled=False,
        checkpoint_dir=tmp_path / "checkpoints",
        early_stopping_patience=100,
    )


@pytest.fixture
def cv_folds(tmp_path: Path) -> list[dict[str, Path]]:
    manifest = write_synthetic_manifest(tmp_path / "pool", num_per_class=4, shape=(16, 16, 16))
    return create_cross_validation_folds([manifest], tmp_path / "cv", n_folds=2, seed=42)


def _tiny_search_space(**overrides: object) -> HPOSearchSpace:
    # Restricted to keep the test fast and deterministic: only base_channels=4
    # (matches the tiny base model) and num_blocks=1 (only 1 pooling stage,
    # always valid for a 16^3 input) unless a test overrides it.
    defaults: dict[str, object] = {
        "learning_rate": (1e-3, 1e-2),
        "batch_size": [2],
        "optimizer": ["adam"],
        "weight_decay": (1e-5, 1e-3),
        "dropout_fc": (0.1, 0.3),
        "base_channels": [4],
        "num_blocks": [1],
    }
    defaults.update(overrides)
    return HPOSearchSpace(**defaults)  # type: ignore[arg-type]


def test_run_hyperparameter_search_completes_all_trials(
    tmp_path: Path, cv_folds: list[dict[str, Path]]
) -> None:
    hpo_config = HPOConfig(
        study_name="test-study",
        n_trials=2,
        n_folds=2,
        epochs_per_trial=1,
        seed=42,
        metric_name="accuracy",
        metric_mode="max",
        search_space=_tiny_search_space(),
        mlflow_enabled=False,
    )

    result = run_hyperparameter_search(
        hpo_config=hpo_config,
        base_model_config=_model_config(),
        base_training_config=_training_config(tmp_path),
        folds=cv_folds,
        preprocessing_config=_preprocessing_config(),
    )

    assert len(result.trials) == 2
    assert result.best_trial_number in {t.trial_number for t in result.trials}


def test_best_trial_is_actually_the_best_scoring_one(
    tmp_path: Path, cv_folds: list[dict[str, Path]]
) -> None:
    hpo_config = HPOConfig(
        study_name="test-study-2",
        n_trials=3,
        n_folds=2,
        epochs_per_trial=1,
        seed=1,
        metric_name="accuracy",
        metric_mode="max",
        search_space=_tiny_search_space(),
        mlflow_enabled=False,
    )

    result = run_hyperparameter_search(
        hpo_config=hpo_config,
        base_model_config=_model_config(),
        base_training_config=_training_config(tmp_path),
        folds=cv_folds,
        preprocessing_config=_preprocessing_config(),
    )

    manual_best = max(result.trials, key=lambda t: t.metric_value)
    assert result.best_trial_number == manual_best.trial_number
    assert result.best_metric_value == pytest.approx(manual_best.metric_value)


def test_best_configs_reflect_the_winning_trials_sampled_params(
    tmp_path: Path, cv_folds: list[dict[str, Path]]
) -> None:
    hpo_config = HPOConfig(
        study_name="test-study-3",
        n_trials=2,
        n_folds=2,
        epochs_per_trial=1,
        seed=42,
        metric_name="accuracy",
        metric_mode="max",
        # vary learning rate/optimizer only, to check they propagate through
        # to the returned best_training_config
        search_space=_tiny_search_space(optimizer=["adam", "sgd"]),
        mlflow_enabled=False,
    )

    result = run_hyperparameter_search(
        hpo_config=hpo_config,
        base_model_config=_model_config(),
        base_training_config=_training_config(tmp_path),
        folds=cv_folds,
        preprocessing_config=_preprocessing_config(),
    )

    best_trial = next(t for t in result.trials if t.trial_number == result.best_trial_number)
    assert result.best_training_config.learning_rate == pytest.approx(
        best_trial.params["learning_rate"]
    )
    assert result.best_training_config.optimizer == best_trial.params["optimizer"]
    # the final config keeps the *base* full epoch count, not epochs_per_trial
    assert result.best_training_config.epochs == 1  # base config's epochs in this test


def test_invalid_sampled_architecture_is_caught_not_fatal(
    tmp_path: Path, cv_folds: list[dict[str, Path]]
) -> None:
    """num_blocks=4 needs input axis >= 2**4=16, which our 16^3 input barely
    satisfies (valid) — but num_blocks=5 needs >=32, which is NOT satisfiable
    and must fail ModelConfig validation. Optuna's `catch` should mark that
    trial failed and continue the study, not abort it."""
    hpo_config = HPOConfig(
        study_name="test-study-invalid",
        n_trials=4,
        n_folds=2,
        epochs_per_trial=1,
        seed=42,
        metric_name="accuracy",
        metric_mode="max",
        search_space=_tiny_search_space(num_blocks=[1, 5]),  # 5 is invalid for a 16^3 input
        mlflow_enabled=False,
    )

    result = run_hyperparameter_search(
        hpo_config=hpo_config,
        base_model_config=_model_config(),
        base_training_config=_training_config(tmp_path),
        folds=cv_folds,
        preprocessing_config=_preprocessing_config(),
    )

    # at least one trial must have failed (num_blocks=5) without aborting the study
    assert len(result.trials) < 4
    assert len(result.trials) > 0
    assert all(t.params["num_blocks"] != 5 for t in result.trials)


def test_all_trials_failing_raises_configuration_error(
    tmp_path: Path, cv_folds: list[dict[str, Path]]
) -> None:
    hpo_config = HPOConfig(
        study_name="test-study-all-invalid",
        n_trials=2,
        n_folds=2,
        epochs_per_trial=1,
        seed=42,
        metric_name="accuracy",
        metric_mode="max",
        search_space=_tiny_search_space(num_blocks=[5]),  # always invalid for 16^3
        mlflow_enabled=False,
    )

    with pytest.raises(ConfigurationError):
        run_hyperparameter_search(
            hpo_config=hpo_config,
            base_model_config=_model_config(),
            base_training_config=_training_config(tmp_path),
            folds=cv_folds,
            preprocessing_config=_preprocessing_config(),
        )


def test_fold_count_mismatch_raises_configuration_error(
    tmp_path: Path, cv_folds: list[dict[str, Path]]
) -> None:
    hpo_config = HPOConfig(n_trials=1, n_folds=5, mlflow_enabled=False)  # cv_folds has only 2
    with pytest.raises(ConfigurationError):
        run_hyperparameter_search(
            hpo_config=hpo_config,
            base_model_config=_model_config(),
            base_training_config=_training_config(tmp_path),
            folds=cv_folds,
            preprocessing_config=_preprocessing_config(),
        )
