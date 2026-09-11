"""Hyperparameter search (Optuna), evaluated by cross-validation on the
development pool.

Every trial's sampled configuration goes through cross-validation
(`training.cross_validation.run_cross_validation`) — the same `Trainer` used
everywhere else — so a trial's score is a mean-over-folds metric, not a
single lucky/unlucky split. Nothing here ever sees the test set: `folds` is
built exclusively from the development pool (requirement: "do not optimize
against the test set"). See docs/TECHNICAL_SPECIFICATION.md §5, and
docs/model_selection.md.

`optuna`/`mlflow` are training-phase dependencies (requirements/train.txt),
imported lazily so this module stays importable without them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ad_mri.core.exceptions import ConfigurationError
from ad_mri.core.logging import get_logger
from ad_mri.datasets.preprocessing_config import PreprocessingConfig
from ad_mri.modeling.model_config import ModelConfig
from ad_mri.training.cross_validation import CrossValidationResult, run_cross_validation
from ad_mri.training.hpo_config import HPOConfig, HPOSearchSpace
from ad_mri.training.mlflow_utils import start_mlflow_run
from ad_mri.training.training_config import TrainingConfig

logger = get_logger(__name__)

_MODEL_PARAM_KEYS = ("base_channels", "num_blocks", "dropout_fc")
_TRAINING_PARAM_KEYS = ("learning_rate", "batch_size", "optimizer", "weight_decay")


@dataclass
class TrialResult:
    trial_number: int
    params: dict[str, Any]
    metric_value: float
    cv_result: CrossValidationResult


@dataclass
class HPOResult:
    study_name: str
    metric_name: str
    metric_mode: str
    best_trial_number: int
    best_metric_value: float
    best_model_config: ModelConfig
    best_training_config: TrainingConfig
    trials: list[TrialResult] = field(default_factory=list)
    mlflow_run_id: str | None = None


def _apply_overrides(config: Any, overrides: dict[str, Any]) -> Any:
    """Reconstruct `config` with `overrides` applied, going through full
    validation. `.model_copy(update=...)` skips re-validation, which would let
    an invalid sampled combination (e.g. `num_blocks` too deep for
    `input_shape`) silently through instead of failing loudly."""
    merged = {**config.model_dump(mode="python"), **overrides}
    return type(config)(**merged)


def _sample_hyperparameters(trial: Any, space: HPOSearchSpace) -> dict[str, Any]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", *space.learning_rate, log=True),
        "batch_size": trial.suggest_categorical("batch_size", space.batch_size),
        "optimizer": trial.suggest_categorical("optimizer", space.optimizer),
        "weight_decay": trial.suggest_float("weight_decay", *space.weight_decay, log=True),
        "dropout_fc": trial.suggest_float("dropout_fc", *space.dropout_fc),
        "base_channels": trial.suggest_categorical("base_channels", space.base_channels),
        "num_blocks": trial.suggest_categorical("num_blocks", space.num_blocks),
    }


def run_hyperparameter_search(
    hpo_config: HPOConfig,
    base_model_config: ModelConfig,
    base_training_config: TrainingConfig,
    folds: list[dict[str, Path]],
    preprocessing_config: PreprocessingConfig | None = None,
) -> HPOResult:
    """Run `hpo_config.n_trials` Optuna trials, each scored by
    `hpo_config.n_folds`-fold cross-validation on `folds`.

    Selects the trial whose mean CV `hpo_config.metric_name` is best per
    `hpo_config.metric_mode` — the "predefined validation metric" requirement.
    A trial whose sampled hyperparameters produce an invalid config (caught as
    `ConfigurationError`) is marked failed and skipped, not fatal to the study.

    Raises:
        ConfigurationError: `folds` doesn't match `hpo_config.n_folds`, the
            configured metric never appears in the CV results, or every trial
            failed (nothing to select from).
    """
    import optuna

    if len(folds) != hpo_config.n_folds:
        raise ConfigurationError(
            f"hpo_config.n_folds={hpo_config.n_folds} but {len(folds)} fold(s) were provided."
        )

    trials: list[TrialResult] = []

    def objective(trial: optuna.Trial) -> float:
        sampled = _sample_hyperparameters(trial, hpo_config.search_space)

        trial_model_config = _apply_overrides(
            base_model_config, {key: sampled[key] for key in _MODEL_PARAM_KEYS}
        )
        trial_training_config = _apply_overrides(
            base_training_config,
            {
                **{key: sampled[key] for key in _TRAINING_PARAM_KEYS},
                "epochs": hpo_config.epochs_per_trial,
                "mlflow_enabled": hpo_config.mlflow_enabled,
                "mlflow_experiment_name": hpo_config.mlflow_experiment_name,
                "mlflow_tracking_uri": hpo_config.mlflow_tracking_uri,
            },
        )

        cv_result = run_cross_validation(
            model_config=trial_model_config,
            training_config=trial_training_config,
            folds=folds,
            preprocessing_config=preprocessing_config,
            run_name_prefix=f"trial{trial.number}",
            mlflow_parent_active=True,
        )

        if hpo_config.metric_name not in cv_result.metric_summary:
            raise ConfigurationError(
                f"metric_name {hpo_config.metric_name!r} not found among CV metrics: "
                f"{sorted(cv_result.metric_summary)}"
            )
        metric_value = cv_result.metric_summary[hpo_config.metric_name].mean

        trials.append(
            TrialResult(
                trial_number=trial.number,
                params=dict(sampled),
                metric_value=metric_value,
                cv_result=cv_result,
            )
        )
        logger.info(
            "hpo_trial_complete", trial=trial.number, params=sampled, metric_value=metric_value
        )
        return metric_value

    direction: Literal["maximize", "minimize"] = (
        "maximize" if hpo_config.metric_mode == "max" else "minimize"
    )
    sampler = optuna.samplers.TPESampler(seed=hpo_config.seed)
    study = optuna.create_study(
        study_name=hpo_config.study_name, direction=direction, sampler=sampler
    )

    mlflow_module, run_context = start_mlflow_run(
        enabled=hpo_config.mlflow_enabled,
        tracking_uri=hpo_config.mlflow_tracking_uri,
        experiment_name=hpo_config.mlflow_experiment_name,
        run_name=hpo_config.study_name,
    )

    with run_context as study_run:
        study.optimize(objective, n_trials=hpo_config.n_trials, catch=(ConfigurationError,))

        if not trials:
            raise ConfigurationError(
                f"All {hpo_config.n_trials} HPO trial(s) failed; no model could be selected."
            )

        best = (
            max(trials, key=lambda t: t.metric_value)
            if direction == "maximize"
            else min(trials, key=lambda t: t.metric_value)
        )
        _log_study_summary(mlflow_module, trials, best)
        study_run_id = study_run.info.run_id if mlflow_module is not None and study_run else None

    best_model_config = _apply_overrides(
        base_model_config, {key: best.params[key] for key in _MODEL_PARAM_KEYS}
    )
    # The final selected training config keeps the *base* (full) epoch count —
    # HPO trials train short (epochs_per_trial) purely to rank configurations
    # quickly; the winner is retrained to full convergence separately.
    best_training_config = _apply_overrides(
        base_training_config, {key: best.params[key] for key in _TRAINING_PARAM_KEYS}
    )

    logger.info(
        "hpo_search_complete",
        n_trials_completed=len(trials),
        n_trials_requested=hpo_config.n_trials,
        best_trial=best.trial_number,
        best_metric_value=best.metric_value,
    )

    return HPOResult(
        study_name=hpo_config.study_name,
        metric_name=hpo_config.metric_name,
        metric_mode=hpo_config.metric_mode,
        best_trial_number=best.trial_number,
        best_metric_value=best.metric_value,
        best_model_config=best_model_config,
        best_training_config=best_training_config,
        trials=trials,
        mlflow_run_id=study_run_id,
    )


def _log_study_summary(mlflow_module: Any, trials: list[TrialResult], best: TrialResult) -> None:
    if mlflow_module is None:
        return
    try:
        mlflow_module.log_param("n_trials_completed", len(trials))
        mlflow_module.log_param("best_trial_number", best.trial_number)
        mlflow_module.log_metric("best_metric_value", best.metric_value)
    except Exception as exc:
        logger.warning("mlflow_log_hpo_summary_failed", error=str(exc))
