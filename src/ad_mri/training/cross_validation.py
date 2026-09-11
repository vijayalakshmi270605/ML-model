"""K-fold cross-validation training: runs `Trainer.fit()` once per fold and
summarizes results (mean/std/CI) across folds for robust model selection.

Reuses `Trainer` unchanged — cross-validation is orchestration around the
Phase 5 training loop, not a second copy of it. See
docs/TECHNICAL_SPECIFICATION.md §5, §10, and docs/model_selection.md for the
train/val/cross-validation/test-set distinction this module exists to enforce.

`mlflow` is a training-phase dependency (requirements/train.txt), imported
lazily so this module stays importable without it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ad_mri.core.logging import get_logger
from ad_mri.datasets.preprocessing_config import PreprocessingConfig
from ad_mri.evaluation.statistics import MetricSummary, summarize_fold_metrics
from ad_mri.modeling.model_config import ModelConfig
from ad_mri.training.mlflow_utils import start_mlflow_run
from ad_mri.training.trainer import Trainer, TrainingResult
from ad_mri.training.training_config import TrainingConfig

logger = get_logger(__name__)


@dataclass
class FoldResult:
    fold_index: int
    train_manifest_path: Path
    val_manifest_path: Path
    final_val_metrics: dict[str, float]
    best_checkpoint_path: Path
    mlflow_run_id: str | None


@dataclass
class CrossValidationResult:
    fold_results: list[FoldResult] = field(default_factory=list)
    metric_summary: dict[str, MetricSummary] = field(default_factory=dict)
    mlflow_run_id: str | None = None


def run_cross_validation(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    folds: list[dict[str, Path]],
    preprocessing_config: PreprocessingConfig | None = None,
    run_name_prefix: str = "cv",
    confidence_level: float = 0.95,
    mlflow_parent_active: bool = False,
) -> CrossValidationResult:
    """Train one model per fold and summarize the resulting validation metrics.

    `folds` is the output of `data_pipeline.cross_validation.create_cross_validation_folds`
    — a list of `{"train": Path, "val": Path}`. The test set is never a
    parameter here; nothing in this function can touch it.

    `mlflow_parent_active=True` skips opening a new top-level MLflow run and
    nests every fold's run under whatever run is already active (used by
    `training.hpo`, where each trial's run is that parent).
    """
    mlflow_module, run_context = start_mlflow_run(
        enabled=training_config.mlflow_enabled,
        tracking_uri=training_config.mlflow_tracking_uri,
        experiment_name=training_config.mlflow_experiment_name,
        run_name=run_name_prefix,
        nested=mlflow_parent_active,
    )

    fold_results: list[FoldResult] = []
    with run_context as group_run:
        for fold_index, fold in enumerate(folds):
            fold_run_name = f"{run_name_prefix}-fold{fold_index}"
            trainer = Trainer(
                model_config=model_config,
                training_config=training_config,
                train_manifest_path=fold["train"],
                val_manifest_path=fold["val"],
                preprocessing_config=preprocessing_config,
                run_name=fold_run_name,
                # Nest under the group run opened above whenever MLflow is
                # enabled at all; Trainer's own start_mlflow_run() has the
                # same graceful-degradation fallback, so this is safe even if
                # the group run itself failed to start.
                mlflow_nested=training_config.mlflow_enabled,
            )
            result: TrainingResult = trainer.fit()
            fold_results.append(
                FoldResult(
                    fold_index=fold_index,
                    train_manifest_path=fold["train"],
                    val_manifest_path=fold["val"],
                    final_val_metrics=result.final_val_metrics,
                    best_checkpoint_path=result.best_checkpoint_path,
                    mlflow_run_id=result.mlflow_run_id,
                )
            )
            logger.info(
                "cv_fold_complete",
                fold=fold_index,
                val_metrics=result.final_val_metrics,
            )

        metric_summary = summarize_fold_metrics(
            [fr.final_val_metrics for fr in fold_results], confidence_level=confidence_level
        )
        _log_cv_summary(mlflow_module, metric_summary, len(fold_results))

        group_run_id = group_run.info.run_id if mlflow_module is not None and group_run else None

    logger.info(
        "cross_validation_complete",
        n_folds=len(fold_results),
        summary={name: round(s.mean, 4) for name, s in metric_summary.items()},
    )
    return CrossValidationResult(
        fold_results=fold_results, metric_summary=metric_summary, mlflow_run_id=group_run_id
    )


def _log_cv_summary(
    mlflow_module: Any, metric_summary: dict[str, MetricSummary], n_folds: int
) -> None:
    if mlflow_module is None:
        return
    try:
        mlflow_module.log_param("n_folds", n_folds)
        metrics: dict[str, float] = {"n_folds": float(n_folds)}
        for name, summary in metric_summary.items():
            metrics[f"{name}_mean"] = summary.mean
            if summary.std is not None:
                metrics[f"{name}_std"] = summary.std
            if summary.ci_lower is not None and summary.ci_upper is not None:
                metrics[f"{name}_ci_lower"] = summary.ci_lower
                metrics[f"{name}_ci_upper"] = summary.ci_upper
        mlflow_module.log_metrics(metrics)
    except Exception as exc:
        logger.warning("mlflow_log_cv_summary_failed", error=str(exc))
