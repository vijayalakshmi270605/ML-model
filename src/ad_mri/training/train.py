"""CLI entry point for model training.

Runnable directly:

    python -m ad_mri.training.train --model-config ... --training-config ... \\
        --train-manifest ... --val-manifest ...

or via the main CLI (`ad-mri train ...`, see `cli/main.py`) — both call
`run_training()` below, the single training entry point; there is no second,
independently-maintained copy of the argument-parsing/wiring logic. This
module contains only CLI plumbing (argument parsing, config loading, logging
setup) — the actual training logic lives in `training.trainer.Trainer`, and
nothing here is imported by `api/` or `inference/`.
"""

from __future__ import annotations

from pathlib import Path

import typer

from ad_mri.core.config import get_settings
from ad_mri.core.logging import configure_logging, get_logger
from ad_mri.datasets.preprocessing_config import load_preprocessing_config
from ad_mri.modeling.model_config import load_model_config
from ad_mri.training.trainer import Trainer, TrainingResult
from ad_mri.training.training_config import load_training_config

logger = get_logger(__name__)

app = typer.Typer(help="Train an ad-mri CNN classifier.")


def run_training(
    model_config_path: Path,
    training_config_path: Path,
    train_manifest_path: Path,
    val_manifest_path: Path,
    preprocessing_config_path: Path | None = None,
    run_name: str | None = None,
    resume_from: Path | None = None,
) -> TrainingResult:
    """Load configs and run one training job. The single training entry point."""
    settings = get_settings()
    configure_logging(settings)

    model_config = load_model_config(model_config_path)
    training_config = load_training_config(training_config_path)
    preprocessing_config = (
        load_preprocessing_config(preprocessing_config_path)
        if preprocessing_config_path is not None
        else None
    )

    if training_config.mlflow_tracking_uri is None and settings.mlflow_tracking_uri:
        training_config = training_config.model_copy(
            update={"mlflow_tracking_uri": settings.mlflow_tracking_uri}
        )

    logger.info(
        "run_training_invoked",
        model_config=str(model_config_path),
        training_config=str(training_config_path),
        train_manifest=str(train_manifest_path),
        val_manifest=str(val_manifest_path),
        resume_from=str(resume_from) if resume_from else None,
    )

    trainer = Trainer(
        model_config=model_config,
        training_config=training_config,
        train_manifest_path=train_manifest_path,
        val_manifest_path=val_manifest_path,
        preprocessing_config=preprocessing_config,
        run_name=run_name,
        resume_from=resume_from,
    )
    return trainer.fit()


@app.command()
def main(
    model_config: Path = typer.Option(
        ...,
        "--model-config",
        exists=True,
        dir_okay=False,
        help="Path to a configs/model/*.yaml file.",
    ),
    training_config: Path = typer.Option(
        ...,
        "--training-config",
        exists=True,
        dir_okay=False,
        help="Path to a configs/training/*.yaml file.",
    ),
    train_manifest: Path = typer.Option(
        ..., "--train-manifest", exists=True, dir_okay=False, help="data/splits/<name>/train.csv"
    ),
    val_manifest: Path = typer.Option(
        ..., "--val-manifest", exists=True, dir_okay=False, help="data/splits/<name>/val.csv"
    ),
    preprocessing_config: Path | None = typer.Option(
        None, "--preprocessing-config", help="Default: configs/preprocessing/default.yaml"
    ),
    run_name: str | None = typer.Option(None, "--run-name", help="Default: run-<timestamp>"),
    resume_from: Path | None = typer.Option(
        None,
        "--resume-from",
        help="Path to a checkpoint (best.pt/last.pt) to resume training from.",
    ),
) -> None:
    """Train a model. Runnable as `python -m ad_mri.training.train` or `ad-mri train`."""
    resolved_preprocessing_config = preprocessing_config or Path(
        "configs/preprocessing/default.yaml"
    )

    result = run_training(
        model_config_path=model_config,
        training_config_path=training_config,
        train_manifest_path=train_manifest,
        val_manifest_path=val_manifest,
        preprocessing_config_path=resolved_preprocessing_config,
        run_name=run_name,
        resume_from=resume_from,
    )

    typer.echo(
        f"Training complete: {result.epochs_run} epoch(s) run, best epoch {result.best_epoch}."
    )
    typer.echo(f"Best checkpoint: {result.best_checkpoint_path}")
    typer.echo(f"Last checkpoint: {result.last_checkpoint_path}")
    if result.training_curves_path:
        typer.echo(f"Training curves: {result.training_curves_path}")
    if result.confusion_matrix_path:
        typer.echo(f"Confusion matrix: {result.confusion_matrix_path}")
    if result.mlflow_run_id:
        typer.echo(f"MLflow run ID: {result.mlflow_run_id}")
    typer.echo(f"Final validation metrics: {result.final_val_metrics}")


if __name__ == "__main__":
    app()
