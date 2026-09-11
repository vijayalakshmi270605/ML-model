"""ad-mri command-line interface.

`serve`, `data` (ingest/validate/manifest/split/cross-validate), `model`,
`cv` (run/search), `train`, `evaluate`, `explain` (run/global-shap), and
`predict` are fully functional. The remaining offline pipeline commands
(`preprocess`/`register-model`) are stubs that report which roadmap phase
implements them, rather than failing silently.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
import uvicorn

from ad_mri.core.config import get_settings
from ad_mri.core.disclaimers import PREDICTION_DISCLAIMER
from ad_mri.core.logging import configure_logging, get_logger
from ad_mri.data_pipeline.cross_validation import create_cross_validation_folds
from ad_mri.data_pipeline.dataset_config import load_dataset_config
from ad_mri.data_pipeline.ingestion import ingest_raw_scans
from ad_mri.data_pipeline.manifest import (
    build_manifest,
    summarize_class_distribution,
    write_manifest,
)
from ad_mri.data_pipeline.schema import RowError
from ad_mri.data_pipeline.splitting import create_subject_level_split
from ad_mri.data_pipeline.validation import validate_inventory
from ad_mri.datasets.preprocessing_config import load_preprocessing_config
from ad_mri.datasets.visualization import plot_preprocessing_sample
from ad_mri.modeling.model_config import load_model_config

app = typer.Typer(help="ad-mri: Alzheimer's Disease MRI classification system CLI.")
data_app = typer.Typer(help="Dataset ingestion, validation, manifest, and split commands.")
model_app = typer.Typer(help="Model architecture inspection commands.")
cv_app = typer.Typer(help="Cross-validation and hyperparameter-search commands.")
explain_app = typer.Typer(help="LIME/SHAP explanation commands.")
monitor_app = typer.Typer(help="Data/model drift reports and delayed model-performance monitoring.")
app.add_typer(data_app, name="data")
app.add_typer(model_app, name="model")
app.add_typer(cv_app, name="cv")
app.add_typer(explain_app, name="explain")
app.add_typer(monitor_app, name="monitor")

logger = get_logger(__name__)

_DATASET_CONFIG_OPTION = typer.Option(
    ..., "--dataset-config", exists=True, dir_okay=False, help="Path to a configs/data/*.yaml file."
)


def _not_implemented(command: str, phase: str) -> None:
    typer.secho(f"'{command}' is not implemented yet - lands in {phase}.", fg=typer.colors.YELLOW)
    raise typer.Exit(code=1)


def _echo_errors(label: str, errors: list[RowError]) -> None:
    if not errors:
        return
    typer.secho(f"{len(errors)} {label}:", fg=typer.colors.YELLOW)
    for err in errors[:20]:
        typer.echo(f"  - {err.subject_id}: {err.field}: {err.message}")
    if len(errors) > 20:
        typer.echo(f"  ... and {len(errors) - 20} more")


@app.command()
def serve(
    host: str | None = typer.Option(None, help="Override API_HOST."),
    port: int | None = typer.Option(None, help="Override API_PORT."),
    reload: bool | None = typer.Option(None, help="Override API_RELOAD."),
) -> None:
    """Run the FastAPI inference service."""
    settings = get_settings()
    configure_logging(settings)
    uvicorn.run(
        "ad_mri.api.main:app",
        host=host or settings.api_host,
        port=port or settings.api_port,
        reload=settings.api_reload if reload is None else reload,
    )


@data_app.command("ingest")
def data_ingest(
    dataset_config: Path = _DATASET_CONFIG_OPTION,
    interim_dir: Path = typer.Option(
        Path("data/interim"), help="Where inventory/converted DICOM output goes."
    ),
) -> None:
    """Discover raw scans for a dataset and write a raw inventory CSV."""
    configure_logging(get_settings())
    config = load_dataset_config(dataset_config)
    inventory_path = interim_dir / config.name / "inventory.csv"
    _, report = ingest_raw_scans(
        config, interim_dir=interim_dir, inventory_output_path=inventory_path
    )

    typer.echo(
        f"Discovered {report.subjects_discovered} subject(s), {report.scans_discovered} scan(s) "
        f"({report.dicom_series_converted} DICOM series converted)."
    )
    _echo_errors("issue(s) encountered", report.errors)
    typer.echo(f"Inventory written to {inventory_path}")


@data_app.command("validate")
def data_validate(
    dataset_config: Path = _DATASET_CONFIG_OPTION,
    inventory: Path | None = typer.Option(None, help="Default: data/interim/<name>/inventory.csv"),
    output: Path | None = typer.Option(None, help="Default: data/interim/<name>/validated.csv"),
) -> None:
    """Validate a raw inventory: existence, format, corruption, dimensions, labels."""
    configure_logging(get_settings())
    config = load_dataset_config(dataset_config)
    inventory_path = inventory or Path("data/interim") / config.name / "inventory.csv"
    output_path = output or Path("data/interim") / config.name / "validated.csv"

    inventory_df = pd.read_csv(inventory_path)
    validated_df, result = validate_inventory(inventory_df, config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    validated_df.to_csv(output_path, index=False)

    typer.echo(
        f"Validated {result.total_rows} row(s): "
        f"{result.passed_rows} passed, {result.failed_rows} failed."
    )
    _echo_errors("validation error(s)", result.errors)
    typer.echo(f"Validated inventory written to {output_path}")
    if not result.is_valid:
        raise typer.Exit(code=1)


@data_app.command("manifest")
def data_manifest(
    dataset_config: Path = _DATASET_CONFIG_OPTION,
    validated: Path | None = typer.Option(None, help="Default: data/interim/<name>/validated.csv"),
    output: Path | None = typer.Option(None, help="Default: data/processed/<name>/manifest.csv"),
    include_failed: bool = typer.Option(False, help="Include QC-failed rows (for audit only)."),
) -> None:
    """Build the final versioned manifest from a validated inventory."""
    configure_logging(get_settings())
    config = load_dataset_config(dataset_config)
    validated_path = validated or Path("data/interim") / config.name / "validated.csv"
    output_path = output or Path("data/processed") / config.name / "manifest.csv"

    validated_df = pd.read_csv(validated_path)
    manifest_df = build_manifest(validated_df, include_failed=include_failed)
    write_manifest(manifest_df, output_path)
    class_counts = summarize_class_distribution(manifest_df)

    typer.echo(f"Manifest written to {output_path} ({len(manifest_df)} row(s)).")
    typer.echo(f"Class distribution: {class_counts}")


@data_app.command("split")
def data_split(
    dataset_config: Path = _DATASET_CONFIG_OPTION,
    manifest: Path | None = typer.Option(None, help="Default: data/processed/<name>/manifest.csv"),
    output_dir: Path | None = typer.Option(None, help="Default: data/splits/<name>/"),
) -> None:
    """Create a reproducible, subject-level, leakage-safe train/val/test split."""
    configure_logging(get_settings())
    config = load_dataset_config(dataset_config)
    manifest_path = manifest or Path("data/processed") / config.name / "manifest.csv"
    split_output_dir = output_dir or Path("data/splits") / config.name

    paths = create_subject_level_split(
        manifest_path,
        split_output_dir,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        seed=config.seed,
    )
    typer.echo(f"Splits written (seed={config.seed}):")
    for name, path in paths.items():
        typer.echo(f"  {name}: {path}")


@data_app.command("cross-validate")
def data_cross_validate(
    train_manifest: Path = typer.Option(
        ..., "--train-manifest", exists=True, dir_okay=False, help="data/splits/<name>/train.csv"
    ),
    val_manifest: Path = typer.Option(
        ..., "--val-manifest", exists=True, dir_okay=False, help="data/splits/<name>/val.csv"
    ),
    output_dir: Path = typer.Option(
        ..., "--output-dir", help="Where fold_0/, fold_1/, ... are written."
    ),
    n_folds: int = typer.Option(5, "--n-folds"),
    seed: int = typer.Option(42, "--seed"),
) -> None:
    """Build K stratified, subject-level, leakage-safe CV folds from the
    development pool (train + val). The test set is never an input here —
    it structurally cannot leak into cross-validation."""
    configure_logging(get_settings())
    folds = create_cross_validation_folds(
        [train_manifest, val_manifest], output_dir, n_folds=n_folds, seed=seed
    )
    typer.echo(f"{len(folds)} fold(s) written to {output_dir} (seed={seed}):")
    for i, fold in enumerate(folds):
        typer.echo(f"  fold {i}: train={fold['train']} val={fold['val']}")


@data_app.command("preview-preprocessing")
def data_preview_preprocessing(
    image: Path = typer.Option(..., "--image", exists=True, dir_okay=False, help="A NIfTI file."),
    preprocessing_config: Path | None = typer.Option(
        None, "--preprocessing-config", help="Default: configs/preprocessing/default.yaml"
    ),
    output: Path | None = typer.Option(None, help="Default: <image>.preprocessing_preview.png"),
) -> None:
    """Render an original-vs-preprocessed comparison figure for one scan (sanity check)."""
    configure_logging(get_settings())
    config_path = preprocessing_config or Path("configs/preprocessing/default.yaml")
    config = load_preprocessing_config(config_path)

    output_path = plot_preprocessing_sample(image, config=config, output_path=output)
    typer.echo(f"Preview written to {output_path}")
    typer.echo(f"Preprocessing version: {config.version} (hash {config.compute_hash()[:12]})")


@model_app.command("summary")
def model_summary(
    model_config: Path = typer.Option(
        ...,
        "--model-config",
        exists=True,
        dir_okay=False,
        help="Path to a configs/model/*.yaml file.",
    ),
    batch_size: int = typer.Option(1, help="Batch size used for the dummy forward pass."),
) -> None:
    """Print a layer-by-layer summary (shapes, parameter count, estimated MACs/FLOPs)."""
    from ad_mri.modeling.factory import build_model
    from ad_mri.modeling.summary import format_summary, summarize_model

    configure_logging(get_settings())
    config = load_model_config(model_config)
    model = build_model(config)

    input_shape = (config.in_channels, *config.input_shape)
    summary = summarize_model(model, input_shape=input_shape, batch_size=batch_size)

    typer.echo(
        f"Architecture: {config.architecture} v{config.version} (hash {config.compute_hash()[:12]})"
    )
    typer.echo(format_summary(summary))


@cv_app.command("run")
def cv_run(
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
    cv_dir: Path = typer.Option(
        ..., "--cv-dir", help="Directory of fold_*/ produced by `ad-mri data cross-validate`."
    ),
    preprocessing_config: Path | None = typer.Option(
        None, "--preprocessing-config", help="Default: configs/preprocessing/default.yaml"
    ),
    output_dir: Path | None = typer.Option(
        None, "--output-dir", help="Default: artifacts/cv_reports/<run-name-prefix>"
    ),
    run_name_prefix: str = typer.Option("cv", "--run-name-prefix"),
) -> None:
    """Train one model per CV fold; report mean/std/95% CI validation metrics.

    Operates only on the development-pool folds under --cv-dir — the test set
    is never an input to this command, so it structurally cannot leak in.
    """
    from ad_mri.data_pipeline.cross_validation import discover_cv_folds
    from ad_mri.training.cross_validation import run_cross_validation
    from ad_mri.training.reporting import render_cv_report
    from ad_mri.training.training_config import load_training_config

    configure_logging(get_settings())
    resolved_preprocessing = preprocessing_config or Path("configs/preprocessing/default.yaml")

    folds = discover_cv_folds(cv_dir)
    cv_result = run_cross_validation(
        model_config=load_model_config(model_config),
        training_config=load_training_config(training_config),
        folds=folds,
        preprocessing_config=load_preprocessing_config(resolved_preprocessing),
        run_name_prefix=run_name_prefix,
    )

    report_dir = output_dir or Path("artifacts/cv_reports") / run_name_prefix
    report_paths = render_cv_report(cv_result, report_dir)

    typer.echo(f"Cross-validation complete: {len(cv_result.fold_results)} fold(s).")
    for name, summary in cv_result.metric_summary.items():
        std = f" +/- {summary.std:.4f}" if summary.std is not None else ""
        ci = (
            f" (95% CI [{summary.ci_lower:.4f}, {summary.ci_upper:.4f}])"
            if summary.ci_lower is not None
            else ""
        )
        typer.echo(f"  {name}: {summary.mean:.4f}{std}{ci}")
    typer.echo(f"Report: {report_paths['markdown']}")
    typer.echo(f"Plot: {report_paths['plot']}")


@cv_app.command("search")
def cv_search(
    model_config: Path = typer.Option(
        ...,
        "--model-config",
        exists=True,
        dir_okay=False,
        help="Base architecture; search overrides parts of it.",
    ),
    training_config: Path = typer.Option(
        ...,
        "--training-config",
        exists=True,
        dir_okay=False,
        help="Base hyperparameters; search overrides parts of it.",
    ),
    hpo_config: Path = typer.Option(
        ..., "--hpo-config", exists=True, dir_okay=False, help="Path to a configs/hpo/*.yaml file."
    ),
    cv_dir: Path = typer.Option(
        ..., "--cv-dir", help="Directory of fold_*/ produced by `ad-mri data cross-validate`."
    ),
    preprocessing_config: Path | None = typer.Option(
        None, "--preprocessing-config", help="Default: configs/preprocessing/default.yaml"
    ),
    output_dir: Path | None = typer.Option(
        None, "--output-dir", help="Default: artifacts/hpo_reports/<study-name>"
    ),
) -> None:
    """Hyperparameter search: each trial is scored by cross-validation on the
    development-pool folds under --cv-dir. Never touches the test set."""
    from ad_mri.data_pipeline.cross_validation import discover_cv_folds
    from ad_mri.modeling.model_config import save_model_config
    from ad_mri.training.hpo import run_hyperparameter_search
    from ad_mri.training.hpo_config import load_hpo_config
    from ad_mri.training.reporting import render_hpo_report
    from ad_mri.training.training_config import load_training_config, save_training_config

    configure_logging(get_settings())
    resolved_preprocessing = preprocessing_config or Path("configs/preprocessing/default.yaml")
    h_config = load_hpo_config(hpo_config)

    folds = discover_cv_folds(cv_dir)
    result = run_hyperparameter_search(
        hpo_config=h_config,
        base_model_config=load_model_config(model_config),
        base_training_config=load_training_config(training_config),
        folds=folds,
        preprocessing_config=load_preprocessing_config(resolved_preprocessing),
    )

    report_dir = output_dir or Path("artifacts/hpo_reports") / h_config.study_name
    report_paths = render_hpo_report(result, report_dir)
    best_model_path = save_model_config(result.best_model_config, report_dir / "best_model.yaml")
    best_training_path = save_training_config(
        result.best_training_config, report_dir / "best_training.yaml"
    )

    typer.echo(f"HPO complete: {len(result.trials)} trial(s) completed.")
    typer.echo(
        f"Best trial: #{result.best_trial_number} "
        f"({result.metric_name}={result.best_metric_value:.4f})"
    )
    typer.echo(f"Report: {report_paths['markdown']}")
    typer.echo(f"Best model config: {best_model_path}")
    typer.echo(f"Best training config: {best_training_path}")
    typer.echo(
        "Next: retrain on the full train+val pool with these configs "
        "(`ad-mri train --model-config <best_model> --training-config <best_training> "
        "--train-manifest .../train.csv --val-manifest .../val.csv`), then evaluate ONCE "
        "on the held-out test set — never before this point."
    )


@app.command()
def preprocess() -> None:
    """Run classical spatial preprocessing (skull-strip/atlas registration).

    Note: resampling/normalization/spatial-standardization are already
    implemented and run inline via `datasets.transforms.build_transforms` for
    both training and inference — see `ad-mri data preview-preprocessing` to
    inspect their effect. This command covers the remaining, heavier classical
    neuroimaging steps (brain extraction, atlas registration), not yet wired.
    """
    _not_implemented("preprocess", "a future hardening phase (skull-strip/registration)")


@app.command()
def train(
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
    """Train a model. Equivalent to `python -m ad_mri.training.train`."""
    from ad_mri.training.train import run_training

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
    if result.mlflow_run_id:
        typer.echo(f"MLflow run ID: {result.mlflow_run_id}")


@app.command()
def evaluate(
    model: Path = typer.Option(
        ..., "--model", exists=True, dir_okay=False, help="Path to a checkpoint (best.pt/last.pt)."
    ),
    dataset: Path = typer.Option(
        ...,
        "--dataset",
        exists=True,
        dir_okay=False,
        help="Manifest CSV to evaluate on — normally a held-out test.csv.",
    ),
    preprocessing_config: Path | None = typer.Option(
        None, "--preprocessing-config", help="Default: configs/preprocessing/default.yaml"
    ),
    output_dir: Path | None = typer.Option(
        None, "--output-dir", help="Default: artifacts/evaluation/<checkpoint-stem>-<timestamp>"
    ),
    threshold: float | None = typer.Option(
        None,
        "--threshold",
        help="Binary classification only: decision threshold on the positive class "
        "probability. Default: argmax (equivalent to 0.5).",
    ),
    batch_size: int = typer.Option(8, "--batch-size"),
) -> None:
    """Evaluate a trained model on a dataset. Equivalent to
    `python -m ad_mri.evaluation.evaluate`."""
    from ad_mri.evaluation.evaluate import run_evaluation

    configure_logging(get_settings())
    result = run_evaluation(
        model_path=model,
        dataset_manifest_path=dataset,
        preprocessing_config_path=preprocessing_config,
        output_dir=output_dir,
        threshold=threshold,
        batch_size=batch_size,
    )

    report = result.report
    typer.echo(f"Evaluated {report.n_samples} sample(s) from {dataset}.")
    typer.echo(
        f"Accuracy: {report.metrics['accuracy']:.4f}  F1 (macro): {report.metrics['f1_macro']:.4f}"
    )
    typer.echo(f"Report directory: {result.output_dir}")


@explain_app.command("run")
def explain_run(
    model: Path = typer.Option(
        ..., "--model", exists=True, dir_okay=False, help="Path to a checkpoint (best.pt/last.pt)."
    ),
    image: Path = typer.Option(
        ..., "--image", exists=True, dir_okay=False, help="A NIfTI/DICOM scan to explain."
    ),
    preprocessing_config: Path | None = typer.Option(
        None, "--preprocessing-config", help="Default: configs/preprocessing/default.yaml"
    ),
    methods: str = typer.Option("lime,shap", "--methods", help="Comma-separated: lime, shap."),
    lime_config: Path | None = typer.Option(
        None, "--lime-config", help="Default: configs/explainability/lime_default.yaml"
    ),
    shap_config: Path | None = typer.Option(
        None, "--shap-config", help="Default: configs/explainability/shap_default.yaml"
    ),
    background_manifest: Path | None = typer.Option(
        None, "--background-manifest", help="Required when --methods includes 'shap'."
    ),
    input_id: str | None = typer.Option(None, "--input-id", help="Default: the image filename."),
    target_class: str | None = typer.Option(
        None, "--target-class", help="Explain this class by name. Default: the predicted class."
    ),
    output_dir: Path | None = typer.Option(
        None, "--output-dir", help="Default: artifacts/explanations/<input-id>-<timestamp>"
    ),
    seed: int = typer.Option(42, "--seed"),
) -> None:
    """Explain one prediction (LIME and/or SHAP). Equivalent to
    `python -m ad_mri.explainability.explain`."""
    from ad_mri.explainability.explain import run_explanation

    configure_logging(get_settings())
    bundle = run_explanation(
        model_path=model,
        image_path=image,
        preprocessing_config_path=preprocessing_config,
        methods=tuple(m.strip() for m in methods.split(",") if m.strip()),
        lime_config_path=lime_config,
        shap_config_path=shap_config,
        background_manifest_path=background_manifest,
        input_id=input_id,
        target_class=target_class,
        output_dir=output_dir,
        seed=seed,
    )
    typer.echo(
        f"Predicted: {bundle.prediction.predicted_class} ({bundle.prediction.confidence:.4f})"
    )
    typer.echo(f"Explained class: {bundle.class_names[bundle.target_class_index]}")
    typer.echo(f"Output directory: {bundle.output_dir}")


@explain_app.command("global-shap")
def explain_global_shap(
    model: Path = typer.Option(
        ..., "--model", exists=True, dir_okay=False, help="Path to a checkpoint (best.pt/last.pt)."
    ),
    samples_manifest: Path = typer.Option(
        ..., "--samples-manifest", exists=True, dir_okay=False, help="Samples to aggregate over."
    ),
    background_manifest: Path = typer.Option(
        ..., "--background-manifest", exists=True, dir_okay=False, help="SHAP reference set."
    ),
    target_class: str = typer.Option(
        ..., "--target-class", help="Class to aggregate importance for."
    ),
    preprocessing_config: Path | None = typer.Option(
        None, "--preprocessing-config", help="Default: configs/preprocessing/default.yaml"
    ),
    shap_config: Path | None = typer.Option(
        None, "--shap-config", help="Default: configs/explainability/shap_default.yaml"
    ),
    n_samples: int = typer.Option(8, "--n-samples", help="How many samples to aggregate over."),
    region_pool_size: int = typer.Option(4, "--region-pool-size"),
    output_dir: Path | None = typer.Option(
        None, "--output-dir", help="Default: artifacts/explanations/global-shap-<timestamp>"
    ),
) -> None:
    """Aggregate |SHAP value| across several samples into one downsampled
    region-importance grid for one class — "global importance", meaningful
    only in this multi-sample, downsampled form (see docs/explainability.md)."""
    import time

    from ad_mri.explainability.config import load_shap_config
    from ad_mri.explainability.sampling import load_volume_batch
    from ad_mri.explainability.service import ExplainabilityService
    from ad_mri.explainability.shap_explainer import compute_global_shap_importance

    configure_logging(get_settings())
    service = ExplainabilityService(model, preprocessing_config)
    if target_class not in service.class_names:
        typer.secho(
            f"Unknown class {target_class!r}; known classes: {service.class_names}",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)
    target_class_index = service.class_names.index(target_class)

    resolved_shap_config = load_shap_config(
        shap_config or Path("configs/explainability/shap_default.yaml")
    )
    samples, sample_subject_ids = load_volume_batch(
        samples_manifest, service.preprocessing_config, n_samples, seed=resolved_shap_config.seed
    )
    background, _ = load_volume_batch(
        background_manifest,
        service.preprocessing_config,
        resolved_shap_config.n_background_samples,
        seed=resolved_shap_config.seed,
    )
    resolved_output_dir = output_dir or (
        Path("artifacts/explanations") / f"global-shap-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    result = compute_global_shap_importance(
        service.model,
        samples,
        background,
        service.class_names,
        target_class_index,
        resolved_shap_config,
        resolved_output_dir,
        service.device,
        region_pool_size=region_pool_size,
    )
    typer.echo(f"Aggregated over {result.n_samples_used} sample(s): {sample_subject_ids}")
    typer.echo(f"Region grid shape: {result.region_grid_shape}")
    for name, path in sorted(result.artifacts.items()):
        typer.echo(f"  {name}: {path}")


@app.command()
def predict(
    model: Path = typer.Option(
        ..., "--model", exists=True, dir_okay=False, help="Path to a checkpoint (best.pt/last.pt)."
    ),
    image: Path = typer.Option(
        ..., "--image", exists=True, dir_okay=False, help="A NIfTI/DICOM scan to classify."
    ),
    preprocessing_config: Path | None = typer.Option(
        None, "--preprocessing-config", help="Default: configs/preprocessing/default.yaml"
    ),
    calibration_config: Path | None = typer.Option(
        None, "--calibration-config", help="Default: uncalibrated (plain softmax)."
    ),
    device: str = typer.Option("auto", "--device", help="'auto' | 'cpu' | 'cuda'"),
    timeout_seconds: float = typer.Option(30.0, "--timeout-seconds"),
    request_id: str | None = typer.Option(None, "--request-id", help="Default: a generated UUID."),
    explain: bool = typer.Option(False, "--explain", help="Also generate a LIME/SHAP explanation."),
    methods: str = typer.Option("lime,shap", "--methods", help="Used only with --explain."),
    background_manifest: Path | None = typer.Option(
        None, "--background-manifest", help="Required for a SHAP explanation."
    ),
) -> None:
    """Classify one MRI scan. Equivalent to `python -m ad_mri.inference.predict`."""
    import json

    from ad_mri.inference.predict import run_prediction

    configure_logging(get_settings())
    result = run_prediction(
        model_path=model,
        image_path=image,
        preprocessing_config_path=preprocessing_config,
        calibration_config_path=calibration_config,
        device_choice=device,
        timeout_seconds=timeout_seconds,
        request_id=request_id,
        include_explanation=explain,
        explanation_methods=(
            tuple(m.strip() for m in methods.split(",") if m.strip()) if explain else None
        ),
        background_manifest_path=background_manifest,
    )
    typer.echo(json.dumps(result.to_dict(), indent=2))
    typer.secho(f"\n{PREDICTION_DISCLAIMER}", fg=typer.colors.YELLOW)


@app.command("register-model")
def register_model() -> None:
    """Register a trained model artifact in the MLflow Model Registry."""
    _not_implemented("register-model", "the training phase")


# --- monitor: data/model drift reports, delayed model-performance monitoring ---


@monitor_app.command("build-drift-reference")
def monitor_build_drift_reference(
    manifest: Path = typer.Option(
        ..., "--manifest", exists=True, dir_okay=False, help="Training/reference manifest CSV."
    ),
    preprocessing_config: Path = typer.Option(
        Path("configs/preprocessing/default.yaml"), "--preprocessing-config"
    ),
    output: Path = typer.Option(Path("artifacts/monitoring/drift_reference.json"), "--output"),
    max_samples: int | None = typer.Option(
        None, "--max-samples", help="Cap the number of manifest rows processed (default: all)."
    ),
) -> None:
    """Build the training-distribution baseline `monitor drift-report` compares
    a production window against. Run once per model version (or whenever the
    training distribution meaningfully changes)."""
    import pandas as pd

    from ad_mri.datasets.preprocessing_config import load_preprocessing_config
    from ad_mri.monitoring.drift import build_drift_reference

    configure_logging(get_settings())
    df = pd.read_csv(manifest)
    image_paths = [Path(p) for p in df["image_path"]]
    labels = [str(label) for label in df["label"]] if "label" in df.columns else None

    reference = build_drift_reference(
        image_paths,
        labels,
        load_preprocessing_config(preprocessing_config),
        source_manifest=str(manifest),
        max_samples=max_samples,
    )
    reference.save(output)
    typer.echo(f"Built a drift reference from {reference.n_samples} sample(s): {output}")


@monitor_app.command("drift-report")
def monitor_drift_report(
    reference: Path = typer.Option(
        Path("artifacts/monitoring/drift_reference.json"), "--reference"
    ),
    since_hours: float | None = typer.Option(
        None, "--since-hours", help="Only consider predictions from this many hours ago onward."
    ),
    output_dir: Path = typer.Option(Path("artifacts/monitoring/reports"), "--output-dir"),
) -> None:
    """Compare recent production input statistics against the reference
    distribution and write a data-drift report (JSON + markdown)."""
    from datetime import UTC, datetime, timedelta

    from ad_mri.db.monitoring_service import MonitoringService
    from ad_mri.db.session import get_sessionmaker
    from ad_mri.monitoring.drift import DriftReference, generate_data_drift_report
    from ad_mri.monitoring.thresholds import PSI_ALERT_THRESHOLD

    configure_logging(get_settings())
    drift_reference = DriftReference.load(reference)
    window_start = datetime.now(UTC) - timedelta(hours=since_hours) if since_hours else None

    session = get_sessionmaker()()
    try:
        service = MonitoringService(session)
        rows = service.get_input_statistics_window(start=window_start)
        current_class_counts = service.get_predicted_class_counts(start=window_start)
    finally:
        session.close()

    current_samples: dict[str, list[float]] = {
        name: [getattr(row, name) for row in rows]
        for name in (
            "mean_intensity",
            "std_intensity",
            "p01_intensity",
            "p50_intensity",
            "p99_intensity",
            "nonzero_fraction",
        )
    }

    report = generate_data_drift_report(
        drift_reference,
        current_samples,
        current_class_counts or None,
        window_start=window_start,
        window_end=datetime.now(UTC),
    )
    paths = report.save(output_dir)

    if report.any_drifted:
        # Durably recorded so the long-running API process can reflect this
        # verdict into DRIFT_ALERT_ACTIVE on its next /metrics scrape - see
        # monitoring.alert_state's module docstring for why this indirection
        # (through system_events, not a direct gauge set) is required for a
        # short-lived CLI process.
        from ad_mri.db.audit_service import AuditService

        session = get_sessionmaker()()
        try:
            AuditService(session).record_event(
                event_type="data_drift_detected",
                severity="warning",
                detail={
                    "drifted_features": [r.feature for r in report.feature_results if r.drifted],
                    "class_distribution_drifted": report.class_distribution_drifted,
                },
            )
        finally:
            session.close()

    typer.secho(
        f"Data drift: {'DETECTED' if report.any_drifted else 'none'} "
        f"({report.current_n_samples} sample(s) vs. {report.reference_n_samples} reference)",
        fg=typer.colors.RED if report.any_drifted else typer.colors.GREEN,
    )
    typer.echo(f"Report: {paths['markdown']}")
    if report.any_drifted:
        typer.echo(
            f"See docs/monitoring.md '\"Retraining triggers\"' - PSI >= {PSI_ALERT_THRESHOLD} "
            "on one or more features is one of the documented triggers."
        )


@monitor_app.command("model-drift-report")
def monitor_model_drift_report(
    model_version_id: str = typer.Option(..., "--model-version-id"),
    class_names: str = typer.Option(
        ..., "--class-names", help="Comma-separated, in the model's own index order, e.g. AD,CN."
    ),
    since_hours: float | None = typer.Option(None, "--since-hours"),
    output_dir: Path = typer.Option(Path("artifacts/monitoring/reports"), "--output-dir"),
) -> None:
    """Compute delayed, ground-truth-backed performance metrics for
    `model_version_id` and compare against its registered baseline (see
    `register-baseline`). Requires ground-truth labels to have been recorded
    (`record-ground-truth` / `POST /api/v1/monitoring/ground-truth`) - see
    `MIN_LABELED_SAMPLES_FOR_MODEL_DRIFT` in `monitoring.thresholds` for the
    minimum sample size this refuses to compute a verdict below."""
    import uuid
    from datetime import UTC, datetime, timedelta

    from ad_mri.db.monitoring_service import MonitoringService
    from ad_mri.db.session import get_sessionmaker
    from ad_mri.monitoring.model_drift import compute_delayed_performance

    configure_logging(get_settings())
    version_uuid = uuid.UUID(model_version_id)
    since = datetime.now(UTC) - timedelta(hours=since_hours) if since_hours else None

    session = get_sessionmaker()()
    try:
        service = MonitoringService(session)
        pairs = service.get_labeled_predictions(version_uuid, since=since)
        version = service.get_model_version(version_uuid)
        baseline_metrics = version.baseline_metrics if version else None
    finally:
        session.close()

    report = compute_delayed_performance(
        model_version_id, pairs, class_names.split(","), baseline_metrics
    )
    paths = report.save(output_dir)

    if report.degraded:
        # See the matching comment in monitor_drift_report - same
        # indirection, for the same reason.
        from ad_mri.db.audit_service import AuditService

        alert_session = get_sessionmaker()()
        try:
            AuditService(alert_session).record_event(
                event_type="model_drift_detected",
                severity="critical",
                detail={
                    "model_version_id": model_version_id,
                    "degraded_metrics": report.degraded_metrics,
                },
            )
        finally:
            alert_session.close()

    if report.insufficient_samples:
        typer.secho(
            f"Insufficient labeled samples ({report.n_labeled_samples}) - no verdict computed.",
            fg=typer.colors.YELLOW,
        )
    else:
        typer.secho(
            f"Model drift: {'DEGRADED' if report.degraded else 'no significant degradation'} "
            f"({report.n_labeled_samples} labeled sample(s))",
            fg=typer.colors.RED if report.degraded else typer.colors.GREEN,
        )
    typer.echo(f"Report: {paths['markdown']}")


@monitor_app.command("register-baseline")
def monitor_register_baseline(
    model_version_id: str = typer.Option(..., "--model-version-id"),
    from_evaluation_report: Path = typer.Option(
        ...,
        "--from-evaluation-report",
        exists=True,
        dir_okay=False,
        help="A report.json written by `ad-mri evaluate` (its 'metrics' key is used as-is).",
    ),
) -> None:
    """Register this model version's held-out evaluation metrics as the
    baseline `model-drift-report` compares delayed production performance
    against. Run once per model version, right after evaluating it."""
    import json
    import uuid

    from ad_mri.db.monitoring_service import MonitoringService
    from ad_mri.db.session import get_sessionmaker

    configure_logging(get_settings())
    metrics = json.loads(from_evaluation_report.read_text(encoding="utf-8"))["metrics"]

    session = get_sessionmaker()()
    try:
        MonitoringService(session).set_baseline_metrics(uuid.UUID(model_version_id), metrics)
    finally:
        session.close()

    typer.echo(f"Registered baseline metrics for model version {model_version_id}.")


@monitor_app.command("record-ground-truth")
def monitor_record_ground_truth(
    request_id: str = typer.Option(..., "--request-id"),
    true_class: str = typer.Option(..., "--true-class"),
    label_source: str = typer.Option(..., "--label-source"),
) -> None:
    """Attach a true class to a past prediction from the command line -
    equivalent to `POST /api/v1/monitoring/ground-truth`, for offline/batch
    labeling workflows."""
    from ad_mri.db.monitoring_service import MonitoringService
    from ad_mri.db.session import get_sessionmaker

    configure_logging(get_settings())
    session = get_sessionmaker()()
    try:
        MonitoringService(session).record_ground_truth(
            external_request_id=request_id, true_class=true_class, label_source=label_source
        )
    finally:
        session.close()

    typer.echo(f"Recorded ground truth for request {request_id}: {true_class} ({label_source}).")


if __name__ == "__main__":
    app()
