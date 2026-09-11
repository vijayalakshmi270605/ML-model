"""CLI entry point for standalone model evaluation.

Runnable directly:

    python -m ad_mri.evaluation.evaluate --model <checkpoint.pt> --dataset <test_manifest.csv>

or via the main CLI (`ad-mri evaluate ...`, see `cli/main.py`) — both call
`run_evaluation()` below, the single evaluation entry point. This module
contains only CLI plumbing (argument parsing, config loading, model loading,
the inference loop over the dataset); all metric/curve/report computation
lives in `evaluation.report`/`evaluation.metrics`/`evaluation.curves`, and
nothing here is imported by `training/` — evaluation is a read-only consumer
of a checkpoint someone else already trained. See
docs/TECHNICAL_SPECIFICATION.md §10, docs/evaluation.md.

**Never use the test set during model selection**: this command is the only
place `--dataset` is meant to point at a held-out test manifest — every
earlier command (`ad-mri train`, `ad-mri cv run`, `ad-mri cv search`) only
ever accepts train/val manifests, so there is no code path upstream of this
one that could have optimized against test-set performance.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer

from ad_mri.core.config import get_settings
from ad_mri.core.exceptions import ConfigurationError
from ad_mri.core.logging import configure_logging, get_logger
from ad_mri.data_pipeline.validation import compute_sha256
from ad_mri.datasets.preprocessing_config import PreprocessingConfig, load_preprocessing_config
from ad_mri.evaluation.metrics import apply_binary_threshold
from ad_mri.evaluation.report import (
    EvaluationReport,
    evaluate_predictions,
    render_evaluation_report,
)
from ad_mri.modeling.checkpoint import load_checkpoint

logger = get_logger(__name__)

app = typer.Typer(help="Evaluate a trained ad-mri checkpoint on a held-out dataset.")


@dataclass
class EvaluationRunResult:
    report: EvaluationReport
    output_dir: Path
    written: dict[str, Path]


def _resolve_class_names(metadata_extra: dict[str, Any], checkpoint_path: Path) -> list[str]:
    class_names = metadata_extra.get("class_names")
    if not class_names:
        raise ConfigurationError(
            f"Checkpoint at {checkpoint_path} does not carry a class_names mapping "
            "(expected in its training-provenance metadata) — cannot reliably map "
            "predicted indices back to label names. Re-train with the current "
            "training pipeline (Phase 5+), which always records this."
        )
    return list(class_names)


def _run_inference(
    model: Any,
    manifest_path: Path,
    label_to_index: dict[str, int],
    preprocessing_config: PreprocessingConfig,
    device: Any,
    batch_size: int,
    num_workers: int,
) -> tuple[list[int], list[int], list[list[float]], list[str]]:
    import torch
    from torch.utils.data import DataLoader

    from ad_mri.core.exceptions import DataValidationError, PreprocessingError
    from ad_mri.datasets.dataset import MRIDataset
    from ad_mri.datasets.transforms import build_transforms

    transform = build_transforms(
        preprocessing_config, train=False
    )  # deterministic only, no augmentation
    dataset = MRIDataset(manifest_path, transform=transform, label_to_index=label_to_index)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model.eval()
    y_true: list[int] = []
    y_pred_argmax: list[int] = []
    y_proba: list[list[float]] = []
    subject_ids: list[str] = []

    iterator = iter(loader)
    while True:
        try:
            batch = next(iterator)
        except StopIteration:
            break
        except (PreprocessingError, DataValidationError) as exc:
            logger.warning("evaluation_batch_skipped_due_to_bad_sample", error=str(exc))
            continue

        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"]
        with torch.no_grad():
            logits = model(images)
            probs = torch.softmax(logits.float(), dim=1)
            preds = torch.argmax(probs, dim=1)

        y_true.extend(labels.tolist())
        y_pred_argmax.extend(preds.cpu().tolist())
        y_proba.extend(probs.cpu().tolist())
        subject_ids.extend(str(sid) for sid in batch["subject_id"])

    if not y_true:
        raise ConfigurationError(
            f"No usable samples found in {manifest_path} — nothing to evaluate."
        )
    return y_true, y_pred_argmax, y_proba, subject_ids


def run_evaluation(
    model_path: Path,
    dataset_manifest_path: Path,
    preprocessing_config_path: Path | None = None,
    output_dir: Path | None = None,
    threshold: float | None = None,
    batch_size: int = 8,
    num_workers: int = 0,
    device_choice: str = "auto",
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> EvaluationRunResult:
    """Load a checkpoint, run it over `dataset_manifest_path`, and produce a
    full evaluation report. The single evaluation entry point.

    Raises:
        ConfigurationError: the checkpoint is unreadable/corrupted, doesn't
            carry a class-label mapping, its architecture doesn't match the
            dataset's class count, the given `--threshold` is used with a
            non-binary model, or the given preprocessing config doesn't match
            the one the checkpoint was trained with.
    """
    import torch

    model, model_config, metadata = load_checkpoint(model_path)
    class_names = _resolve_class_names(metadata.extra, model_path)
    if len(class_names) != model_config.num_classes:
        raise ConfigurationError(
            f"Checkpoint class_names {class_names} has {len(class_names)} entries but "
            f"model_config.num_classes={model_config.num_classes}."
        )
    label_to_index = {name: index for index, name in enumerate(class_names)}

    resolved_preprocessing_path = preprocessing_config_path or Path(
        "configs/preprocessing/default.yaml"
    )
    preprocessing_config = load_preprocessing_config(resolved_preprocessing_path)
    if tuple(preprocessing_config.target_shape) != tuple(model_config.input_shape):
        raise ConfigurationError(
            f"preprocessing_config.target_shape {preprocessing_config.target_shape} "
            f"!= model_config.input_shape {model_config.input_shape}."
        )
    stored_preprocessing_hash = metadata.extra.get("preprocessing_config_hash")
    if (
        stored_preprocessing_hash
        and stored_preprocessing_hash != preprocessing_config.compute_hash()
    ):
        raise ConfigurationError(
            f"preprocessing config at {resolved_preprocessing_path} (hash "
            f"{preprocessing_config.compute_hash()[:12]}) does not match the one the "
            f"checkpoint was trained with (hash {stored_preprocessing_hash[:12]}) — "
            "evaluating with mismatched preprocessing would silently invalidate every "
            "metric below. Pass the exact --preprocessing-config used for training."
        )

    if threshold is not None and len(class_names) != 2:
        raise ConfigurationError(
            f"--threshold is only meaningful for binary classification; this checkpoint "
            f"has {len(class_names)} classes ({class_names})."
        )

    if device_choice == "cpu":
        device = torch.device("cpu")
    elif device_choice == "cuda":
        if not torch.cuda.is_available():
            raise ConfigurationError("device='cuda' was requested but CUDA is not available.")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    logger.info(
        "evaluation_started",
        checkpoint=str(model_path),
        dataset=str(dataset_manifest_path),
        classes=class_names,
        device=str(device),
        threshold=threshold,
    )

    y_true, y_pred_argmax, y_proba, subject_ids = _run_inference(
        model,
        dataset_manifest_path,
        label_to_index,
        preprocessing_config,
        device,
        batch_size,
        num_workers,
    )
    y_pred = apply_binary_threshold(y_proba, threshold) if threshold is not None else y_pred_argmax

    report = evaluate_predictions(
        y_true,
        y_pred,
        y_proba,
        class_names,
        model_version=model_config.version,
        model_config_hash=model_config.compute_hash(),
        checkpoint_path=str(model_path),
        dataset_manifest_path=str(dataset_manifest_path),
        dataset_manifest_sha256=compute_sha256(dataset_manifest_path),
        preprocessing_version=preprocessing_config.version,
        preprocessing_config_hash=preprocessing_config.compute_hash(),
        threshold=threshold,
        subject_ids=subject_ids,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    resolved_output_dir = output_dir or (
        Path("artifacts/evaluation") / f"{model_path.stem}-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    written = render_evaluation_report(report, resolved_output_dir)

    logger.info(
        "evaluation_complete",
        output_dir=str(resolved_output_dir),
        accuracy=report.metrics.get("accuracy"),
        f1_macro=report.metrics.get("f1_macro"),
    )
    return EvaluationRunResult(report=report, output_dir=resolved_output_dir, written=written)


@app.command()
def main(
    model: Path = typer.Option(
        ..., "--model", exists=True, dir_okay=False, help="Path to a checkpoint (best.pt/last.pt)."
    ),
    dataset: Path = typer.Option(
        ...,
        "--dataset",
        exists=True,
        dir_okay=False,
        help="Manifest CSV to evaluate on — normally a held-out test.csv, never the "
        "manifest used for training/cross-validation/HPO.",
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
        help="Binary classification only: decision threshold on the "
        "positive (index-1) class probability. Default: argmax (equivalent to 0.5).",
    ),
    batch_size: int = typer.Option(8, "--batch-size"),
    num_workers: int = typer.Option(0, "--num-workers"),
    device: str = typer.Option("auto", "--device", help="'auto' | 'cpu' | 'cuda'"),
    confidence_level: float = typer.Option(0.95, "--confidence-level"),
    n_bootstrap: int = typer.Option(
        1000, "--n-bootstrap", help="Bootstrap resamples for confidence intervals."
    ),
    seed: int = typer.Option(42, "--seed", help="Bootstrap resampling seed (reproducibility)."),
) -> None:
    """Evaluate a trained model on a dataset. Runnable as
    `python -m ad_mri.evaluation.evaluate` or `ad-mri evaluate`."""
    configure_logging(get_settings())

    result = run_evaluation(
        model_path=model,
        dataset_manifest_path=dataset,
        preprocessing_config_path=preprocessing_config,
        output_dir=output_dir,
        threshold=threshold,
        batch_size=batch_size,
        num_workers=num_workers,
        device_choice=device,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    report = result.report
    typer.echo(f"Evaluated {report.n_samples} sample(s) from {dataset}.")
    typer.echo(f"Classes: {report.class_names}")
    typer.echo(f"Accuracy: {report.metrics['accuracy']:.4f}")
    typer.echo(f"F1 (macro): {report.metrics['f1_macro']:.4f}")
    if "roc_auc" in report.metrics:
        typer.echo(f"ROC-AUC: {report.metrics['roc_auc']:.4f}")
    elif "roc_auc_macro" in report.metrics:
        typer.echo(f"ROC-AUC (macro): {report.metrics['roc_auc_macro']:.4f}")
    typer.echo(f"Report directory: {result.output_dir}")
    for name, path in sorted(result.written.items()):
        typer.echo(f"  {name}: {path}")


if __name__ == "__main__":
    app()
