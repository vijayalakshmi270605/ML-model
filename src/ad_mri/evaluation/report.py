"""Comprehensive evaluation report: orchestrates `evaluation.metrics`,
`evaluation.confusion_matrix`, `evaluation.curves`, and
`evaluation.statistics` into one `EvaluationReport`, then renders it as a
confusion-matrix PNG, ROC-curve PNG, PR-curve PNG, text classification report,
JSON report, and CSV report — the deliverables of the standalone evaluation
CLI (`evaluation.evaluate`). See docs/TECHNICAL_SPECIFICATION.md §10,
docs/evaluation.md.

This module contains no training logic and is never imported by `training/`;
it is imported by `evaluation.evaluate` (the CLI) and can equally be called
from a notebook/script for one-off analysis.

`scikit-learn` is a training-phase dependency (requirements/train.txt),
imported lazily so this module stays importable without it.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ad_mri.core.logging import get_logger
from ad_mri.evaluation.confusion_matrix import compute_confusion_matrix, render_confusion_matrix
from ad_mri.evaluation.curves import (
    CurveSet,
    compute_pr_curve,
    compute_roc_curve,
    curve_set_to_dict,
    render_pr_curve,
    render_roc_curve,
)
from ad_mri.evaluation.metrics import compute_classification_metrics
from ad_mri.evaluation.statistics import MetricSummary, bootstrap_metric_ci

logger = get_logger(__name__)

#: Metrics a bootstrap confidence interval is computed for — a fixed,
#: label-only subset (no y_proba-dependent metric) plus, when available,
#: the headline AUC metric — kept small deliberately so the report highlights
#: uncertainty on the numbers a reader is most likely to act on, rather than
#: bootstrapping every one of the ~5*n_classes metrics in the flat dict.
_CI_LABEL_METRICS = ("accuracy", "precision_macro", "recall_macro", "specificity_macro", "f1_macro")


@dataclass(frozen=True)
class EvaluationReport:
    """Everything needed to reproduce and audit one evaluation run, bundled
    with the exact model/preprocessing version it was produced with
    (requirement: "store evaluation results with model version")."""

    model_version: str
    model_config_hash: str
    checkpoint_path: str
    preprocessing_version: str | None
    preprocessing_config_hash: str | None
    dataset_manifest_path: str
    dataset_manifest_sha256: str
    n_samples: int
    class_names: list[str]
    threshold: float | None
    evaluated_at: str
    metrics: dict[str, float]
    confidence_intervals: dict[str, MetricSummary]
    confusion_matrix: list[list[int]]
    classification_report_text: str
    roc_curve: CurveSet
    pr_curve: CurveSet
    predictions: list[dict[str, Any]] = field(default_factory=list)


def evaluate_predictions(
    y_true: list[int],
    y_pred: list[int],
    y_proba: list[list[float]],
    class_names: list[str],
    *,
    model_version: str,
    model_config_hash: str,
    checkpoint_path: str,
    dataset_manifest_path: str,
    dataset_manifest_sha256: str,
    preprocessing_version: str | None = None,
    preprocessing_config_hash: str | None = None,
    threshold: float | None = None,
    subject_ids: list[str] | None = None,
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> EvaluationReport:
    """Compute every metric/curve/report artifact for one batch of predictions
    against ground truth. Pure computation — writes nothing to disk; see
    `render_evaluation_report` for that.

    `class_names` documents the label taxonomy explicitly (index -> name),
    fulfilling "clearly document class labels" — every output (confusion
    matrix axes, per-class metric keys, curve legends) is labeled by name,
    never by bare integer index.
    """
    from sklearn.metrics import classification_report

    metrics = compute_classification_metrics(y_true, y_pred, class_names, y_proba=y_proba)
    confusion = compute_confusion_matrix(y_true, y_pred, class_names)
    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    )

    confidence_intervals: dict[str, MetricSummary] = {}
    for metric_key in _CI_LABEL_METRICS:
        if metric_key not in metrics:
            continue
        metric_fn = _label_metric_fn(metric_key, class_names)
        confidence_intervals[metric_key] = bootstrap_metric_ci(
            metric_fn,
            y_true,
            y_pred,
            metric_name=metric_key,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            seed=seed,
        )

    auc_metric_key = (
        "roc_auc"
        if "roc_auc" in metrics
        else ("roc_auc_macro" if "roc_auc_macro" in metrics else None)
    )
    if auc_metric_key is not None:
        auc_fn = _auc_metric_fn(auc_metric_key, class_names)
        confidence_intervals[auc_metric_key] = bootstrap_metric_ci(
            auc_fn,
            y_true,
            y_proba,
            metric_name=auc_metric_key,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            seed=seed,
        )

    roc_curve_set = compute_roc_curve(y_true, y_proba, class_names)
    pr_curve_set = compute_pr_curve(y_true, y_proba, class_names)

    predictions: list[dict[str, Any]] = []
    for index, (true_label, pred_label, proba_row) in enumerate(
        zip(y_true, y_pred, y_proba, strict=True)
    ):
        predictions.append(
            {
                "subject_id": subject_ids[index] if subject_ids else None,
                "true_label": class_names[true_label],
                "predicted_label": class_names[pred_label],
                "correct": true_label == pred_label,
                **{
                    f"proba_{name}": proba_row[class_index]
                    for class_index, name in enumerate(class_names)
                },
            }
        )

    return EvaluationReport(
        model_version=model_version,
        model_config_hash=model_config_hash,
        checkpoint_path=checkpoint_path,
        preprocessing_version=preprocessing_version,
        preprocessing_config_hash=preprocessing_config_hash,
        dataset_manifest_path=dataset_manifest_path,
        dataset_manifest_sha256=dataset_manifest_sha256,
        n_samples=len(y_true),
        class_names=list(class_names),
        threshold=threshold,
        evaluated_at=datetime.now(UTC).isoformat(),
        metrics=metrics,
        confidence_intervals=confidence_intervals,
        confusion_matrix=confusion,
        classification_report_text=report_text,
        roc_curve=roc_curve_set,
        pr_curve=pr_curve_set,
        predictions=predictions,
    )


def _argmax(row: list[float]) -> int:
    best_index = 0
    for index in range(1, len(row)):
        if row[index] > row[best_index]:
            best_index = index
    return best_index


def _label_metric_fn(metric_key: str, class_names: list[str]) -> Any:
    """Build a `(y_true, y_pred) -> float` closure for one flat metric key, for
    use as a `bootstrap_metric_ci` callback."""

    def _fn(y_true: list[int], y_pred: list[int]) -> float:
        return compute_classification_metrics(y_true, y_pred, class_names).get(
            metric_key, float("nan")
        )

    return _fn


def _auc_metric_fn(metric_key: str, class_names: list[str]) -> Any:
    """Build a `(y_true, y_proba) -> float` closure for the headline AUC
    metric, for use as a `bootstrap_metric_ci` callback. A degenerate
    bootstrap resample (missing a class) yields NaN for that replicate,
    which `numpy.percentile` propagates — acceptable for a supplementary
    interval around an already-computed point estimate, and visible rather
    than silently dropped."""

    def _fn(y_true: list[int], y_proba: list[list[float]]) -> float:
        # A dummy y_pred (argmax) is required by compute_classification_metrics's
        # signature but does not affect the AUC-family metrics it returns.
        y_pred = [_argmax(row) for row in y_proba]
        metrics = compute_classification_metrics(y_true, y_pred, class_names, y_proba=y_proba)
        return metrics.get(metric_key, float("nan"))

    return _fn


def _metric_summary_to_dict(summary: MetricSummary) -> dict[str, Any]:
    """JSON-safe form of a `MetricSummary` — bootstrap replicate values are
    dropped (there can be `n_bootstrap` of them; the summary statistics are
    what the report is for) but fold-style small `values` lists are kept."""
    payload = asdict(summary)
    if summary.n > 20:  # heuristic: this was a bootstrap summary, not a small fold list
        payload["values"] = None
    return payload


def render_evaluation_report(report: EvaluationReport, output_dir: Path) -> dict[str, Path]:
    """Write every Phase 7 deliverable for one `EvaluationReport` under
    `output_dir`: confusion matrix PNG, ROC curve PNG, PR curve PNG, text
    classification report, JSON report, CSV report, per-sample predictions
    CSV, and a human-readable markdown summary.

    Returns a `{name: Path}` dict of everything written. Deterministic given
    an identical `report` (requirement: "the final report must be reproducible").
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    cm_path = render_confusion_matrix(
        report.confusion_matrix, report.class_names, output_dir / "confusion_matrix.png"
    )
    written["confusion_matrix"] = cm_path

    roc_path = render_roc_curve(report.roc_curve, output_dir / "roc_curve.png")
    if roc_path is not None:
        written["roc_curve"] = roc_path

    pr_path = render_pr_curve(report.pr_curve, output_dir / "pr_curve.png")
    if pr_path is not None:
        written["pr_curve"] = pr_path

    classification_report_path = output_dir / "classification_report.txt"
    classification_report_path.write_text(report.classification_report_text, encoding="utf-8")
    written["classification_report"] = classification_report_path

    json_path = output_dir / "evaluation_report.json"
    json_payload = {
        "model_version": report.model_version,
        "model_config_hash": report.model_config_hash,
        "checkpoint_path": report.checkpoint_path,
        "preprocessing_version": report.preprocessing_version,
        "preprocessing_config_hash": report.preprocessing_config_hash,
        "dataset_manifest_path": report.dataset_manifest_path,
        "dataset_manifest_sha256": report.dataset_manifest_sha256,
        "n_samples": report.n_samples,
        "class_names": report.class_names,
        "threshold": report.threshold,
        "evaluated_at": report.evaluated_at,
        "metrics": report.metrics,
        "confidence_intervals": {
            key: _metric_summary_to_dict(summary)
            for key, summary in report.confidence_intervals.items()
        },
        "confusion_matrix": report.confusion_matrix,
        "roc_curve": curve_set_to_dict(report.roc_curve),
        "pr_curve": curve_set_to_dict(report.pr_curve),
    }
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    written["json"] = json_path

    csv_path = output_dir / "evaluation_report.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value", "ci_lower", "ci_upper", "confidence_level"])
        for metric_name, value in sorted(report.metrics.items()):
            summary = report.confidence_intervals.get(metric_name)
            writer.writerow(
                [
                    metric_name,
                    value,
                    summary.ci_lower if summary else "",
                    summary.ci_upper if summary else "",
                    summary.confidence_level if summary else "",
                ]
            )
    written["csv"] = csv_path

    predictions_path = output_dir / "predictions.csv"
    if report.predictions:
        with predictions_path.open("w", newline="", encoding="utf-8") as fh:
            dict_writer = csv.DictWriter(fh, fieldnames=list(report.predictions[0].keys()))
            dict_writer.writeheader()
            dict_writer.writerows(report.predictions)
        written["predictions"] = predictions_path

    markdown_path = output_dir / "evaluation_report.md"
    markdown_path.write_text(_render_markdown(report, written), encoding="utf-8")
    written["markdown"] = markdown_path

    logger.info(
        "evaluation_report_rendered",
        output_dir=str(output_dir),
        n_samples=report.n_samples,
        accuracy=report.metrics.get("accuracy"),
    )
    return written


def _render_markdown(report: EvaluationReport, written: dict[str, Path]) -> str:
    preprocessing_line = f"- **Preprocessing version:** {report.preprocessing_version or 'n/a'}"
    if report.preprocessing_config_hash:
        preprocessing_line += f" (config hash `{report.preprocessing_config_hash[:12]}`)"

    class_labels = ", ".join(f"{i}={name}" for i, name in enumerate(report.class_names))
    threshold_text = report.threshold if report.threshold is not None else "argmax (default)"

    lines: list[str] = [
        "# Evaluation Report",
        "",
        f"- **Model version:** {report.model_version} "
        f"(config hash `{report.model_config_hash[:12]}`)",
        f"- **Checkpoint:** `{report.checkpoint_path}`",
        preprocessing_line,
        f"- **Dataset:** `{report.dataset_manifest_path}` "
        f"(sha256 `{report.dataset_manifest_sha256[:12]}`, {report.n_samples} sample(s))",
        f"- **Class labels:** {class_labels}",
        f"- **Decision threshold:** {threshold_text}",
        f"- **Evaluated at:** {report.evaluated_at}",
        "",
        "## Headline metrics",
        "",
        "| Metric | Value | 95% CI |",
        "|---|---|---|",
    ]
    headline = ["accuracy", "precision_macro", "recall_macro", "specificity_macro", "f1_macro"]
    for key in ("roc_auc", "roc_auc_macro"):
        if key in report.metrics:
            headline.append(key)
    for key in ("pr_auc", "pr_auc_macro"):
        if key in report.metrics:
            headline.append(key)
    for key in headline:
        if key not in report.metrics:
            continue
        summary = report.confidence_intervals.get(key)
        ci = (
            f"[{summary.ci_lower:.4f}, {summary.ci_upper:.4f}]"
            if summary and summary.ci_lower is not None
            else "n/a"
        )
        lines.append(f"| {key} | {report.metrics[key]:.4f} | {ci} |")

    lines += [
        "",
        "## Per-class metrics",
        "",
        "| Class | Precision | Recall | Specificity | F1 |",
        "|---|---|---|---|---|",
    ]
    for name in report.class_names:
        lines.append(
            f"| {name} | {report.metrics.get(f'precision_{name}', float('nan')):.4f} "
            f"| {report.metrics.get(f'recall_{name}', float('nan')):.4f} "
            f"| {report.metrics.get(f'specificity_{name}', float('nan')):.4f} "
            f"| {report.metrics.get(f'f1_{name}', float('nan')):.4f} |"
        )

    lines += [
        "",
        "## Artifacts",
        "",
    ]
    for name, path in sorted(written.items()):
        if name == "markdown":
            continue
        lines.append(f"- **{name}:** `{path.name}`")

    lines += [
        "",
        "## Classification report",
        "",
        "```",
        report.classification_report_text.rstrip(),
        "```",
        "",
    ]
    return "\n".join(lines) + "\n"
