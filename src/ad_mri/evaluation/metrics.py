"""Classification evaluation metrics: accuracy, precision/recall/specificity/F1
(macro + weighted + per-class), ROC-AUC, and PR-AUC where applicable.

Used identically by the training loop's per-epoch validation (Phase 5) and by
the standalone evaluation CLI (Phase 7, `evaluation.evaluate`) — one
implementation, not duplicated. See docs/TECHNICAL_SPECIFICATION.md §10,
docs/evaluation.md.

`scikit-learn` is a training-phase dependency (requirements/train.txt),
imported lazily so this module stays importable without it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ad_mri.core.exceptions import ConfigurationError


def _specificity_per_class(confusion: Any) -> list[float]:
    """Specificity (true-negative rate) per class, one-vs-rest, from a
    `[n_classes, n_classes]` confusion matrix (rows=true, cols=predicted)."""
    total = confusion.sum()
    specificities: list[float] = []
    for i in range(confusion.shape[0]):
        true_positive = confusion[i, i]
        false_negative = confusion[i, :].sum() - true_positive
        false_positive = confusion[:, i].sum() - true_positive
        true_negative = total - true_positive - false_negative - false_positive
        denominator = true_negative + false_positive
        specificities.append(float(true_negative / denominator) if denominator > 0 else 0.0)
    return specificities


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    y_proba: Sequence[Sequence[float]] | None = None,
) -> dict[str, float]:
    """Compute accuracy, macro/weighted/per-class precision/recall/specificity/F1,
    and (if `y_proba` is given) ROC-AUC and PR-AUC.

    ROC-AUC/PR-AUC are "applicable" (computed) only when `y_proba` is given and
    every class in `class_names` is present in `y_true` at least once —
    scikit-learn cannot define either curve for a class with zero positive
    samples, so they're silently omitted (not fabricated as 0.0/NaN) in that case.

    Returns a flat `{metric_name: value}` dict — e.g. `"precision_CN"`,
    `"specificity_macro"`, `"f1_macro"`, `"roc_auc_macro"`, `"pr_auc"` — ready
    to hand to `mlflow.log_metrics` or a JSON/CSV report.
    """
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    labels = list(range(len(class_names)))
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }

    confusion = confusion_matrix(y_true, y_pred, labels=labels)
    per_class_specificity = _specificity_per_class(confusion)
    support = confusion.sum(axis=1)  # true-class counts, for weighted averaging
    total_support = support.sum()

    for average in ("macro", "weighted"):
        metrics[f"precision_{average}"] = float(
            precision_score(y_true, y_pred, labels=labels, average=average, zero_division=0)
        )
        metrics[f"recall_{average}"] = float(
            recall_score(y_true, y_pred, labels=labels, average=average, zero_division=0)
        )
        metrics[f"f1_{average}"] = float(
            f1_score(y_true, y_pred, labels=labels, average=average, zero_division=0)
        )
        if average == "macro":
            metrics["specificity_macro"] = float(sum(per_class_specificity) / len(labels))
        elif total_support > 0:
            metrics["specificity_weighted"] = float(
                sum(s * w for s, w in zip(per_class_specificity, support, strict=True))
                / total_support
            )

    per_class_precision = precision_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_class_recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    for name, precision, recall, f1, specificity in zip(
        class_names,
        per_class_precision,
        per_class_recall,
        per_class_f1,
        per_class_specificity,
        strict=True,
    ):
        metrics[f"precision_{name}"] = float(precision)
        metrics[f"recall_{name}"] = float(recall)
        metrics[f"specificity_{name}"] = float(specificity)
        metrics[f"f1_{name}"] = float(f1)

    if y_proba is not None:
        classes_present = set(y_true)
        if len(classes_present) == len(class_names):
            try:
                if len(class_names) == 2:
                    positive_scores = [row[1] for row in y_proba]
                    metrics["roc_auc"] = float(roc_auc_score(y_true, positive_scores))
                    metrics["pr_auc"] = float(average_precision_score(y_true, positive_scores))
                else:
                    metrics["roc_auc_macro"] = float(
                        roc_auc_score(
                            y_true, y_proba, multi_class="ovr", average="macro", labels=labels
                        )
                    )
                    per_class_ap = []
                    for class_index, name in enumerate(class_names):
                        binary_true = [1 if label == class_index else 0 for label in y_true]
                        class_scores = [row[class_index] for row in y_proba]
                        ap = float(average_precision_score(binary_true, class_scores))
                        metrics[f"pr_auc_{name}"] = ap
                        per_class_ap.append(ap)
                    metrics["pr_auc_macro"] = float(sum(per_class_ap) / len(per_class_ap))
            except ValueError:
                pass  # degenerate proba matrix (e.g. all-identical scores) — omit, don't fabricate

    return metrics


def apply_binary_threshold(
    y_proba: Sequence[Sequence[float]], threshold: float, positive_index: int = 1
) -> list[int]:
    """Binary-only decision rule: predict `positive_index` when its probability
    is `>= threshold`, else the other class. `threshold=0.5` is equivalent to
    plain argmax over a 2-class softmax — this is a strict generalization, not
    a separate code path.

    Raises:
        ConfigurationError: `threshold` is outside `[0, 1]`.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ConfigurationError(f"threshold must be in [0, 1], got {threshold!r}.")
    negative_index = 1 - positive_index
    return [
        positive_index if row[positive_index] >= threshold else negative_index for row in y_proba
    ]


def format_metrics(metrics: dict[str, float], prefix: str = "") -> str:
    """One-line, human-readable rendering for training logs (not for MLflow —
    that gets the raw dict via `mlflow.log_metrics`)."""
    ordered_keys: Sequence[str] = (
        "accuracy",
        "precision_macro",
        "recall_macro",
        "specificity_macro",
        "f1_macro",
        "roc_auc",
        "roc_auc_macro",
        "pr_auc",
        "pr_auc_macro",
    )
    parts: list[Any] = [
        f"{prefix}{key}={metrics[key]:.4f}" for key in ordered_keys if key in metrics
    ]
    return " ".join(parts)
