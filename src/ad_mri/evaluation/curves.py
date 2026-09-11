"""ROC and Precision-Recall curve computation and rendering.

Binary classification gets a single curve (positive class = `class_names[1]`).
Multiclass gets one one-vs-rest curve per class plus a macro-averaged curve,
mirroring `evaluation.metrics`' one-vs-rest ROC-AUC/PR-AUC. A class absent
from `y_true` has no curve defined for it — it's omitted with a warning, never
fabricated. See docs/TECHNICAL_SPECIFICATION.md §10, docs/evaluation.md.

`scikit-learn`/`matplotlib`/`numpy` are training-phase dependencies
(requirements/train.txt), imported lazily so this module stays importable
without them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ad_mri.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class CurveData:
    """One ROC or PR curve: `x`/`y` are the plotted axes (fpr/tpr for ROC,
    recall/precision for PR), `area` is the corresponding AUC/average-precision."""

    name: str
    x: list[float]
    y: list[float]
    area: float


@dataclass(frozen=True)
class CurveSet:
    kind: str  # "roc" | "pr"
    curves: list[CurveData] = field(default_factory=list)


def _binarize(y_true: Sequence[int], class_index: int) -> list[int]:
    return [1 if label == class_index else 0 for label in y_true]


def _class_indices_to_plot(class_names: Sequence[str]) -> list[int]:
    """Binary: only the positive class (index 1) — a 2-class one-vs-rest curve
    for index 0 would just be its mirror image, not a second, independent
    curve. Multiclass: every class index."""
    if len(class_names) == 2:
        return [1]
    return list(range(len(class_names)))


def compute_roc_curve(
    y_true: Sequence[int], y_proba: Sequence[Sequence[float]], class_names: Sequence[str]
) -> CurveSet:
    """One-vs-rest ROC curve(s). Binary: single curve for the positive class
    (index 1). Multiclass: one curve per class present in `y_true`."""
    from sklearn.metrics import auc, roc_curve

    classes_present = set(y_true)
    curves: list[CurveData] = []
    for class_index in _class_indices_to_plot(class_names):
        name = class_names[class_index]
        if class_index not in classes_present:
            logger.warning("roc_curve_class_absent_from_y_true", class_name=name)
            continue
        binary_true = _binarize(y_true, class_index)
        scores = [row[class_index] for row in y_proba]
        fpr, tpr, _ = roc_curve(binary_true, scores)
        curves.append(
            CurveData(name=name, x=fpr.tolist(), y=tpr.tolist(), area=float(auc(fpr, tpr)))
        )
    return CurveSet(kind="roc", curves=curves)


def compute_pr_curve(
    y_true: Sequence[int], y_proba: Sequence[Sequence[float]], class_names: Sequence[str]
) -> CurveSet:
    """One-vs-rest Precision-Recall curve(s). Binary: single curve for the
    positive class (index 1). Multiclass: one curve per class present in
    `y_true`."""
    from sklearn.metrics import average_precision_score, precision_recall_curve

    classes_present = set(y_true)
    curves: list[CurveData] = []
    for class_index in _class_indices_to_plot(class_names):
        name = class_names[class_index]
        if class_index not in classes_present:
            logger.warning("pr_curve_class_absent_from_y_true", class_name=name)
            continue
        binary_true = _binarize(y_true, class_index)
        scores = [row[class_index] for row in y_proba]
        precision, recall, _ = precision_recall_curve(binary_true, scores)
        area = float(average_precision_score(binary_true, scores))
        curves.append(CurveData(name=name, x=recall.tolist(), y=precision.tolist(), area=area))

    return CurveSet(kind="pr", curves=curves)


def _render_curve_set(
    curve_set: CurveSet,
    output_path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    diagonal_reference: bool,
) -> Path | None:
    if not curve_set.curves:
        logger.warning("curve_render_skipped_no_curves", kind=curve_set.kind)
        return None

    import matplotlib

    matplotlib.use("Agg")  # headless-safe: never assumes a display is present
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5.5))
    for curve in curve_set.curves:
        area_label = "AUC" if curve_set.kind == "roc" else "AP"
        ax.plot(curve.x, curve.y, label=f"{curve.name} ({area_label}={curve.area:.3f})")
    if diagonal_reference:
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1, label="chance")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title)
    ax.legend(loc="lower right" if diagonal_reference else "lower left", fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def render_roc_curve(curve_set: CurveSet, output_path: Path) -> Path | None:
    """Render ROC curve(s) as a PNG. Returns `None` (no file written) if no
    class had a defined curve."""
    return _render_curve_set(
        curve_set,
        output_path,
        title="ROC Curve",
        x_label="False Positive Rate",
        y_label="True Positive Rate",
        diagonal_reference=True,
    )


def render_pr_curve(curve_set: CurveSet, output_path: Path) -> Path | None:
    """Render Precision-Recall curve(s) as a PNG. Returns `None` (no file
    written) if no class had a defined curve."""
    return _render_curve_set(
        curve_set,
        output_path,
        title="Precision-Recall Curve",
        x_label="Recall",
        y_label="Precision",
        diagonal_reference=False,
    )


def curve_set_to_dict(curve_set: CurveSet) -> dict[str, Any]:
    """JSON-serializable form of a `CurveSet` (for the JSON evaluation report)."""
    return {
        "kind": curve_set.kind,
        "curves": [{"name": c.name, "x": c.x, "y": c.y, "area": c.area} for c in curve_set.curves],
    }
