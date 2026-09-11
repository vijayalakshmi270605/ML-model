"""Confusion matrix computation and rendering.

`scikit-learn`/`matplotlib` are training-phase dependencies
(requirements/train.txt), imported lazily so this module stays importable
without them.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


def compute_confusion_matrix(
    y_true: Sequence[int], y_pred: Sequence[int], class_names: Sequence[str]
) -> list[list[int]]:
    """Rows are true labels, columns are predicted labels, in `class_names` order."""
    from sklearn.metrics import confusion_matrix

    labels = list(range(len(class_names)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return matrix.tolist()


def render_confusion_matrix(
    matrix: list[list[int]], class_names: Sequence[str], output_path: Path
) -> Path:
    """Render a confusion matrix as an annotated heatmap PNG."""
    import matplotlib

    matplotlib.use("Agg")  # headless-safe: never assumes a display is present
    import matplotlib.pyplot as plt
    import numpy as np

    array = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(1.2 * len(class_names) + 2, 1.2 * len(class_names) + 2))
    im = ax.imshow(array, cmap="Blues")
    fig.colorbar(im, ax=ax, label="Count")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    max_count = array.max() if array.size else 0
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            color = "white" if max_count and array[i, j] > max_count / 2 else "black"
            ax.text(j, i, str(array[i, j]), ha="center", va="center", color=color)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path
