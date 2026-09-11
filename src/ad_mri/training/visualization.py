"""Training curve visualization: loss and a headline metric over epochs.

Rendered once at the end of a run (and logged as an MLflow artifact by
`Trainer`) so training behavior can be sanity-checked visually — e.g.
overfitting (train/val loss diverging) or a learning-rate schedule that's too
aggressive. See docs/TECHNICAL_SPECIFICATION.md §5, §9.

`matplotlib` is a training-phase dependency (requirements/train.txt), imported
lazily so this module stays importable without it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def plot_training_curves(
    history: list[dict[str, float]],
    output_path: Path,
    metric_name: str = "accuracy",
) -> Path:
    """Render train/val loss and `metric_name` (e.g. "accuracy", "f1_macro")
    over epochs as a two-panel PNG.

    `history` is a list of one dict per completed epoch, as produced by
    `Trainer.fit()` — expected keys: `epoch`, `train_loss`, `val_loss`, and
    `f"val_{metric_name}"` (falls back to loss-only if the metric is absent).
    """
    import matplotlib

    matplotlib.use("Agg")  # headless-safe: never assumes a display is present
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    train_loss = [row.get("train_loss") for row in history]
    val_loss = [row.get("val_loss") for row in history]
    metric_key = f"val_{metric_name}"
    has_metric = all(metric_key in row for row in history) and len(history) > 0

    num_panels = 2 if has_metric else 1
    fig, axes = plt.subplots(1, num_panels, figsize=(6 * num_panels, 4.5))
    axes_list: list[Any] = list(axes) if num_panels > 1 else [axes]

    loss_ax = axes_list[0]
    loss_ax.plot(epochs, train_loss, label="train_loss", marker="o", markersize=3)
    loss_ax.plot(epochs, val_loss, label="val_loss", marker="o", markersize=3)
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")
    loss_ax.set_title("Training / Validation Loss")
    loss_ax.legend()
    loss_ax.grid(alpha=0.3)

    if has_metric:
        metric_ax = axes_list[1]
        metric_values = [row[metric_key] for row in history]
        metric_ax.plot(
            epochs, metric_values, label=metric_key, color="darkorange", marker="o", markersize=3
        )
        metric_ax.set_xlabel("Epoch")
        metric_ax.set_ylabel(metric_name)
        metric_ax.set_title(f"Validation {metric_name}")
        metric_ax.legend()
        metric_ax.grid(alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path
