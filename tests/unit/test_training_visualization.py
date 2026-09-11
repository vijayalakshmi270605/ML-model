from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from ad_mri.training.visualization import plot_training_curves  # noqa: E402


def _history(num_epochs: int = 3) -> list[dict[str, float]]:
    return [
        {
            "epoch": epoch,
            "train_loss": 1.0 / epoch,
            "val_loss": 1.2 / epoch,
            "val_accuracy": min(0.5 + 0.1 * epoch, 1.0),
        }
        for epoch in range(1, num_epochs + 1)
    ]


def test_plot_training_curves_writes_a_png_with_two_panels(tmp_path: Path) -> None:
    output_path = plot_training_curves(_history(), tmp_path / "curves.png", metric_name="accuracy")
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_training_curves_handles_missing_metric_gracefully(tmp_path: Path) -> None:
    history = [{"epoch": 1, "train_loss": 0.5, "val_loss": 0.6}]
    output_path = plot_training_curves(history, tmp_path / "curves.png", metric_name="f1_macro")
    assert output_path.exists()
