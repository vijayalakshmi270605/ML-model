from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("matplotlib")

from ad_mri.evaluation.confusion_matrix import (  # noqa: E402
    compute_confusion_matrix,
    render_confusion_matrix,
)


def test_compute_confusion_matrix_matches_hand_calculation() -> None:
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    # row 0 (true CN): 1 predicted CN, 1 predicted AD
    # row 1 (true AD): 0 predicted CN, 2 predicted AD
    matrix = compute_confusion_matrix(y_true, y_pred, class_names=["CN", "AD"])
    assert matrix == [[1, 1], [0, 2]]


def test_compute_confusion_matrix_perfect_predictions_is_diagonal() -> None:
    y_true = [0, 1, 2, 0, 1, 2]
    matrix = compute_confusion_matrix(y_true, y_true, class_names=["CN", "MCI", "AD"])
    assert matrix == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]


def test_render_confusion_matrix_writes_a_png(tmp_path: Path) -> None:
    matrix = [[1, 1], [0, 2]]
    output_path = render_confusion_matrix(
        matrix, class_names=["CN", "AD"], output_path=tmp_path / "cm.png"
    )
    assert output_path.exists()
    assert output_path.stat().st_size > 0
