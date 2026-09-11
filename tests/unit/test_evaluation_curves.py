"""Tests for evaluation.curves — checked against hand-verifiable examples."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("matplotlib")

from ad_mri.evaluation.curves import (  # noqa: E402
    compute_pr_curve,
    compute_roc_curve,
    curve_set_to_dict,
    render_pr_curve,
    render_roc_curve,
)


def test_binary_roc_curve_has_a_single_perfectly_separated_curve() -> None:
    y_true = [0, 0, 1, 1]
    y_proba = [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]]
    curve_set = compute_roc_curve(y_true, y_proba, class_names=["CN", "AD"])
    assert curve_set.kind == "roc"
    assert len(curve_set.curves) == 1
    assert curve_set.curves[0].name == "AD"
    assert curve_set.curves[0].area == pytest.approx(1.0)


def test_binary_pr_curve_has_a_single_perfectly_separated_curve() -> None:
    y_true = [0, 0, 1, 1]
    y_proba = [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]]
    curve_set = compute_pr_curve(y_true, y_proba, class_names=["CN", "AD"])
    assert len(curve_set.curves) == 1
    assert curve_set.curves[0].area == pytest.approx(1.0)


def test_multiclass_roc_curve_has_one_curve_per_present_class() -> None:
    y_true = [0, 1, 2, 0, 1, 2]
    y_proba = [
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.7, 0.2, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.2, 0.7],
    ]
    curve_set = compute_roc_curve(y_true, y_proba, class_names=["CN", "MCI", "AD"])
    assert {c.name for c in curve_set.curves} == {"CN", "MCI", "AD"}
    assert all(c.area == pytest.approx(1.0) for c in curve_set.curves)


def test_curve_omitted_for_a_class_absent_from_y_true() -> None:
    y_true = [0, 0, 0, 0]  # "AD" never appears
    y_proba = [[0.9, 0.1]] * 4
    curve_set = compute_roc_curve(y_true, y_proba, class_names=["CN", "AD"])
    assert curve_set.curves == []  # binary path: the one curve needs the positive class present


def test_render_roc_curve_writes_a_png(tmp_path: Path) -> None:
    y_true = [0, 0, 1, 1]
    y_proba = [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]]
    curve_set = compute_roc_curve(y_true, y_proba, class_names=["CN", "AD"])
    output_path = render_roc_curve(curve_set, tmp_path / "roc.png")
    assert output_path is not None
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_render_pr_curve_writes_a_png(tmp_path: Path) -> None:
    y_true = [0, 0, 1, 1]
    y_proba = [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]]
    curve_set = compute_pr_curve(y_true, y_proba, class_names=["CN", "AD"])
    output_path = render_pr_curve(curve_set, tmp_path / "pr.png")
    assert output_path is not None
    assert output_path.exists()


def test_render_returns_none_and_writes_nothing_when_no_curves(tmp_path: Path) -> None:
    y_true = [0, 0, 0, 0]
    y_proba = [[0.9, 0.1]] * 4
    curve_set = compute_roc_curve(y_true, y_proba, class_names=["CN", "AD"])
    output_path = tmp_path / "roc.png"
    result = render_roc_curve(curve_set, output_path)
    assert result is None
    assert not output_path.exists()


def test_curve_set_to_dict_is_json_serializable() -> None:
    import json

    y_true = [0, 0, 1, 1]
    y_proba = [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]]
    curve_set = compute_roc_curve(y_true, y_proba, class_names=["CN", "AD"])
    payload = curve_set_to_dict(curve_set)
    serialized = json.dumps(payload)  # must not raise
    assert json.loads(serialized)["kind"] == "roc"
