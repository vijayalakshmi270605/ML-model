"""Tests for evaluation.statistics — checked against hand-verifiable values."""

from __future__ import annotations

import pytest

pytest.importorskip("scipy")

from ad_mri.evaluation.statistics import (  # noqa: E402
    bootstrap_metric_ci,
    summarize_fold_metrics,
    summarize_values,
)


def test_mean_matches_hand_calculation() -> None:
    summary = summarize_values("acc", [0.8, 0.9, 0.7, 0.85, 0.75])
    assert summary.mean == pytest.approx(0.8)
    assert summary.n == 5


def test_std_matches_hand_calculation() -> None:
    # values = [2, 4, 4, 4, 5, 5, 7, 9]; sample std (ddof=1) = 2.13809...
    values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    summary = summarize_values("x", values)
    assert summary.mean == pytest.approx(5.0)
    assert summary.std == pytest.approx(2.13809, abs=1e-4)


def test_single_value_has_no_std_or_ci() -> None:
    summary = summarize_values("acc", [0.9])
    assert summary.mean == 0.9
    assert summary.std is None
    assert summary.ci_lower is None
    assert summary.ci_upper is None


def test_ci_widens_with_higher_confidence_level() -> None:
    values = [0.7, 0.75, 0.8, 0.85, 0.9]
    ci_95 = summarize_values("acc", values, confidence_level=0.95)
    ci_99 = summarize_values("acc", values, confidence_level=0.99)
    width_95 = ci_95.ci_upper - ci_95.ci_lower  # type: ignore[operator]
    width_99 = ci_99.ci_upper - ci_99.ci_lower  # type: ignore[operator]
    assert width_99 > width_95


def test_ci_is_symmetric_around_the_mean() -> None:
    summary = summarize_values("acc", [0.6, 0.7, 0.8, 0.9])
    lower_gap = summary.mean - summary.ci_lower  # type: ignore[operator]
    upper_gap = summary.ci_upper - summary.mean  # type: ignore[operator]
    assert lower_gap == pytest.approx(upper_gap)


def test_summarize_values_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="No values"):
        summarize_values("acc", [])


def test_summarize_fold_metrics_covers_common_keys_only() -> None:
    fold_metrics = [
        {"loss": 0.5, "accuracy": 0.8, "roc_auc": 0.9},
        {"loss": 0.4, "accuracy": 0.85},  # no roc_auc this fold
    ]
    summary = summarize_fold_metrics(fold_metrics)
    assert set(summary) == {"loss", "accuracy"}
    assert summary["accuracy"].mean == pytest.approx(0.825)


def test_summarize_fold_metrics_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="No fold metrics"):
        summarize_fold_metrics([])


def _accuracy(y_true: list[int], y_pred: list[int]) -> float:
    correct = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == p)
    return correct / len(y_true)


def test_bootstrap_point_estimate_matches_hand_calculation() -> None:
    # 8 samples, 6 correct -> accuracy = 0.75, independent of resampling
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    y_pred = [0, 0, 0, 1, 1, 1, 1, 0]
    summary = bootstrap_metric_ci(
        _accuracy, y_true, y_pred, metric_name="accuracy", n_bootstrap=200
    )
    assert summary.mean == pytest.approx(0.75)
    assert summary.n == 8


def test_bootstrap_ci_contains_the_point_estimate() -> None:
    y_true = [0, 0, 0, 0, 1, 1, 1, 1, 0, 1]
    y_pred = [0, 0, 0, 1, 1, 1, 1, 0, 0, 1]
    summary = bootstrap_metric_ci(
        _accuracy, y_true, y_pred, metric_name="accuracy", n_bootstrap=500
    )
    assert summary.ci_lower is not None and summary.ci_upper is not None
    assert summary.ci_lower <= summary.mean <= summary.ci_upper


def test_bootstrap_is_reproducible_given_the_same_seed() -> None:
    y_true = [0, 0, 1, 1, 0, 1, 1, 0]
    y_pred = [0, 1, 1, 1, 0, 0, 1, 0]
    first = bootstrap_metric_ci(
        _accuracy, y_true, y_pred, metric_name="acc", n_bootstrap=100, seed=7
    )
    second = bootstrap_metric_ci(
        _accuracy, y_true, y_pred, metric_name="acc", n_bootstrap=100, seed=7
    )
    assert first.ci_lower == second.ci_lower
    assert first.ci_upper == second.ci_upper
    assert first.values == second.values


def test_bootstrap_different_seeds_can_give_different_intervals() -> None:
    y_true = [0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1]
    y_pred = [0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0]
    first = bootstrap_metric_ci(
        _accuracy, y_true, y_pred, metric_name="acc", n_bootstrap=50, seed=1
    )
    second = bootstrap_metric_ci(
        _accuracy, y_true, y_pred, metric_name="acc", n_bootstrap=50, seed=2
    )
    assert first.values != second.values


def test_bootstrap_single_sample_has_no_ci() -> None:
    summary = bootstrap_metric_ci(_accuracy, [1], [1], metric_name="acc")
    assert summary.mean == 1.0
    assert summary.ci_lower is None
    assert summary.ci_upper is None


def test_bootstrap_rejects_mismatched_array_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        bootstrap_metric_ci(_accuracy, [0, 1], [0, 1, 1], metric_name="acc")


def test_bootstrap_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="No values"):
        bootstrap_metric_ci(_accuracy, [], [], metric_name="acc")


def test_bootstrap_supports_multiple_parallel_arrays() -> None:
    # metric_fn receiving (y_true, y_proba) instead of (y_true, y_pred)
    def _mean_positive_score(y_true: list[int], y_proba: list[list[float]]) -> float:
        return sum(row[1] for row in y_proba) / len(y_proba)

    y_true = [0, 1, 0, 1]
    y_proba = [[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]]
    summary = bootstrap_metric_ci(
        _mean_positive_score, y_true, y_proba, metric_name="mean_score", n_bootstrap=100
    )
    assert summary.mean == pytest.approx((0.1 + 0.8 + 0.3 + 0.9) / 4)
