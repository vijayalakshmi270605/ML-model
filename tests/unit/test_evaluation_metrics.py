"""Tests for evaluation.metrics — checked against hand-verifiable examples,
not just "some value came back"."""

from __future__ import annotations

import pytest

pytest.importorskip("sklearn")

from ad_mri.core.exceptions import ConfigurationError  # noqa: E402
from ad_mri.evaluation.metrics import (  # noqa: E402
    apply_binary_threshold,
    compute_classification_metrics,
    format_metrics,
)


def test_accuracy_matches_hand_calculation() -> None:
    # 4 samples, 3 correct -> accuracy = 0.75
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    metrics = compute_classification_metrics(y_true, y_pred, class_names=["CN", "AD"])
    assert metrics["accuracy"] == pytest.approx(0.75)


def test_per_class_precision_recall_match_hand_calculation() -> None:
    # class 0 ("CN"): 2 true, 1 predicted (correctly) -> precision=1.0, recall=0.5
    # class 1 ("AD"): 2 true, 3 predicted (2 correct)  -> precision=2/3, recall=1.0
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    metrics = compute_classification_metrics(y_true, y_pred, class_names=["CN", "AD"])

    assert metrics["precision_CN"] == pytest.approx(1.0)
    assert metrics["recall_CN"] == pytest.approx(0.5)
    assert metrics["precision_AD"] == pytest.approx(2 / 3)
    assert metrics["recall_AD"] == pytest.approx(1.0)


def test_perfect_predictions_give_perfect_scores() -> None:
    y_true = [0, 1, 2, 0, 1, 2]
    metrics = compute_classification_metrics(y_true, y_true, class_names=["CN", "MCI", "AD"])
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["f1_macro"] == pytest.approx(1.0)


def test_roc_auc_binary_is_computed_when_proba_given() -> None:
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    y_proba = [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]]
    metrics = compute_classification_metrics(
        y_true, y_pred, class_names=["CN", "AD"], y_proba=y_proba
    )
    assert "roc_auc" in metrics
    assert metrics["roc_auc"] == pytest.approx(1.0)  # perfectly separated scores


def test_roc_auc_omitted_when_a_class_is_absent_from_y_true() -> None:
    y_true = [0, 0, 0, 0]  # class "AD" never appears
    y_pred = [0, 0, 0, 0]
    y_proba = [[0.9, 0.1]] * 4
    metrics = compute_classification_metrics(
        y_true, y_pred, class_names=["CN", "AD"], y_proba=y_proba
    )
    assert "roc_auc" not in metrics  # omitted, not fabricated as 0.0/NaN


def test_roc_auc_absent_without_proba() -> None:
    metrics = compute_classification_metrics([0, 1], [0, 1], class_names=["CN", "AD"])
    assert "roc_auc" not in metrics
    assert "roc_auc_macro" not in metrics


def test_multiclass_roc_auc_macro_is_computed() -> None:
    y_true = [0, 1, 2, 0, 1, 2]
    y_proba = [
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.7, 0.2, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.2, 0.7],
    ]
    metrics = compute_classification_metrics(
        y_true, y_true, class_names=["CN", "MCI", "AD"], y_proba=y_proba
    )
    assert "roc_auc_macro" in metrics
    assert metrics["roc_auc_macro"] == pytest.approx(1.0)


def test_format_metrics_renders_known_keys() -> None:
    text = format_metrics({"accuracy": 0.5, "f1_macro": 0.4, "roc_auc": 0.9})
    assert "accuracy=0.5000" in text
    assert "f1_macro=0.4000" in text
    assert "roc_auc=0.9000" in text


def test_specificity_matches_hand_calculation() -> None:
    # confusion matrix (CN=0, AD=1): [[TN=1,FP=1],[FN=0,TP=2]]
    # CN specificity = TN/(TN+FP) for class CN treated as negative-of-AD... compute directly:
    # class CN (index 0): TP=1, FN=1, FP=0, TN=2 -> specificity = 2/2 = 1.0
    # class AD (index 1): TP=2, FN=0, FP=1, TN=1 -> specificity = 1/2 = 0.5
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    metrics = compute_classification_metrics(y_true, y_pred, class_names=["CN", "AD"])
    assert metrics["specificity_CN"] == pytest.approx(1.0)
    assert metrics["specificity_AD"] == pytest.approx(0.5)
    assert metrics["specificity_macro"] == pytest.approx(0.75)


def test_perfect_predictions_give_perfect_specificity() -> None:
    y_true = [0, 1, 2, 0, 1, 2]
    metrics = compute_classification_metrics(y_true, y_true, class_names=["CN", "MCI", "AD"])
    assert metrics["specificity_macro"] == pytest.approx(1.0)


def test_pr_auc_binary_is_computed_when_proba_given() -> None:
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    y_proba = [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]]
    metrics = compute_classification_metrics(
        y_true, y_pred, class_names=["CN", "AD"], y_proba=y_proba
    )
    assert metrics["pr_auc"] == pytest.approx(1.0)  # perfectly separated scores


def test_pr_auc_omitted_when_a_class_is_absent_from_y_true() -> None:
    y_true = [0, 0, 0, 0]
    y_pred = [0, 0, 0, 0]
    y_proba = [[0.9, 0.1]] * 4
    metrics = compute_classification_metrics(
        y_true, y_pred, class_names=["CN", "AD"], y_proba=y_proba
    )
    assert "pr_auc" not in metrics


def test_multiclass_pr_auc_macro_is_computed() -> None:
    y_true = [0, 1, 2, 0, 1, 2]
    y_proba = [
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.7, 0.2, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.2, 0.7],
    ]
    metrics = compute_classification_metrics(
        y_true, y_true, class_names=["CN", "MCI", "AD"], y_proba=y_proba
    )
    assert "pr_auc_macro" in metrics
    assert metrics["pr_auc_macro"] == pytest.approx(1.0)


def test_apply_binary_threshold_matches_hand_calculation() -> None:
    y_proba = [[0.9, 0.1], [0.6, 0.4], [0.4, 0.6], [0.1, 0.9]]
    # default-equivalent threshold 0.5 -> [0, 0, 1, 1]
    assert apply_binary_threshold(y_proba, 0.5) == [0, 0, 1, 1]
    # stricter threshold 0.7 -> only the last row clears it
    assert apply_binary_threshold(y_proba, 0.7) == [0, 0, 0, 1]
    # threshold 0.0 -> everything is positive
    assert apply_binary_threshold(y_proba, 0.0) == [1, 1, 1, 1]


def test_apply_binary_threshold_rejects_out_of_range_value() -> None:
    with pytest.raises(ConfigurationError):
        apply_binary_threshold([[0.5, 0.5]], 1.5)
    with pytest.raises(ConfigurationError):
        apply_binary_threshold([[0.5, 0.5]], -0.1)
