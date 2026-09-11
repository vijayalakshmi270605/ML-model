"""Tests for `monitoring.model_drift` - real metric computation
(`evaluation.metrics.compute_classification_metrics`), no mocks."""

from __future__ import annotations

from ad_mri.monitoring.model_drift import compute_delayed_performance
from ad_mri.monitoring.thresholds import MIN_LABELED_SAMPLES_FOR_MODEL_DRIFT


def _perfect_pairs(n: int) -> list[tuple[str, str]]:
    return [("AD", "AD") if i % 2 == 0 else ("CN", "CN") for i in range(n)]


class TestInsufficientSamples:
    def test_reports_insufficient_samples_below_the_floor(self) -> None:
        pairs = _perfect_pairs(MIN_LABELED_SAMPLES_FOR_MODEL_DRIFT - 1)
        report = compute_delayed_performance("model-1", pairs, ["AD", "CN"], baseline_metrics=None)

        assert report.insufficient_samples is True
        assert report.degraded is False
        assert report.current_metrics == {}
        assert "Insufficient" in report.to_markdown()

    def test_computes_a_verdict_at_exactly_the_floor(self) -> None:
        pairs = _perfect_pairs(MIN_LABELED_SAMPLES_FOR_MODEL_DRIFT)
        report = compute_delayed_performance("model-1", pairs, ["AD", "CN"], baseline_metrics=None)
        assert report.insufficient_samples is False


class TestNoBaseline:
    def test_computes_current_metrics_without_a_degradation_verdict(self) -> None:
        pairs = _perfect_pairs(50)
        report = compute_delayed_performance("model-1", pairs, ["AD", "CN"], baseline_metrics=None)

        assert report.insufficient_samples is False
        assert report.current_metrics["accuracy"] == 1.0
        assert report.degraded is False  # nothing to compare against
        assert report.degraded_metrics == []
        assert "No baseline_metrics" in report.to_markdown()


class TestDegradationDetection:
    def test_no_degradation_when_current_matches_baseline(self) -> None:
        pairs = _perfect_pairs(50)  # accuracy 1.0
        report = compute_delayed_performance(
            "model-1", pairs, ["AD", "CN"], baseline_metrics={"accuracy": 1.0, "f1_macro": 1.0}
        )
        assert report.degraded is False
        assert "no significant degradation" in report.to_markdown()

    def test_flags_a_real_accuracy_drop(self) -> None:
        # 30 correct, 20 wrong -> accuracy 0.6, an 0.35 drop from a 0.95 baseline.
        pairs = [("AD", "AD")] * 30 + [("AD", "CN")] * 20
        report = compute_delayed_performance(
            "model-1", pairs, ["AD", "CN"], baseline_metrics={"accuracy": 0.95, "f1_macro": 0.95}
        )
        assert report.degraded is True
        assert "accuracy" in report.degraded_metrics
        assert "DEGRADED" in report.to_markdown()

    def test_a_small_drop_below_threshold_does_not_flag(self) -> None:
        # 1 error spread across each class (49/50 correct = 0.98 accuracy,
        # f1_macro also ~0.98) vs. a 1.0 baseline - a small, real drop well
        # under the 0.05 threshold, on both monitored metrics.
        pairs = [("AD", "AD")] * 24 + [("AD", "CN")] + [("CN", "CN")] * 24 + [("CN", "AD")]
        report = compute_delayed_performance(
            "model-1", pairs, ["AD", "CN"], baseline_metrics={"accuracy": 1.0, "f1_macro": 1.0}
        )
        assert report.degraded is False

    def test_unknown_labels_are_skipped_not_fatal(self) -> None:
        pairs = _perfect_pairs(50) + [("Unknown", "AD")] * 5
        report = compute_delayed_performance("model-1", pairs, ["AD", "CN"], baseline_metrics=None)
        assert report.insufficient_samples is False
        # the 5 unknown-label pairs are excluded from the metric computation
        assert report.current_metrics["accuracy"] == 1.0


class TestReportRendering:
    def test_save_writes_real_json_and_markdown_files(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        pairs = _perfect_pairs(50)
        report = compute_delayed_performance(
            "model-1", pairs, ["AD", "CN"], baseline_metrics={"accuracy": 1.0, "f1_macro": 1.0}
        )
        paths = report.save(tmp_path)
        assert paths["json"].exists()
        assert paths["markdown"].exists()
        assert "Model Drift Report" in paths["markdown"].read_text(encoding="utf-8")
