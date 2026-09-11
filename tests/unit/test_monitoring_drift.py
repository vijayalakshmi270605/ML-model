"""Tests for `monitoring.drift` - real numpy/scipy computation, no mocks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ad_mri.monitoring.drift import (
    DriftReference,
    compute_image_statistics,
    generate_data_drift_report,
    population_stability_index,
)


class TestComputeImageStatistics:
    def test_computes_real_statistics_from_a_numpy_array(self) -> None:
        array = np.zeros((4, 4, 4), dtype=np.float32)
        array[0, 0, 0] = 1.0
        array[0, 0, 1] = 2.0

        stats = compute_image_statistics(array)

        assert stats.voxel_count == 64
        assert stats.nonzero_fraction == pytest.approx(2 / 64)
        assert stats.mean_intensity == pytest.approx(3.0 / 64)

    def test_accepts_a_torch_like_object_with_a_numpy_method(self) -> None:
        class FakeTensor:
            def numpy(self) -> np.ndarray:
                return np.ones((2, 2, 2), dtype=np.float32)

        stats = compute_image_statistics(FakeTensor())
        assert stats.mean_intensity == pytest.approx(1.0)
        assert stats.nonzero_fraction == pytest.approx(1.0)

    def test_empty_array_does_not_raise(self) -> None:
        stats = compute_image_statistics(np.zeros((0,), dtype=np.float32))
        assert stats.voxel_count == 0
        assert stats.nonzero_fraction == 0.0


class TestPopulationStabilityIndex:
    def test_identical_distributions_have_near_zero_psi(self) -> None:
        rng = np.random.default_rng(42)
        reference = rng.normal(0, 1, 2000)
        current = rng.normal(0, 1, 2000)
        psi = population_stability_index(reference, current)
        assert psi < 0.1

    def test_a_shifted_distribution_has_a_high_psi(self) -> None:
        rng = np.random.default_rng(42)
        reference = rng.normal(0, 1, 2000)
        current = rng.normal(5, 1, 2000)  # a real, large mean shift
        psi = population_stability_index(reference, current)
        assert psi >= 0.25

    def test_empty_inputs_return_zero_not_an_error(self) -> None:
        assert population_stability_index([], [1.0, 2.0]) == 0.0
        assert population_stability_index([1.0, 2.0], []) == 0.0

    def test_degenerate_reference_returns_zero(self) -> None:
        """Every reference value identical - no meaningful binning possible."""
        assert population_stability_index([5.0] * 100, [1.0, 2.0, 3.0]) == 0.0


class TestDriftReferenceRoundTrip:
    def test_save_and_load_round_trips_exactly(self, tmp_path: Path) -> None:
        reference = DriftReference(
            feature_samples={"mean_intensity": [0.1, 0.2, 0.3]},
            class_counts={"AD": 10, "CN": 12},
            n_samples=22,
            built_at="2026-01-01T00:00:00+00:00",
            source_manifest="manifest.csv",
        )
        path = tmp_path / "reference.json"
        reference.save(path)

        loaded = DriftReference.load(path)
        assert loaded == reference

    def test_load_raises_a_configuration_error_when_missing(self, tmp_path: Path) -> None:
        from ad_mri.core.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="build-drift-reference"):
            DriftReference.load(tmp_path / "does-not-exist.json")


class TestGenerateDataDriftReport:
    def _reference(self) -> DriftReference:
        rng = np.random.default_rng(0)
        return DriftReference(
            feature_samples={
                "mean_intensity": list(rng.normal(0.5, 0.1, 500)),
                "std_intensity": list(rng.normal(0.2, 0.05, 500)),
                "p01_intensity": list(rng.normal(0.0, 0.02, 500)),
                "p50_intensity": list(rng.normal(0.5, 0.1, 500)),
                "p99_intensity": list(rng.normal(1.0, 0.1, 500)),
                "nonzero_fraction": list(rng.uniform(0.4, 0.6, 500)),
            },
            class_counts={"AD": 250, "CN": 250},
            n_samples=500,
            built_at="2026-01-01T00:00:00+00:00",
            source_manifest="manifest.csv",
        )

    def test_no_drift_when_current_matches_reference(self) -> None:
        rng = np.random.default_rng(1)
        reference = self._reference()
        current = {
            "mean_intensity": list(rng.normal(0.5, 0.1, 300)),
            "std_intensity": list(rng.normal(0.2, 0.05, 300)),
            "p01_intensity": list(rng.normal(0.0, 0.02, 300)),
            "p50_intensity": list(rng.normal(0.5, 0.1, 300)),
            "p99_intensity": list(rng.normal(1.0, 0.1, 300)),
            "nonzero_fraction": list(rng.uniform(0.4, 0.6, 300)),
        }
        report = generate_data_drift_report(reference, current, {"AD": 150, "CN": 150})
        assert report.any_drifted is False
        assert report.class_distribution_drifted is False

    def test_drift_detected_when_current_is_shifted(self) -> None:
        rng = np.random.default_rng(1)
        reference = self._reference()
        current = {
            "mean_intensity": list(rng.normal(5.0, 0.1, 300)),  # large real shift
            "std_intensity": list(rng.normal(0.2, 0.05, 300)),
            "p01_intensity": list(rng.normal(0.0, 0.02, 300)),
            "p50_intensity": list(rng.normal(0.5, 0.1, 300)),
            "p99_intensity": list(rng.normal(1.0, 0.1, 300)),
            "nonzero_fraction": list(rng.uniform(0.4, 0.6, 300)),
        }
        report = generate_data_drift_report(reference, current)
        assert report.any_drifted is True
        mean_result = next(r for r in report.feature_results if r.feature == "mean_intensity")
        assert mean_result.drifted is True
        assert mean_result.severity == "alert"

    def test_class_distribution_drift_is_detected(self) -> None:
        reference = self._reference()  # 50/50 AD/CN
        report = generate_data_drift_report(
            reference,
            reference.feature_samples,  # feature distributions unchanged
            current_class_counts={"AD": 490, "CN": 10},  # heavily skewed
        )
        assert report.class_distribution_drifted is True
        assert report.any_drifted is True

    def test_report_renders_to_markdown_and_json(self, tmp_path: Path) -> None:
        reference = self._reference()
        report = generate_data_drift_report(reference, reference.feature_samples)
        paths = report.save(tmp_path)
        assert paths["json"].exists()
        assert paths["markdown"].exists()
        assert "Data Drift Report" in paths["markdown"].read_text(encoding="utf-8")

        import json

        payload = json.loads(paths["json"].read_text(encoding="utf-8"))
        assert "feature_results" in payload
        assert len(payload["feature_results"]) == 6
