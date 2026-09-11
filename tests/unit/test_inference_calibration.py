"""Tests for inference.calibration — checked against hand-verifiable values,
plus a real (no-mock) temperature fit."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

from ad_mri.core.exceptions import ConfigurationError  # noqa: E402
from ad_mri.inference.calibration import (  # noqa: E402
    CalibrationConfig,
    apply_calibration,
    fit_temperature,
    load_calibration_config,
    save_calibration_config,
)


def test_none_method_is_plain_softmax() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    calibrated = apply_calibration(logits, CalibrationConfig(method="none"))
    expected = torch.softmax(logits, dim=-1)
    assert torch.allclose(calibrated, expected)


def test_temperature_one_is_identical_to_none() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    none_result = apply_calibration(logits, CalibrationConfig(method="none"))
    temp_one_result = apply_calibration(
        logits, CalibrationConfig(method="temperature", temperature=1.0)
    )
    assert torch.allclose(none_result, temp_one_result)


def test_high_temperature_flattens_the_distribution() -> None:
    logits = torch.tensor([[1.0, 2.0, 10.0]])
    sharp = apply_calibration(logits, CalibrationConfig(method="temperature", temperature=1.0))
    flat = apply_calibration(logits, CalibrationConfig(method="temperature", temperature=100.0))
    # a very high temperature pushes every probability toward uniform (1/3)
    assert flat.max().item() < sharp.max().item()
    assert flat.max().item() == pytest.approx(1 / 3, abs=0.05)


def test_temperature_scaling_never_changes_the_argmax() -> None:
    logits = torch.tensor([[0.1, 5.0, -3.0, 2.0]])
    for temperature in (0.1, 0.5, 1.0, 2.0, 10.0):
        calibrated = apply_calibration(
            logits, CalibrationConfig(method="temperature", temperature=temperature)
        )
        assert int(torch.argmax(calibrated)) == int(torch.argmax(logits))


def test_probabilities_always_sum_to_one() -> None:
    logits = torch.randn(5, 4)
    for config in (
        CalibrationConfig(method="none"),
        CalibrationConfig(method="temperature", temperature=2.5),
    ):
        calibrated = apply_calibration(logits, config)
        sums = calibrated.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(5), atol=1e-5)


def test_apply_calibration_rejects_unknown_method() -> None:
    logits = torch.tensor([[1.0, 2.0]])
    bad_config = CalibrationConfig().model_copy(update={"method": "not-a-real-method"})
    with pytest.raises(ConfigurationError, match="Unknown calibration method"):
        apply_calibration(logits, bad_config)


def test_fit_temperature_recovers_a_known_overconfident_scale() -> None:
    """Build logits that are a known multiple of the "true" calibrated
    logits (i.e. deliberately overconfident by a known factor), and check
    that fitting recovers approximately that factor as the temperature."""
    torch.manual_seed(0)
    n_samples, n_classes = 500, 3
    true_logits = torch.randn(n_samples, n_classes)
    labels = torch.distributions.Categorical(logits=true_logits).sample()

    known_overconfidence_factor = 3.0
    overconfident_logits = true_logits * known_overconfidence_factor

    fitted_temperature = fit_temperature(overconfident_logits, labels)
    # dividing by ~known_overconfidence_factor undoes the overconfidence
    assert fitted_temperature == pytest.approx(known_overconfidence_factor, rel=0.35)


def test_fit_temperature_is_bounded() -> None:
    torch.manual_seed(1)
    logits = torch.randn(50, 2)
    labels = torch.randint(0, 2, (50,))
    temperature = fit_temperature(logits, labels, bounds=(0.5, 2.0))
    assert 0.5 <= temperature <= 2.0


def test_calibration_config_round_trips_through_yaml(tmp_path: Path) -> None:
    original = CalibrationConfig(method="temperature", temperature=1.73)
    path = save_calibration_config(original, tmp_path / "calibration.yaml")
    reloaded = load_calibration_config(path)
    assert reloaded == original
    assert reloaded.compute_hash() == original.compute_hash()


def test_calibration_config_hash_changes_with_temperature() -> None:
    a = CalibrationConfig(method="temperature", temperature=1.0)
    b = CalibrationConfig(method="temperature", temperature=2.0)
    assert a.compute_hash() != b.compute_hash()


def test_load_calibration_config_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="not found"):
        load_calibration_config(tmp_path / "nonexistent.yaml")


def test_temperature_must_be_positive() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CalibrationConfig(temperature=0.0)
    with pytest.raises(ValidationError):
        CalibrationConfig(temperature=-1.0)
