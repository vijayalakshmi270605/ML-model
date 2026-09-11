"""Tests for explainability.config — LimeConfig/ShapConfig validation,
content-hashing, and load/save round-tripping."""

from __future__ import annotations

from pathlib import Path

import pytest

from ad_mri.core.exceptions import ConfigurationError
from ad_mri.explainability.config import (
    LimeConfig,
    ShapConfig,
    load_lime_config,
    load_shap_config,
    save_lime_config,
    save_shap_config,
)


def test_lime_config_defaults_are_valid() -> None:
    config = LimeConfig()
    assert config.segmentation_algorithm == "quickshift"
    assert config.num_samples >= 10


def test_shap_config_defaults_are_valid() -> None:
    config = ShapConfig()
    assert config.explainer == "gradient"
    assert config.n_background_samples >= 1


def test_lime_config_hash_changes_with_any_parameter() -> None:
    base = LimeConfig()
    changed = LimeConfig(num_samples=base.num_samples + 1)
    assert base.compute_hash() != changed.compute_hash()


def test_lime_config_hash_is_deterministic() -> None:
    a = LimeConfig(seed=7, num_samples=123)
    b = LimeConfig(seed=7, num_samples=123)
    assert a.compute_hash() == b.compute_hash()


def test_shap_config_hash_changes_with_any_parameter() -> None:
    base = ShapConfig()
    changed = ShapConfig(nsamples=base.nsamples + 1)
    assert base.compute_hash() != changed.compute_hash()


def test_lime_config_round_trips_through_yaml(tmp_path: Path) -> None:
    original = LimeConfig(num_samples=77, seed=3, segmentation_algorithm="slic")
    path = save_lime_config(original, tmp_path / "lime.yaml")
    reloaded = load_lime_config(path)
    assert reloaded == original
    assert reloaded.compute_hash() == original.compute_hash()


def test_shap_config_round_trips_through_yaml(tmp_path: Path) -> None:
    original = ShapConfig(nsamples=99, n_background_samples=12)
    path = save_shap_config(original, tmp_path / "shap.yaml")
    reloaded = load_shap_config(path)
    assert reloaded == original


def test_save_lime_config_writes_a_content_hash_field(tmp_path: Path) -> None:
    import yaml

    config = LimeConfig()
    path = save_lime_config(config, tmp_path / "lime.yaml")
    raw = yaml.safe_load(path.read_text())
    assert raw["config_hash"] == config.compute_hash()


def test_load_lime_config_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="not found"):
        load_lime_config(tmp_path / "nonexistent.yaml")


def test_load_shap_config_invalid_value_raises(tmp_path: Path) -> None:
    path = tmp_path / "shap.yaml"
    path.write_text("explainer: not-a-real-explainer\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Invalid SHAP config"):
        load_shap_config(path)


def test_slice_index_accepts_middle_or_integer_string() -> None:
    LimeConfig(slice_index="middle")
    LimeConfig(slice_index="5")
    LimeConfig(slice_index="-1")


def test_slice_axis_out_of_range_rejected() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        LimeConfig(slice_axis=3)
