from __future__ import annotations

from pathlib import Path

import pytest

from ad_mri.core.exceptions import ConfigurationError
from ad_mri.datasets.preprocessing_config import (
    AugmentationConfig,
    PreprocessingConfig,
    load_preprocessing_config,
    save_preprocessing_config,
)


def test_defaults_are_valid() -> None:
    config = PreprocessingConfig()
    assert config.target_shape == (128, 128, 128)
    assert config.normalization == "zscore_nonzero"
    assert config.augmentation.allow_flip is False  # medically-conservative default


def test_compute_hash_is_deterministic_and_content_sensitive() -> None:
    a = PreprocessingConfig()
    b = PreprocessingConfig()
    c = PreprocessingConfig(target_shape=(64, 64, 64))

    assert a.compute_hash() == b.compute_hash()
    assert a.compute_hash() != c.compute_hash()


def test_rejects_non_positive_target_shape() -> None:
    with pytest.raises(ConfigurationError):
        PreprocessingConfig(target_shape=(0, 128, 128))


def test_rejects_invalid_percentile_clip_range() -> None:
    with pytest.raises(ConfigurationError):
        PreprocessingConfig(percentile_clip=(99.5, 0.5))


def test_rejects_invalid_expected_dim_range() -> None:
    with pytest.raises(ConfigurationError):
        PreprocessingConfig(expected_dim_range=(4, 3))


def test_augmentation_config_defaults_are_conservative() -> None:
    aug = AugmentationConfig()
    assert aug.rotation_range_deg <= 25.0  # never a full/near-full rotation
    assert aug.allow_flip is False


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    config = PreprocessingConfig(version="9.9.9", target_shape=(64, 64, 64))
    path = tmp_path / "preprocessing.yaml"

    save_preprocessing_config(config, path)
    reloaded = load_preprocessing_config(path)

    assert reloaded.version == "9.9.9"
    assert reloaded.target_shape == (64, 64, 64)
    assert reloaded.compute_hash() == config.compute_hash()


def test_load_preprocessing_config_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        load_preprocessing_config(tmp_path / "does-not-exist.yaml")


def test_the_shipped_default_preprocessing_config_is_valid() -> None:
    """configs/preprocessing/default.yaml is documentation-by-example — it must stay valid."""
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "configs" / "preprocessing" / "default.yaml"
    config = load_preprocessing_config(config_path)
    assert config.version == "1.0.0"
