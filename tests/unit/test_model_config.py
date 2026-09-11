from __future__ import annotations

from pathlib import Path

import pydantic
import pytest

from ad_mri.core.exceptions import ConfigurationError
from ad_mri.modeling.model_config import ModelConfig, load_model_config, save_model_config


def _config(**overrides: object) -> ModelConfig:
    defaults: dict[str, object] = {"num_classes": 3}
    defaults.update(overrides)
    return ModelConfig(**defaults)  # type: ignore[arg-type]


def test_num_classes_is_required() -> None:
    with pytest.raises(pydantic.ValidationError):
        ModelConfig()  # type: ignore[call-arg]


def test_defaults_are_valid() -> None:
    config = _config()
    assert config.spatial_dims == 3
    assert config.input_shape == (128, 128, 128)
    assert config.num_blocks == 4


def test_resolve_channel_schedule_double() -> None:
    config = _config(base_channels=32, num_blocks=4, channel_growth="double")
    assert config.resolve_channel_schedule() == [32, 64, 128, 256]


def test_resolve_channel_schedule_linear() -> None:
    config = _config(base_channels=16, num_blocks=3, channel_growth="linear")
    assert config.resolve_channel_schedule() == [16, 32, 48]


def test_resolve_channel_schedule_explicit_override() -> None:
    config = _config(num_blocks=3, channels=(8, 16, 16))
    assert config.resolve_channel_schedule() == [8, 16, 16]


def test_custom_growth_requires_explicit_channels() -> None:
    with pytest.raises(ConfigurationError):
        _config(channel_growth="custom", channels=None)


def test_rejects_channels_length_mismatch() -> None:
    with pytest.raises(ConfigurationError):
        _config(num_blocks=4, channels=(8, 16))


def test_rejects_input_shape_dims_mismatch() -> None:
    with pytest.raises(ConfigurationError):
        _config(spatial_dims=3, input_shape=(128, 128))  # only 2 dims for spatial_dims=3


def test_rejects_input_too_small_for_pooling_depth() -> None:
    with pytest.raises(ConfigurationError):
        _config(spatial_dims=3, input_shape=(8, 8, 8), num_blocks=4, pool_kernel_size=2)


def test_2d_config_is_valid() -> None:
    config = _config(spatial_dims=2, input_shape=(64, 64), num_blocks=3)
    assert len(config.resolve_channel_schedule()) == 3


def test_compute_hash_is_deterministic_and_content_sensitive() -> None:
    a = _config()
    b = _config()
    c = _config(base_channels=64)
    assert a.compute_hash() == b.compute_hash()
    assert a.compute_hash() != c.compute_hash()


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    config = _config(num_classes=4, base_channels=16)
    path = tmp_path / "model.yaml"

    save_model_config(config, path)
    reloaded = load_model_config(path)

    assert reloaded.num_classes == 4
    assert reloaded.base_channels == 16
    assert reloaded.compute_hash() == config.compute_hash()


def test_load_model_config_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        load_model_config(tmp_path / "does-not-exist.yaml")


def test_the_shipped_default_model_config_is_valid() -> None:
    """configs/model/cnn3d_default.yaml is documentation-by-example — it must stay valid."""
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "configs" / "model" / "cnn3d_default.yaml"
    config = load_model_config(config_path)
    assert config.num_classes == 3
    assert config.architecture == "convnet"
