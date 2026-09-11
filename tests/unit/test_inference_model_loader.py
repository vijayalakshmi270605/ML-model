"""Tests for inference.model_loader.ModelLoader — real checkpoint loading and
in-memory caching (no mocks)."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from ad_mri.core.exceptions import ConfigurationError  # noqa: E402
from ad_mri.inference.model_loader import ModelLoader  # noqa: E402


def _build_checkpoint(
    tmp_path: Path,
    name: str = "model.pt",
    *,
    with_class_names: bool = True,
    preprocessing_hash: str | None = "__use_real__",
) -> tuple[Path, Path]:
    from ad_mri.datasets.preprocessing_config import PreprocessingConfig, save_preprocessing_config
    from ad_mri.modeling.checkpoint import save_checkpoint
    from ad_mri.modeling.factory import build_model
    from ad_mri.modeling.model_config import ModelConfig

    model_config = ModelConfig(
        num_classes=2,
        spatial_dims=3,
        input_shape=(16, 16, 16),
        num_blocks=1,
        base_channels=4,
        fc_hidden_dim=8,
    )
    preprocessing_config = PreprocessingConfig(
        target_shape=(16, 16, 16), target_spacing=(1.0, 1.0, 1.0), min_voxel_count=1000
    )
    preprocessing_path = save_preprocessing_config(
        preprocessing_config, tmp_path / f"{name}.preprocessing.yaml"
    )

    extra: dict[str, object] = {}
    if with_class_names:
        extra["class_names"] = ["AD", "CN"]
    if preprocessing_hash == "__use_real__":
        extra["preprocessing_config_hash"] = preprocessing_config.compute_hash()
    elif preprocessing_hash is not None:
        extra["preprocessing_config_hash"] = preprocessing_hash

    model = build_model(model_config)
    checkpoint_path = tmp_path / name
    save_checkpoint(model, model_config, checkpoint_path, extra=extra)
    return checkpoint_path, preprocessing_path


def test_load_returns_a_ready_to_serve_model(tmp_path: Path) -> None:
    checkpoint_path, preprocessing_path = _build_checkpoint(tmp_path)
    loader = ModelLoader()
    cached = loader.load(checkpoint_path, preprocessing_path)

    assert cached.class_names == ["AD", "CN"]
    assert not cached.model.training  # .eval() was called
    assert cached.device == torch.device("cpu")
    assert cached.checkpoint_path == checkpoint_path


def test_repeated_load_returns_the_identical_cached_object(tmp_path: Path) -> None:
    checkpoint_path, preprocessing_path = _build_checkpoint(tmp_path)
    loader = ModelLoader()
    first = loader.load(checkpoint_path, preprocessing_path)
    second = loader.load(checkpoint_path, preprocessing_path)
    assert first is second  # not just equal — the exact same cached object


def test_is_cached_reflects_cache_state(tmp_path: Path) -> None:
    checkpoint_path, preprocessing_path = _build_checkpoint(tmp_path)
    loader = ModelLoader()
    assert not loader.is_cached(checkpoint_path, preprocessing_path)
    loader.load(checkpoint_path, preprocessing_path)
    assert loader.is_cached(checkpoint_path, preprocessing_path)


def test_clear_evicts_the_cache(tmp_path: Path) -> None:
    checkpoint_path, preprocessing_path = _build_checkpoint(tmp_path)
    loader = ModelLoader()
    first = loader.load(checkpoint_path, preprocessing_path)
    loader.clear()
    assert not loader.is_cached(checkpoint_path, preprocessing_path)
    second = loader.load(checkpoint_path, preprocessing_path)
    assert first is not second  # reloaded from disk, a genuinely new object


def test_two_different_checkpoints_get_separate_cache_entries(tmp_path: Path) -> None:
    checkpoint_a, preprocessing_a = _build_checkpoint(tmp_path, name="a.pt")
    checkpoint_b, preprocessing_b = _build_checkpoint(tmp_path, name="b.pt")
    loader = ModelLoader()
    cached_a = loader.load(checkpoint_a, preprocessing_a)
    cached_b = loader.load(checkpoint_b, preprocessing_b)
    assert cached_a is not cached_b
    assert loader.is_cached(checkpoint_a, preprocessing_a)
    assert loader.is_cached(checkpoint_b, preprocessing_b)


def test_missing_class_names_raises_configuration_error(tmp_path: Path) -> None:
    checkpoint_path, preprocessing_path = _build_checkpoint(tmp_path, with_class_names=False)
    loader = ModelLoader()
    with pytest.raises(ConfigurationError, match="class_names"):
        loader.load(checkpoint_path, preprocessing_path)


def test_mismatched_preprocessing_hash_raises_configuration_error(tmp_path: Path) -> None:
    checkpoint_path, preprocessing_path = _build_checkpoint(tmp_path, preprocessing_hash="0" * 64)
    loader = ModelLoader()
    with pytest.raises(ConfigurationError, match="does not match"):
        loader.load(checkpoint_path, preprocessing_path)


def test_unreadable_checkpoint_raises_configuration_error(tmp_path: Path) -> None:
    checkpoint_path, preprocessing_path = _build_checkpoint(tmp_path)
    loader = ModelLoader()
    with pytest.raises(ConfigurationError):
        loader.load(tmp_path / "does-not-exist.pt", preprocessing_path)


def test_invalid_device_choice_raises_configuration_error(tmp_path: Path) -> None:
    checkpoint_path, preprocessing_path = _build_checkpoint(tmp_path)
    loader = ModelLoader()
    with pytest.raises(ConfigurationError, match="Unknown device choice"):
        loader.load(checkpoint_path, preprocessing_path, device_choice="tpu")


def test_cuda_requested_but_unavailable_raises_configuration_error(tmp_path: Path) -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA is available in this environment; cannot test the unavailable path.")
    checkpoint_path, preprocessing_path = _build_checkpoint(tmp_path)
    loader = ModelLoader()
    with pytest.raises(ConfigurationError, match="CUDA is not available"):
        loader.load(checkpoint_path, preprocessing_path, device_choice="cuda")
