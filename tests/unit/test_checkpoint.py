"""Tests for modeling.checkpoint: a checkpoint is always self-describing."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")

from ad_mri.core.exceptions import ConfigurationError  # noqa: E402
from ad_mri.modeling.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from ad_mri.modeling.factory import build_model  # noqa: E402
from ad_mri.modeling.model_config import ModelConfig  # noqa: E402


def _tiny_config(**overrides: object) -> ModelConfig:
    defaults: dict[str, object] = {
        "num_classes": 3,
        "spatial_dims": 3,
        "input_shape": (16, 16, 16),
        "num_blocks": 2,
        "base_channels": 4,
        "fc_hidden_dim": 8,
    }
    defaults.update(overrides)
    return ModelConfig(**defaults)  # type: ignore[arg-type]


def test_save_and_load_round_trip_preserves_config_and_weights(tmp_path: Path) -> None:
    config = _tiny_config()
    model = build_model(config)
    checkpoint_path = tmp_path / "model.pt"

    save_checkpoint(model, config, checkpoint_path)
    reloaded_model, reloaded_config, metadata = load_checkpoint(checkpoint_path)

    assert reloaded_config.compute_hash() == config.compute_hash()
    assert metadata.model_config_hash == config.compute_hash()

    x = torch.zeros((1, config.in_channels, *config.input_shape))
    model.eval()
    reloaded_model.eval()
    with torch.no_grad():
        original_output = model(x)
        reloaded_output = reloaded_model(x)
    assert torch.equal(original_output, reloaded_output)


def test_save_checkpoint_persists_extra_metadata(tmp_path: Path) -> None:
    config = _tiny_config()
    model = build_model(config)
    checkpoint_path = tmp_path / "model.pt"

    save_checkpoint(model, config, checkpoint_path, extra={"epoch": 7, "val_accuracy": 0.83})
    _, _, metadata = load_checkpoint(checkpoint_path)

    assert metadata.extra == {"epoch": 7, "val_accuracy": 0.83}


def test_load_checkpoint_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        load_checkpoint(tmp_path / "does-not-exist.pt")


def test_load_checkpoint_rejects_a_non_checkpoint_file(tmp_path: Path) -> None:
    path = tmp_path / "not-a-checkpoint.pt"
    torch.save({"just": "a plain dict"}, path)
    with pytest.raises(ConfigurationError):
        load_checkpoint(path)


def test_load_checkpoint_detects_a_tampered_config_hash(tmp_path: Path) -> None:
    config = _tiny_config()
    model = build_model(config)
    path = tmp_path / "model.pt"
    save_checkpoint(model, config, path)

    payload = torch.load(path, weights_only=False)
    payload["model_config_hash"] = "0" * 64  # corrupt it
    torch.save(payload, path)

    with pytest.raises(ConfigurationError):
        load_checkpoint(path)
