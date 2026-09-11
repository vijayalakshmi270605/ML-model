"""Tests for modeling.architectures / modeling.factory.

Skipped automatically if torch/monai (training-phase extras) aren't installed.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")

from ad_mri.core.exceptions import ConfigurationError  # noqa: E402
from ad_mri.modeling.architectures import ConvNetClassifier  # noqa: E402
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


def test_build_model_returns_a_convnet_classifier() -> None:
    model = build_model(_tiny_config())
    assert isinstance(model, ConvNetClassifier)


def test_build_model_rejects_unknown_architecture() -> None:
    config = _tiny_config(architecture="does-not-exist")
    with pytest.raises(ConfigurationError):
        build_model(config)


def test_forward_pass_output_shape_matches_num_classes() -> None:
    config = _tiny_config(num_classes=5)
    model = build_model(config)
    x = torch.zeros((2, config.in_channels, *config.input_shape))
    logits = model(x)
    assert tuple(logits.shape) == (2, 5)


def test_forward_pass_is_shape_agnostic_thanks_to_global_average_pooling() -> None:
    """Global average pooling means a *larger* spatial input than configured
    still produces the same-shaped output (the network isn't hardcoded to one
    exact input resolution's flatten size)."""
    config = _tiny_config()
    model = build_model(config)
    model.eval()

    small = torch.zeros((1, config.in_channels, 16, 16, 16))
    larger = torch.zeros((1, config.in_channels, 32, 32, 32))

    with torch.no_grad():
        out_small = model(small)
        out_larger = model(larger)

    assert out_small.shape == out_larger.shape == (1, 3)


def test_2d_architecture_builds_and_runs() -> None:
    config = _tiny_config(spatial_dims=2, input_shape=(16, 16))
    model = build_model(config)
    x = torch.zeros((1, config.in_channels, *config.input_shape))
    logits = model(x)
    assert tuple(logits.shape) == (1, 3)


def test_custom_channel_schedule_is_respected() -> None:
    config = _tiny_config(num_blocks=3, channels=(4, 8, 8))
    model = build_model(config)
    x = torch.zeros((1, config.in_channels, *config.input_shape))
    logits = model(x)
    assert tuple(logits.shape) == (1, 3)


@pytest.mark.parametrize("normalization", ["batch", "instance", "none"])
def test_every_normalization_mode_builds_and_runs(normalization: str) -> None:
    config = _tiny_config(normalization=normalization)
    model = build_model(config)
    x = torch.zeros((1, config.in_channels, *config.input_shape))
    logits = model(x)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("activation", ["relu", "leaky_relu", "prelu", "gelu"])
def test_every_activation_mode_builds_and_runs(activation: str) -> None:
    config = _tiny_config(activation=activation)
    model = build_model(config)
    x = torch.zeros((1, config.in_channels, *config.input_shape))
    logits = model(x)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("weight_init", ["kaiming", "xavier"])
def test_weight_initialization_produces_finite_weights(weight_init: str) -> None:
    config = _tiny_config(weight_init=weight_init)
    model = build_model(config)
    for param in model.parameters():
        assert torch.isfinite(param).all()


def test_model_is_deterministic_in_eval_mode() -> None:
    config = _tiny_config(dropout_conv=0.5, dropout_fc=0.5)
    model = build_model(config)
    model.eval()
    x = torch.zeros((1, config.in_channels, *config.input_shape))
    with torch.no_grad():
        out_a = model(x)
        out_b = model(x)
    assert torch.equal(out_a, out_b)  # dropout must be inactive in eval()
