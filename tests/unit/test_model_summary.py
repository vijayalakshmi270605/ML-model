"""Tests for modeling.summary: parameter counting and MACs/FLOPs estimation.

Uses a deliberately tiny, hand-computable config so the estimator's output can
be checked against an exact manual calculation, not just "some positive number".
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")

from ad_mri.modeling.factory import build_model  # noqa: E402
from ad_mri.modeling.model_config import ModelConfig  # noqa: E402
from ad_mri.modeling.summary import count_parameters, format_summary, summarize_model  # noqa: E402


def _hand_computable_config() -> ModelConfig:
    # 1 conv block: Conv3d(1 -> 2, kernel=3, padding=1, bias=True since
    # normalization="none") on an 8x8x8 input, then MaxPool3d(2).
    # Conv MACs   = (8*8*8) * (1/1) * 2 * (3*3*3) = 512 * 2 * 27 = 27,648
    # Conv params = (2*1*27) + 2 bias              = 56
    # FC1 (2 -> 4): MACs = 2*4 = 8,   params = 2*4 + 4 = 12
    # FC2 (4 -> 2): MACs = 4*2 = 8,   params = 4*2 + 2 = 10
    # Total MACs   = 27,648 + 8 + 8 = 27,664
    # Total params = 56 + 12 + 10   = 78
    return ModelConfig(
        num_classes=2,
        spatial_dims=3,
        input_shape=(8, 8, 8),
        num_blocks=1,
        base_channels=2,
        conv_kernel_size=3,
        pool_kernel_size=2,
        normalization="none",
        dropout_conv=0.0,
        dropout_fc=0.0,
        fc_hidden_dim=4,
    )


def test_count_parameters_matches_hand_calculation() -> None:
    model = build_model(_hand_computable_config())
    total, trainable = count_parameters(model)
    assert total == 78
    assert trainable == 78  # nothing frozen


def test_summarize_model_matches_hand_calculated_macs_and_shapes() -> None:
    config = _hand_computable_config()
    model = build_model(config)

    summary = summarize_model(model, input_shape=(config.in_channels, *config.input_shape))

    assert summary.total_params == 78
    assert summary.estimated_macs == 27_664
    assert summary.estimated_flops == 27_664 * 2
    assert summary.input_shape == (1, 1, 8, 8, 8)
    assert summary.output_shape == (1, 2)


def test_summarize_model_records_every_leaf_layer_with_a_shape() -> None:
    config = _hand_computable_config()
    model = build_model(config)
    summary = summarize_model(model, input_shape=(config.in_channels, *config.input_shape))

    layer_types = [layer.layer_type for layer in summary.layers]
    assert "Conv3d" in layer_types
    assert "MaxPool3d" in layer_types
    assert "AdaptiveAvgPool3d" in layer_types
    assert "Linear" in layer_types
    assert all(layer.output_shape for layer in summary.layers)


def test_format_summary_includes_key_totals() -> None:
    config = _hand_computable_config()
    model = build_model(config)
    summary = summarize_model(model, input_shape=(config.in_channels, *config.input_shape))

    text = format_summary(summary)
    assert "Total params:" in text
    assert "78" in text
    assert "27,664" in text  # MACs
    assert "Estimated FLOPs" in text


def test_larger_default_config_has_more_parameters_than_the_tiny_one() -> None:
    tiny = build_model(_hand_computable_config())
    default = build_model(ModelConfig(num_classes=3))
    tiny_total, _ = count_parameters(tiny)
    default_total, _ = count_parameters(default)
    assert default_total > tiny_total
