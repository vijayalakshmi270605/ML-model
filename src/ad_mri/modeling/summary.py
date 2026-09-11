"""Model summary utility: layer-by-layer shapes, parameter counts, and an
analytical MACs/FLOPs estimate.

No extra dependency (fvcore/ptflops/thop) is added for this — Conv/Linear
layers dominate a CNN's compute cost, and their MAC count is a closed-form
function of kernel size and output shape, computed here via forward hooks
during one dummy forward pass. This is an *estimate* (it doesn't count
normalization/activation/pooling ops), stated as such wherever it's reported.
See docs/TECHNICAL_SPECIFICATION.md §5.

`torch` is a training-phase dependency (requirements/train.txt), imported
lazily so this module stays importable without it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: 1 multiply-accumulate == 2 floating-point operations, by convention.
_FLOPS_PER_MAC = 2


@dataclass(frozen=True)
class LayerSummary:
    name: str
    layer_type: str
    output_shape: tuple[int, ...]
    param_count: int
    macs: int


@dataclass(frozen=True)
class ModelSummary:
    layers: list[LayerSummary] = field(default_factory=list)
    total_params: int = 0
    trainable_params: int = 0
    non_trainable_params: int = 0
    input_shape: tuple[int, ...] = ()
    output_shape: tuple[int, ...] = ()
    estimated_macs: int = 0
    estimated_flops: int = 0


def count_parameters(model: Any) -> tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _layer_macs(module: Any, output_shape: tuple[int, ...]) -> int:
    import torch.nn as nn

    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        output_spatial_elements = 1
        for d in output_shape[2:]:
            output_spatial_elements *= d
        kernel_volume = 1
        for k in module.kernel_size:
            kernel_volume *= k
        return (
            output_spatial_elements
            * (module.in_channels // module.groups)
            * module.out_channels
            * kernel_volume
        )
    if isinstance(module, nn.Linear):
        return module.in_features * module.out_features
    return 0


def summarize_model(model: Any, input_shape: tuple[int, ...], batch_size: int = 1) -> ModelSummary:
    """Run one dummy forward pass and record every leaf submodule's output
    shape, parameter count, and estimated MACs.

    `input_shape` excludes the batch dimension, e.g. `(1, 128, 128, 128)` for
    a single-channel 3D volume — matches `ModelConfig.in_channels` +
    `ModelConfig.input_shape`.
    """
    import torch
    import torch.nn as nn

    layers: list[LayerSummary] = []

    def _make_hook(name: str) -> Any:
        def hook(module: nn.Module, _inputs: Any, output: Any) -> None:
            out = output[0] if isinstance(output, (tuple, list)) else output
            shape = tuple(out.shape) if hasattr(out, "shape") else ()
            params = sum(p.numel() for p in module.parameters(recurse=False))
            macs = _layer_macs(module, shape)
            layers.append(LayerSummary(name, type(module).__name__, shape, params, macs))

        return hook

    handles = [
        module.register_forward_hook(_make_hook(name))
        for name, module in model.named_modules()
        if name
        and not list(module.children())  # leaf modules only — avoid double-counting containers
    ]

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.zeros((batch_size, *input_shape), device=device)
    with torch.no_grad():
        output = model(dummy_input)
    model.train(was_training)

    for handle in handles:
        handle.remove()

    total_params, trainable_params = count_parameters(model)
    total_macs = sum(layer.macs for layer in layers)

    return ModelSummary(
        layers=layers,
        total_params=total_params,
        trainable_params=trainable_params,
        non_trainable_params=total_params - trainable_params,
        input_shape=(batch_size, *input_shape),
        output_shape=tuple(output.shape),
        estimated_macs=total_macs,
        estimated_flops=total_macs * _FLOPS_PER_MAC,
    )


def format_summary(summary: ModelSummary) -> str:
    """Render a `ModelSummary` as a human-readable, torchsummary-style table.

    Columns are separated by an explicit two-space gap (not just field-width
    padding) so a value that exactly fills its field width can never visually
    merge into the next column.
    """
    name_width, shape_width, param_width, mac_width = 42, 24, 14, 16
    gap = "  "
    header = (
        f"{'Layer (type)':<{name_width}}{gap}{'Output Shape':<{shape_width}}{gap}"
        f"{'Param #':>{param_width}}{gap}{'MACs':>{mac_width}}"
    )
    separator = "=" * (name_width + shape_width + param_width + mac_width + 3 * len(gap))

    lines = [
        f"Input shape:  {summary.input_shape}",
        f"Output shape: {summary.output_shape}",
        "",
        header,
        separator,
    ]
    for layer in summary.layers:
        label = f"{layer.name} ({layer.layer_type})"
        lines.append(
            f"{label:<{name_width}}{gap}{str(layer.output_shape):<{shape_width}}{gap}"
            f"{layer.param_count:>{param_width},}{gap}{layer.macs:>{mac_width},}"
        )
    lines.append(separator)
    lines.append(f"Total params:          {summary.total_params:,}")
    lines.append(f"Trainable params:      {summary.trainable_params:,}")
    lines.append(f"Non-trainable params:  {summary.non_trainable_params:,}")
    lines.append(separator)
    lines.append(f"Estimated MACs:        {summary.estimated_macs:,}")
    lines.append(
        f"Estimated FLOPs:       {summary.estimated_flops:,}  (2 x MACs, Conv/Linear only)"
    )
    return "\n".join(lines)
