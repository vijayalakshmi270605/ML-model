"""CNN architecture definitions.

`ConvNetClassifier` is a configurable, progressively-deeper 2D/3D CNN: a stack
of `ConvBlock`s (Conv -> Norm -> Activation -> [Dropout] -> [Pool]) followed by
global average pooling and a fully-connected classification head. Every
structural parameter comes from `ModelConfig` (`model_config.py`) — this
module contains no dataset-specific literal (no hardcoded class count, no
hardcoded input resolution).

`spatial_dims` (2 or 3) is handled via MONAI's dimension-agnostic layer
factories (`Conv`/`Norm`/`Pool`), so there is one code path for both 2D
slice-based and 3D volumetric models, not a duplicated 2D/3D branch.

`torch`/`monai` are training-phase dependencies (requirements/train.txt),
imported lazily at call time so this module stays importable without them.
See docs/TECHNICAL_SPECIFICATION.md §5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ad_mri.core.exceptions import ConfigurationError
from ad_mri.core.logging import get_logger
from ad_mri.modeling.model_config import ModelConfig

logger = get_logger(__name__)

if TYPE_CHECKING:
    # Always resolved for the type checker (mypy analyzes only this branch);
    # see the identical pattern/rationale in datasets.dataset.MRIDataset.
    import torch.nn as nn
else:
    try:
        import torch.nn as nn
    except ImportError:  # torch not installed outside the training/worker images
        nn = None  # type: ignore[assignment]

#: architecture name -> model class, populated by `register_architecture`.
_ARCHITECTURE_REGISTRY: dict[str, type[Any]] = {}


def register_architecture(name: str) -> Any:
    """Class decorator: registers a `nn.Module` subclass under `name` for
    `modeling.factory.build_model` to resolve — keeps architecture choice a
    configuration concern (`ModelConfig.architecture`), not a code fork."""

    def decorator(cls: type[Any]) -> type[Any]:
        _ARCHITECTURE_REGISTRY[name] = cls
        return cls

    return decorator


def _build_activation(mode: str) -> Any:
    if mode == "relu":
        return nn.ReLU(inplace=True)
    if mode == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01, inplace=True)
    if mode == "prelu":
        return nn.PReLU()
    if mode == "gelu":
        return nn.GELU()
    raise ConfigurationError(f"Unknown activation: {mode!r}")


def _build_norm(mode: str, spatial_dims: int, num_channels: int) -> Any:
    from monai.networks.layers.factories import Norm

    if mode == "none":
        return None
    if mode == "batch":
        return Norm[Norm.BATCH, spatial_dims](num_channels)
    if mode == "instance":
        return Norm[Norm.INSTANCE, spatial_dims](num_channels)
    raise ConfigurationError(f"Unknown normalization: {mode!r}")


class ConvBlock(nn.Module if nn is not None else object):  # type: ignore[misc]
    """One feature-extraction block: Conv -> Norm -> Activation -> [Dropout] -> [Pool].

    `spatial_dims` selects Conv2d/Conv3d, BatchNorm2d/3d, MaxPool2d/3d via
    MONAI's layer factories, so the exact same block implementation serves
    both 2D and 3D models.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        normalization: str,
        activation: str,
        dropout: float,
        pool_kernel_size: int | None,
    ) -> None:
        super().__init__()
        from monai.networks.layers.factories import Conv, Pool

        conv_cls = Conv[Conv.CONV, spatial_dims]
        norm = _build_norm(normalization, spatial_dims, out_channels)
        self.conv = conv_cls(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=norm is None,
        )
        self.norm = norm
        self.activation = _build_activation(activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        pool_cls = Pool[Pool.MAX, spatial_dims]
        self.pool = pool_cls(kernel_size=pool_kernel_size) if pool_kernel_size else None

    def forward(self, x: Any) -> Any:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        if self.pool is not None:
            x = self.pool(x)
        return x


@register_architecture("convnet")
class ConvNetClassifier(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Configurable N-block CNN classifier for 2D/3D medical images.

    Architecture: `[ConvBlock] * num_blocks` (progressively deeper feature
    extraction — channel width grows per `ModelConfig.channel_growth` while
    each block's pooling halves spatial resolution) -> global average pool
    (makes the classifier robust to the exact input spatial size, rather than
    baking a flatten size into the architecture) -> flatten -> FC head
    (`Linear -> Activation -> Dropout -> Linear(num_classes)`).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        from monai.networks.layers.factories import Pool

        self.config = config

        channel_schedule = config.resolve_channel_schedule()
        blocks: list[Any] = []
        in_channels = config.in_channels
        for out_channels in channel_schedule:
            blocks.append(
                ConvBlock(
                    spatial_dims=config.spatial_dims,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=config.conv_kernel_size,
                    normalization=config.normalization,
                    activation=config.activation,
                    dropout=config.dropout_conv,
                    pool_kernel_size=config.pool_kernel_size,
                )
            )
            in_channels = out_channels
        self.feature_blocks = nn.Sequential(*blocks)

        adaptive_avg_pool_cls = Pool[Pool.ADAPTIVEAVG, config.spatial_dims]
        self.global_pool = adaptive_avg_pool_cls(1)
        self.flatten = nn.Flatten()

        final_channels = channel_schedule[-1]
        self.classifier = nn.Sequential(
            nn.Linear(final_channels, config.fc_hidden_dim),
            _build_activation(config.activation),
            nn.Dropout(config.dropout_fc),
            nn.Linear(config.fc_hidden_dim, config.num_classes),
        )

        self._initialize_weights()
        logger.info(
            "model_built",
            architecture=config.architecture,
            spatial_dims=config.spatial_dims,
            num_blocks=config.num_blocks,
            channels=channel_schedule,
            num_classes=config.num_classes,
            num_parameters=sum(p.numel() for p in self.parameters()),
        )

    def forward(self, x: Any) -> Any:
        """`x`: `(batch, in_channels, *spatial_dims)` -> logits `(batch, num_classes)`.

        Returns raw logits (no softmax) — apply `torch.softmax`/`nn.CrossEntropyLoss`
        as appropriate at the training/inference call site.
        """
        x = self.feature_blocks(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        return self.classifier(x)

    def _initialize_weights(self) -> None:
        """Kaiming (He) or Xavier (Glorot) init for conv/linear weights, zero
        biases, and unit-scale/zero-bias for normalization layers — standard
        practice for ReLU-family networks, applied consistently rather than
        left to PyTorch's default per-layer initialization.
        """
        conv_types = tuple(
            t for t in (getattr(nn, f"Conv{d}d", None) for d in (1, 2, 3)) if t is not None
        )
        norm_types = tuple(
            t
            for name in ("BatchNorm", "InstanceNorm")
            for d in (1, 2, 3)
            if (t := getattr(nn, f"{name}{d}d", None)) is not None
        )

        for module in self.modules():
            if isinstance(module, conv_types):
                if self.config.weight_init == "kaiming":
                    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                else:
                    nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, norm_types):
                if getattr(module, "weight", None) is not None:
                    nn.init.ones_(module.weight)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                if self.config.weight_init == "kaiming":
                    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                else:
                    nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)
