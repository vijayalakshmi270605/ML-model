"""CNN architecture configuration schema.

Every structural choice (depth, channels, normalization, activation, dropout,
classifier head size, weight init) is config-driven — never a literal in
`architectures.py`. `num_classes` has **no default**: it must be set
explicitly from the dataset's label taxonomy (see
`data_pipeline.dataset_config.DatasetConfig.label_map`), per this project's
rule against hardcoded dataset-specific assumptions. See
docs/TECHNICAL_SPECIFICATION.md §5, §16.

Reproducibility/provenance contract (mirrors `PreprocessingConfig`): every
`ModelConfig` carries a human-set semantic `version` and exposes
`compute_hash()` — a content-addressed fingerprint of the exact resolved
architecture — saved alongside model checkpoints (see `modeling.checkpoint`)
so a trained model can always be traced back to the architecture that
produced it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ad_mri.core.exceptions import ConfigurationError

NormalizationMode = Literal["batch", "instance", "none"]
ActivationMode = Literal["relu", "leaky_relu", "prelu", "gelu"]
ChannelGrowth = Literal["double", "linear", "custom"]
WeightInit = Literal["kaiming", "xavier"]


class ModelConfig(BaseModel):
    """CNN architecture parameters, resolved once and shared by every call site
    that needs to know the model's shape (factory, summary, checkpointing)."""

    model_config = ConfigDict(extra="ignore")

    version: str = "1.0.0"
    #: Registry key resolved by `modeling.factory.build_model` — see
    #: `modeling.architectures.register_architecture`.
    architecture: str = "convnet"

    # --- Input (informational + used by the summary utility; the network
    # itself is shape-agnostic thanks to global average pooling, but a
    # concrete shape is required to validate depth and to run a dummy
    # forward pass for the summary/FLOPs utilities). ---
    spatial_dims: Literal[2, 3] = 3
    in_channels: int = Field(default=1, ge=1)
    input_shape: tuple[int, ...] = (128, 128, 128)

    # --- Output — deliberately has no default. ---
    num_classes: int = Field(gt=1)

    # --- Feature extractor (progressively deeper: `num_blocks` conv blocks,
    # each halving spatial resolution and growing channel width). ---
    num_blocks: int = Field(default=4, ge=1, le=8)
    base_channels: int = Field(default=32, ge=1)
    channel_growth: ChannelGrowth = "double"
    #: Explicit per-block output channel counts; overrides `channel_growth`
    #: when set. Required (and validated) when channel_growth == "custom".
    channels: tuple[int, ...] | None = None
    conv_kernel_size: int = Field(default=3, ge=1)
    pool_kernel_size: int = Field(default=2, ge=1)
    normalization: NormalizationMode = "batch"
    activation: ActivationMode = "relu"
    dropout_conv: float = Field(default=0.0, ge=0.0, le=0.9)

    # --- Classification head ---
    fc_hidden_dim: int = Field(default=256, ge=1)
    dropout_fc: float = Field(default=0.5, ge=0.0, le=0.9)

    # --- Weight initialization ---
    weight_init: WeightInit = "kaiming"

    @field_validator("input_shape", "channels", mode="before")
    @classmethod
    def _coerce_tuple(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate(self) -> ModelConfig:
        if len(self.input_shape) != self.spatial_dims:
            raise ConfigurationError(
                f"input_shape {self.input_shape} must have {self.spatial_dims} "
                f"dimension(s) to match spatial_dims={self.spatial_dims}."
            )
        if self.channel_growth == "custom" and self.channels is None:
            raise ConfigurationError(
                "channels must be set explicitly when channel_growth='custom'."
            )
        if self.channels is not None and len(self.channels) != self.num_blocks:
            raise ConfigurationError(
                f"channels has {len(self.channels)} entries but num_blocks={self.num_blocks}."
            )
        # Every block halves spatial resolution (pool_kernel_size stride) —
        # reject configs where the smallest input axis can't survive
        # num_blocks poolings without collapsing to zero.
        min_input_dim = min(self.input_shape)
        min_surviving_dim = self.pool_kernel_size**self.num_blocks
        if min_input_dim < min_surviving_dim:
            raise ConfigurationError(
                f"input_shape {self.input_shape} is too small for {self.num_blocks} "
                f"pooling stage(s) at pool_kernel_size={self.pool_kernel_size} "
                f"(needs every axis >= {min_surviving_dim})."
            )
        return self

    def resolve_channel_schedule(self) -> list[int]:
        """Per-block output channel counts, in order."""
        if self.channels is not None:
            return list(self.channels)
        if self.channel_growth == "double":
            return [self.base_channels * (2**i) for i in range(self.num_blocks)]
        if self.channel_growth == "linear":
            return [self.base_channels * (i + 1) for i in range(self.num_blocks)]
        raise ConfigurationError(
            f"Unresolvable channel_growth: {self.channel_growth!r}"
        )  # pragma: no cover

    def compute_hash(self) -> str:
        """Content-addressed fingerprint of the fully resolved architecture."""
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_model_config(path: Path) -> ModelConfig:
    """Load and validate a model config YAML file.

    Raises:
        ConfigurationError: the file is missing, unreadable, or fails validation.
    """
    if not path.exists():
        raise ConfigurationError(f"Model config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    raw.pop("config_hash", None)  # tolerate re-loading a file written by save_model_config
    try:
        return ModelConfig(**raw)
    except ConfigurationError:
        raise
    except Exception as exc:  # pydantic.ValidationError and friends
        raise ConfigurationError(f"Invalid model config at {path}: {exc}") from exc


def save_model_config(config: ModelConfig, path: Path) -> Path:
    """Persist a model config alongside its content hash, for provenance."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = config.model_dump(mode="json")
    payload["config_hash"] = config.compute_hash()
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)
    return path
