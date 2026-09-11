"""Model checkpoint save/load — always bundled with the exact architecture
configuration that produced it, never weights alone.

A `.pt` checkpoint written here is self-describing: reloading it does not
require the caller to already know (or guess) the architecture's
hyperparameters, and `ModelConfig.compute_hash()` lets a checkpoint be traced
back to its exact resolved architecture. See
docs/TECHNICAL_SPECIFICATION.md §5, §16.

`torch` is a training-phase dependency (requirements/train.txt), imported
lazily so this module stays importable without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ad_mri.core.exceptions import ConfigurationError
from ad_mri.core.logging import get_logger
from ad_mri.modeling.factory import build_model
from ad_mri.modeling.model_config import ModelConfig

logger = get_logger(__name__)

_CHECKPOINT_FORMAT_VERSION = "1.0.0"


@dataclass(frozen=True)
class CheckpointMetadata:
    model_config_hash: str
    model_version: str
    checkpoint_format_version: str
    saved_at: str
    extra: dict[str, Any]


def save_checkpoint(
    model: Any, config: ModelConfig, path: Path, extra: dict[str, Any] | None = None
) -> Path:
    """Save `model`'s weights bundled with `config` (and its content hash).

    `extra` may carry training-phase provenance (epoch, optimizer state,
    dataset/preprocessing config hashes, git commit, ...) once the training
    loop exists — this function itself stays training-agnostic.
    """
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "checkpoint_format_version": _CHECKPOINT_FORMAT_VERSION,
        "state_dict": model.state_dict(),
        "model_config": config.model_dump(mode="json"),
        "model_config_hash": config.compute_hash(),
        "saved_at": datetime.now(UTC).isoformat(),
        "extra": extra or {},
    }
    torch.save(payload, path)
    logger.info(
        "checkpoint_saved",
        path=str(path),
        model_config_hash=payload["model_config_hash"],
        num_parameters=sum(p.numel() for p in model.parameters()),
    )
    return path


def load_checkpoint(
    path: Path, map_location: str = "cpu"
) -> tuple[Any, ModelConfig, CheckpointMetadata]:
    """Load a checkpoint written by `save_checkpoint`: rebuilds the exact
    architecture from its saved config, then loads the weights into it.

    Raises:
        ConfigurationError: `path` doesn't exist, or isn't a checkpoint
            written by `save_checkpoint` (missing required keys).
    """
    import torch

    if not path.exists():
        raise ConfigurationError(f"Checkpoint not found: {path}")

    # weights_only=True (the safe default since torch 2.6): everything this
    # module ever writes is a tensor state_dict plus plain JSON-compatible
    # values (str/dict/float/...) - never a pickled arbitrary object - so
    # nothing here needs the unsafe full-unpickling path. Rejects a
    # checkpoint that isn't shaped that way instead of silently deserializing
    # arbitrary code.
    payload = torch.load(path, map_location=map_location, weights_only=True)
    required_keys = {"state_dict", "model_config", "model_config_hash", "saved_at"}
    missing = required_keys - payload.keys()
    if missing:
        raise ConfigurationError(f"Checkpoint at {path} is missing required key(s): {missing}")

    config = ModelConfig(**payload["model_config"])
    if config.compute_hash() != payload["model_config_hash"]:
        raise ConfigurationError(
            f"Checkpoint at {path} is corrupted: stored model_config_hash does not "
            "match the hash of its own stored model_config."
        )

    model = build_model(config)
    model.load_state_dict(payload["state_dict"])

    metadata = CheckpointMetadata(
        model_config_hash=payload["model_config_hash"],
        model_version=config.version,
        checkpoint_format_version=payload.get("checkpoint_format_version", "unknown"),
        saved_at=payload["saved_at"],
        extra=payload.get("extra", {}),
    )
    logger.info("checkpoint_loaded", path=str(path), model_config_hash=metadata.model_config_hash)
    return model, config, metadata
