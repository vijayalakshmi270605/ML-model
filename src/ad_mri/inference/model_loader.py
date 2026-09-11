"""Loads a versioned model checkpoint + its exact training-time preprocessing
config, and caches the result in memory so a running service loads each
(checkpoint, preprocessing config, device) combination from disk **at most
once** — requirement: "cache model in memory".

Deliberately independent of the MLflow Model Registry (`model_registry.client`,
still a stub pending that phase): "a versioned model artifact" here means a
checkpoint written by `modeling.checkpoint.save_checkpoint` — the exact same
self-describing artifact `evaluation.evaluate` and
`explainability.service.ExplainabilityService` already load directly, bundled
with the architecture config and a content hash that lets it be traced back to
the code/config that produced it. Once the registry exists, this is the layer
that would be extended to resolve a registry URI to a local checkpoint path
before caching — the cache/validation logic below stays the same either way.

`torch` is a training-phase dependency (requirements/train.txt), imported
lazily so this module stays importable without it.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ad_mri.core.exceptions import ConfigurationError
from ad_mri.core.logging import get_logger
from ad_mri.datasets.preprocessing_config import PreprocessingConfig, load_preprocessing_config
from ad_mri.modeling.checkpoint import CheckpointMetadata, load_checkpoint
from ad_mri.modeling.model_config import ModelConfig

logger = get_logger(__name__)


@dataclass(frozen=True)
class CachedModel:
    """One loaded, ready-to-serve model: weights on `device`, in eval mode,
    bundled with everything needed to preprocess an input the same way it was
    trained (requirement: "load exact preprocessing configuration associated
    with that model") and to label its output (`class_names`)."""

    model: Any  # torch.nn.Module, already .eval() and .to(device)
    model_config: ModelConfig
    metadata: CheckpointMetadata
    preprocessing_config: PreprocessingConfig
    class_names: list[str]
    checkpoint_path: Path
    preprocessing_config_path: Path
    device: Any  # torch.device
    loaded_at: str  # ISO-8601 UTC


def _resolve_class_names(metadata_extra: dict[str, Any], checkpoint_path: Path) -> list[str]:
    class_names = metadata_extra.get("class_names")
    if not class_names:
        raise ConfigurationError(
            f"Checkpoint at {checkpoint_path} does not carry a class_names mapping "
            "(expected in its training-provenance metadata) — cannot reliably map "
            "predicted indices back to label names. Re-train with the current "
            "training pipeline (Phase 5+), which always records this."
        )
    return list(class_names)


def _resolve_device(device_choice: str) -> Any:
    import torch

    if device_choice == "cpu":
        return torch.device("cpu")
    if device_choice == "cuda":
        if not torch.cuda.is_available():
            raise ConfigurationError("device='cuda' was requested but CUDA is not available.")
        return torch.device("cuda")
    if device_choice != "auto":
        raise ConfigurationError(f"Unknown device choice: {device_choice!r} (use auto/cpu/cuda).")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelLoader:
    """Thread-safe, process-wide in-memory cache of loaded checkpoints.

    One `ModelLoader` instance is meant to live for the lifetime of a serving
    process (the API's DI provider constructs exactly one via
    `functools.lru_cache`; a standalone CLI/script call constructs its own,
    short-lived one) — repeated `.load()` calls with the same arguments never
    touch disk after the first.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str, str], CachedModel] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _cache_key(
        checkpoint_path: Path, preprocessing_config_path: Path, device_choice: str
    ) -> tuple[str, str, str]:
        return (
            str(Path(checkpoint_path).resolve()),
            str(Path(preprocessing_config_path).resolve()),
            device_choice,
        )

    def load(
        self,
        checkpoint_path: Path,
        preprocessing_config_path: Path,
        device_choice: str = "auto",
    ) -> CachedModel:
        """Return a `CachedModel`, loading from disk only on the first call
        for this exact `(checkpoint_path, preprocessing_config_path,
        device_choice)` combination.

        Raises:
            ConfigurationError: the checkpoint is unreadable/corrupted, lacks a
                `class_names` mapping, its architecture's `num_classes`
                disagrees with `class_names`, the preprocessing config's
                `target_shape` doesn't match the model's `input_shape`, the
                preprocessing config's content hash doesn't match the one the
                checkpoint was trained with, or `device_choice` requests CUDA
                when it's unavailable.
        """
        key = self._cache_key(checkpoint_path, preprocessing_config_path, device_choice)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached

        # Loading (disk I/O, model construction) happens outside the lock so a
        # slow first load of one model doesn't block lookups of an already-
        # cached different one; a duplicate concurrent load of the *same*
        # key is wasted work, not a correctness bug (both loads produce an
        # equivalent CachedModel, and the cache write below is still locked).
        loaded = self._load_uncached(checkpoint_path, preprocessing_config_path, device_choice)
        with self._lock:
            self._cache.setdefault(key, loaded)
            return self._cache[key]

    def _load_uncached(
        self, checkpoint_path: Path, preprocessing_config_path: Path, device_choice: str
    ) -> CachedModel:
        checkpoint_path = Path(checkpoint_path)
        preprocessing_config_path = Path(preprocessing_config_path)

        model, model_config, metadata = load_checkpoint(checkpoint_path)
        class_names = _resolve_class_names(metadata.extra, checkpoint_path)
        if len(class_names) != model_config.num_classes:
            raise ConfigurationError(
                f"Checkpoint class_names {class_names} has {len(class_names)} entries but "
                f"model_config.num_classes={model_config.num_classes}."
            )

        preprocessing_config = load_preprocessing_config(preprocessing_config_path)
        if tuple(preprocessing_config.target_shape) != tuple(model_config.input_shape):
            raise ConfigurationError(
                f"preprocessing_config.target_shape {preprocessing_config.target_shape} "
                f"!= model_config.input_shape {model_config.input_shape}."
            )
        stored_hash = metadata.extra.get("preprocessing_config_hash")
        if stored_hash and stored_hash != preprocessing_config.compute_hash():
            raise ConfigurationError(
                f"preprocessing config at {preprocessing_config_path} (hash "
                f"{preprocessing_config.compute_hash()[:12]}) does not match the one the "
                f"checkpoint was trained with (hash {stored_hash[:12]}) — serving with "
                "mismatched preprocessing would silently invalidate every prediction."
            )

        device = _resolve_device(device_choice)
        model = model.to(device)
        model.eval()

        cached = CachedModel(
            model=model,
            model_config=model_config,
            metadata=metadata,
            preprocessing_config=preprocessing_config,
            class_names=class_names,
            checkpoint_path=checkpoint_path,
            preprocessing_config_path=preprocessing_config_path,
            device=device,
            loaded_at=datetime.now(UTC).isoformat(),
        )
        logger.info(
            "model_loaded_into_cache",
            checkpoint_path=str(checkpoint_path),
            model_version=model_config.version,
            model_config_hash=model_config.compute_hash(),
            device=str(device),
            class_names=class_names,
        )
        return cached

    def is_cached(
        self, checkpoint_path: Path, preprocessing_config_path: Path, device_choice: str = "auto"
    ) -> bool:
        key = self._cache_key(checkpoint_path, preprocessing_config_path, device_choice)
        with self._lock:
            return key in self._cache

    def clear(self) -> None:
        """Evict every cached model. Mainly for tests; a long-running service
        has no routine need to call this (checkpoints are immutable once
        written — there is nothing to go stale)."""
        with self._lock:
            self._cache.clear()
