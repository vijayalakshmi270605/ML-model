"""Hyperparameter-search configuration schema.

Defines the search space (learning rate, batch size, optimizer, weight decay,
dropout, architecture parameters — requirement: "search over" exactly these)
and the search budget/objective. Every trial's sampled `ModelConfig` +
`TrainingConfig` still goes through those models' own validation, so an
invalid sampled combination (e.g. `input_shape` too small for a sampled
`num_blocks`) fails loudly rather than silently producing a broken run. See
docs/TECHNICAL_SPECIFICATION.md §5, §16, and docs/model_selection.md.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ad_mri.core.exceptions import ConfigurationError

_OptimizerChoice = Literal["adam", "adamw", "sgd"]


def _default_optimizers() -> list[_OptimizerChoice]:
    return ["adam", "adamw", "sgd"]


class HPOSearchSpace(BaseModel):
    """One field per hyperparameter under search. A `list` means "categorical
    choice"; a `(low, high)` tuple means "sample continuously in that range"
    (log-uniform for parameters spanning orders of magnitude, e.g. LR/weight
    decay)."""

    model_config = ConfigDict(extra="ignore")

    learning_rate: tuple[float, float] = (1e-5, 1e-2)
    batch_size: list[int] = Field(default_factory=lambda: [4, 8, 16])
    optimizer: list[_OptimizerChoice] = Field(default_factory=_default_optimizers)
    weight_decay: tuple[float, float] = (1e-6, 1e-2)
    dropout_fc: tuple[float, float] = (0.1, 0.7)

    # --- architecture parameters ---
    base_channels: list[int] = Field(default_factory=lambda: [16, 32, 64])
    num_blocks: list[int] = Field(default_factory=lambda: [3, 4, 5])


class HPOConfig(BaseModel):
    """Search budget, cross-validation settings, and the selection objective."""

    model_config = ConfigDict(extra="ignore")

    version: str = "1.0.0"
    study_name: str = "ad-mri-hpo"
    n_trials: int = Field(default=20, ge=1)
    seed: int = 42

    # --- cross-validation used to evaluate each trial ---
    n_folds: int = Field(default=5, ge=2)
    #: Shorter than a full training run — HPO needs many trials, not each one
    #: trained to full convergence. Final model retraining (after the best
    #: hyperparameters are selected) uses the full `epochs` from the base
    #: TrainingConfig instead.
    epochs_per_trial: int = Field(default=10, ge=1)

    # --- selection objective (requirement: "predefined validation metric") ---
    metric_name: str = "f1_macro"
    metric_mode: Literal["min", "max"] = "max"

    search_space: HPOSearchSpace = Field(default_factory=HPOSearchSpace)

    mlflow_enabled: bool = True
    mlflow_experiment_name: str = "ad-mri-hpo"
    mlflow_tracking_uri: str | None = None

    def compute_hash(self) -> str:
        """Content-addressed fingerprint of the fully resolved HPO config."""
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_hpo_config(path: Path) -> HPOConfig:
    """Load and validate an HPO config YAML file.

    Raises:
        ConfigurationError: the file is missing, unreadable, or fails validation.
    """
    if not path.exists():
        raise ConfigurationError(f"HPO config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    raw.pop("config_hash", None)
    try:
        return HPOConfig(**raw)
    except ConfigurationError:
        raise
    except Exception as exc:  # pydantic.ValidationError and friends
        raise ConfigurationError(f"Invalid HPO config at {path}: {exc}") from exc


def save_hpo_config(config: HPOConfig, path: Path) -> Path:
    """Persist an HPO config alongside its content hash, for provenance."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = config.model_dump(mode="json")
    payload["config_hash"] = config.compute_hash()
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)
    return path
