"""Training run configuration schema.

Every training hyperparameter (batch size, LR, epochs, optimizer, scheduler,
weight decay, early stopping, class-imbalance strategy, mixed precision,
determinism) is config-driven — never a literal in `trainer.py`. Mirrors the
`PreprocessingConfig`/`ModelConfig` pattern: a human semver `version` +
`compute_hash()`, so a training run — and the checkpoints it produces — can
always be traced back to the exact hyperparameters used. See
docs/TECHNICAL_SPECIFICATION.md §5, §14, §16.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ad_mri.core.exceptions import ConfigurationError

OptimizerName = Literal["adam", "adamw", "sgd"]
SchedulerName = Literal["none", "step", "cosine", "plateau"]
ClassImbalanceStrategy = Literal["none", "sampler", "loss_weights"]
EarlyStoppingMode = Literal["min", "max"]


class TrainingConfig(BaseModel):
    """Hyperparameters and run-level settings for `training.trainer.Trainer`."""

    model_config = ConfigDict(extra="ignore")

    version: str = "1.0.0"

    # --- Core loop ---
    batch_size: int = Field(default=8, ge=1)
    epochs: int = Field(default=50, ge=1)
    num_workers: int = Field(default=0, ge=0)

    # --- Optimizer ---
    optimizer: OptimizerName = "adamw"
    learning_rate: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    momentum: float = Field(default=0.9, ge=0.0, le=1.0)  # sgd only
    gradient_clip_norm: float | None = Field(default=1.0, gt=0.0)

    # --- LR scheduler ---
    scheduler: SchedulerName = "cosine"
    scheduler_step_size: int = Field(default=10, ge=1)  # step
    scheduler_gamma: float = Field(default=0.1, gt=0.0, lt=1.0)  # step
    scheduler_t_max: int | None = Field(default=None, ge=1)  # cosine; defaults to `epochs`
    scheduler_patience: int = Field(default=5, ge=1)  # plateau

    # --- Early stopping ---
    early_stopping_patience: int = Field(default=10, ge=1)
    early_stopping_metric: str = "val_loss"
    early_stopping_mode: EarlyStoppingMode = "min"
    early_stopping_min_delta: float = Field(default=0.0, ge=0.0)

    # --- Class imbalance (see data_pipeline.class_balance) ---
    class_imbalance_strategy: ClassImbalanceStrategy = "loss_weights"

    # --- Mixed precision / device ---
    mixed_precision: bool = True
    device: Literal["auto", "cpu", "cuda"] = "auto"

    # --- Determinism ---
    seed: int = 42
    #: Stricter (and slower) determinism via cudnn flags +
    #: `torch.use_deterministic_algorithms`. Off by default — full bitwise
    #: determinism is rarely load-bearing for this project and costs
    #: throughput; `seed` alone already makes runs reproducible in practice.
    deterministic: bool = False

    # --- Checkpointing / output ---
    checkpoint_dir: Path = Path("artifacts/checkpoints")
    save_every_n_epochs: int = Field(default=1, ge=1)

    # --- MLflow ---
    mlflow_enabled: bool = True
    mlflow_experiment_name: str = "ad-mri"
    #: None -> mlflow's own default (local ./mlruns) or Settings.mlflow_tracking_uri,
    #: resolved by the caller (training.train), not hardcoded here.
    mlflow_tracking_uri: str | None = None

    log_every_n_steps: int = Field(default=10, ge=1)

    @field_validator("checkpoint_dir", mode="before")
    @classmethod
    def _coerce_path(cls, value: object) -> object:
        return Path(value) if isinstance(value, str) else value

    def resolve_scheduler_t_max(self) -> int:
        return self.scheduler_t_max if self.scheduler_t_max is not None else self.epochs

    def compute_hash(self) -> str:
        """Content-addressed fingerprint of the fully resolved training config."""
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_training_config(path: Path) -> TrainingConfig:
    """Load and validate a training config YAML file.

    Raises:
        ConfigurationError: the file is missing, unreadable, or fails validation.
    """
    if not path.exists():
        raise ConfigurationError(f"Training config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    raw.pop("config_hash", None)  # tolerate re-loading a file written by save_training_config
    try:
        return TrainingConfig(**raw)
    except ConfigurationError:
        raise
    except Exception as exc:  # pydantic.ValidationError and friends
        raise ConfigurationError(f"Invalid training config at {path}: {exc}") from exc


def save_training_config(config: TrainingConfig, path: Path) -> Path:
    """Persist a training config alongside its content hash, for provenance."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = config.model_dump(mode="json")
    payload["config_hash"] = config.compute_hash()
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)
    return path
