"""Probability calibration: applied at inference time only when explicitly
configured — an uncalibrated softmax (`method: "none"`) is the default and
requires no extra step.

Implements temperature scaling (Guo et al., "On Calibration of Modern Neural
Networks", ICML 2017) — the standard, minimal post-hoc calibration method for
neural network classifiers: a single learned scalar `T` divides the logits
before softmax, sharpening (`T<1`) or softening (`T>1`) the probability
distribution without changing the predicted class (softmax is monotonic, so
argmax is invariant to a positive rescaling of the logits). `T` is fit once,
offline, against held-out logits/labels — never during inference itself
(temperature scaling only rescales, it never retrains network weights).

Mirrors the `version` + `compute_hash()` + `load_*`/`save_*` contract used by
every other config module in this project. See docs/inference.md.

`torch`/`scipy` are training-phase dependencies (requirements/train.txt),
imported lazily so this module stays importable without them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ad_mri.core.exceptions import ConfigurationError
from ad_mri.core.logging import get_logger

logger = get_logger(__name__)

CalibrationMethod = Literal["none", "temperature"]


class CalibrationConfig(BaseModel):
    """`method="none"` (the default): inference uses plain softmax, exactly as
    every earlier phase (training validation, evaluation) already does — no
    calibration step runs. `method="temperature"`: inference divides logits
    by `temperature` before softmax."""

    model_config = ConfigDict(extra="ignore")

    version: str = "1.0.0"
    method: CalibrationMethod = "none"
    temperature: float = Field(default=1.0, gt=0.0)

    def compute_hash(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def apply_calibration(logits: Any, config: CalibrationConfig) -> Any:
    """Return class probabilities for `logits` (shape `[batch, num_classes]`),
    calibrated per `config`. `method="none"` is plain softmax — identical to
    the uncalibrated path every earlier phase uses.

    Raises:
        ConfigurationError: `config.method` is not a known method (defensive —
            pydantic's `Literal` already rejects this at config load time; see
            `explainability.shap_explainer._build_explainer` for the same
            defense-in-depth pattern against a `.model_copy()`-bypassed value).
    """
    import torch

    if config.method == "none":
        return torch.softmax(logits, dim=-1)
    if config.method == "temperature":
        return torch.softmax(logits / config.temperature, dim=-1)
    raise ConfigurationError(f"Unknown calibration method: {config.method!r}")  # pragma: no cover


def fit_temperature(
    logits: Any, labels: Any, *, bounds: tuple[float, float] = (0.05, 10.0)
) -> float:
    """Fit a single temperature scalar minimizing negative log-likelihood on
    held-out `logits`/`labels` (shape `[n, num_classes]` / `[n]`) — the
    standard offline calibration-fitting procedure. Never called during
    inference; see `ad-mri calibrate fit` / `inference.calibrate`.

    Returns the fitted temperature, bounded to `bounds` (a temperature outside
    a sane range signals a fitting problem, not a real calibration need).
    """
    import torch
    from scipy.optimize import minimize_scalar

    logits_t = torch.as_tensor(logits, dtype=torch.float32)
    labels_t = torch.as_tensor(labels, dtype=torch.long)

    def _negative_log_likelihood(temperature: float) -> float:
        scaled = logits_t / temperature
        log_probs = torch.log_softmax(scaled, dim=-1)
        picked = log_probs.gather(1, labels_t.unsqueeze(1)).squeeze(1)
        return float(-picked.mean().item())

    result = minimize_scalar(_negative_log_likelihood, bounds=bounds, method="bounded")
    temperature = float(result.x)
    logger.info(
        "temperature_calibration_fit",
        temperature=round(temperature, 4),
        n_samples=int(logits_t.shape[0]),
        final_nll=round(float(result.fun), 6),
    )
    return temperature


def load_calibration_config(path: Path) -> CalibrationConfig:
    """Raises: ConfigurationError — missing file or failed validation."""
    if not path.exists():
        raise ConfigurationError(f"Calibration config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}
    raw.pop("config_hash", None)  # tolerate re-loading a file written by save_calibration_config
    try:
        return CalibrationConfig(**raw)
    except ConfigurationError:
        raise
    except Exception as exc:  # pydantic.ValidationError and friends
        raise ConfigurationError(f"Invalid calibration config at {path}: {exc}") from exc


def save_calibration_config(config: CalibrationConfig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = config.model_dump(mode="json")
    payload["config_hash"] = config.compute_hash()
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)
    return path
