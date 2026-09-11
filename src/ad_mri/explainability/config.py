"""LIME/SHAP explanation-parameter configuration schemas.

Every parameter that affects the *output* of an explanation (sample counts,
segmentation algorithm, which slice is "representative", background set size,
random seed) is config-driven and content-hashed — never a hardcoded literal
in `lime_explainer.py`/`shap_explainer.py` — so an explanation artifact can
always be traced back to exactly the parameters that produced it (requirement:
"make explanation reproducible" / "store ... explanation parameters"). Mirrors
the `version` + `compute_hash()` + `load_*`/`save_*` contract used by every
other config module in this project (`ModelConfig`, `PreprocessingConfig`,
`TrainingConfig`, `HPOConfig`). See docs/explainability.md.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ad_mri.core.exceptions import ConfigurationError

SegmentationAlgorithmName = Literal["quickshift", "slic"]
ShapExplainerKind = Literal["gradient", "deep"]


class LimeConfig(BaseModel):
    """LIME image-explanation parameters. Operates on one representative 2D
    slice of a (possibly 3D) volume — see `explainability.slicing` for why."""

    model_config = ConfigDict(extra="ignore")

    version: str = "1.0.0"

    #: Which spatial axis to slice for a 3D volume (0/1/2); ignored for a 2D model.
    slice_axis: int = Field(default=0, ge=0, le=2)
    #: "middle" or an integer string (see `slicing.resolve_slice_index`).
    slice_index: str = "middle"

    #: Number of perturbed-neighborhood samples used to fit the local surrogate
    #: model — the dominant cost driver (one 3D forward pass per sample).
    num_samples: int = Field(default=200, ge=10)
    #: Batch size for the forward passes LIME issues internally.
    batch_size: int = Field(default=8, ge=1)
    #: Max number of superpixels reported in the explanation (interpretability,
    #: not a hard cap on how many LIME considers internally).
    num_features: int = Field(default=10, ge=1)

    segmentation_algorithm: SegmentationAlgorithmName = "quickshift"
    # quickshift parameters (ignored when segmentation_algorithm == "slic")
    quickshift_kernel_size: float = 4.0
    quickshift_max_dist: float = 200.0
    quickshift_ratio: float = Field(default=0.2, ge=0.0, le=1.0)
    # slic parameters (ignored when segmentation_algorithm == "quickshift")
    slic_n_segments: int = Field(default=50, ge=2)
    slic_compactness: float = Field(default=10.0, gt=0.0)

    seed: int = 42

    def compute_hash(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ShapConfig(BaseModel):
    """SHAP explanation parameters. `GradientExplainer` ("gradient") is the
    default — an expected-gradients method well-suited to CNNs that operates
    directly on the full input tensor (no flattening); "deep" selects
    `DeepExplainer` instead, for architectures where it applies more cleanly."""

    model_config = ConfigDict(extra="ignore")

    version: str = "1.0.0"

    explainer: ShapExplainerKind = "gradient"
    #: Number of real, preprocessed samples drawn from the (versioned) background
    #: manifest to serve as the reference/baseline distribution.
    n_background_samples: int = Field(default=8, ge=1)
    #: Expected-gradients interpolation samples per explanation (GradientExplainer).
    nsamples: int = Field(default=50, ge=1)

    #: Slice used only for the overlay PNG / global-importance visualization —
    #: the SHAP values themselves are computed over the *entire* volume.
    slice_axis: int = Field(default=0, ge=0, le=2)
    slice_index: str = "middle"

    seed: int = 42

    def compute_hash(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _read_yaml(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise ConfigurationError(f"{label} config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    raw.pop("config_hash", None)  # tolerate re-loading a file written by save_*_config
    return dict(raw)


def _write_yaml(config: LimeConfig | ShapConfig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = config.model_dump(mode="json")
    payload["config_hash"] = config.compute_hash()
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)
    return path


def load_lime_config(path: Path) -> LimeConfig:
    """Raises: ConfigurationError — missing file or failed validation."""
    try:
        return LimeConfig(**_read_yaml(path, "LIME"))
    except ConfigurationError:
        raise
    except Exception as exc:  # pydantic.ValidationError and friends
        raise ConfigurationError(f"Invalid LIME config at {path}: {exc}") from exc


def save_lime_config(config: LimeConfig, path: Path) -> Path:
    return _write_yaml(config, path)


def load_shap_config(path: Path) -> ShapConfig:
    """Raises: ConfigurationError — missing file or failed validation."""
    try:
        return ShapConfig(**_read_yaml(path, "SHAP"))
    except ConfigurationError:
        raise
    except Exception as exc:  # pydantic.ValidationError and friends
        raise ConfigurationError(f"Invalid SHAP config at {path}: {exc}") from exc


def save_shap_config(config: ShapConfig, path: Path) -> Path:
    return _write_yaml(config, path)
