"""Preprocessing/augmentation configuration schema.

This is the single configuration surface for `datasets.transforms` — every
spatial/intensity/QC/augmentation parameter used by the preprocessing pipeline
comes from here, never from a literal in `transforms.py`. See
docs/TECHNICAL_SPECIFICATION.md §4, §5, §6, §14.

Reproducibility contract (requirement: "save preprocessing configuration/
version"): every `PreprocessingConfig` carries a human-set semantic `version`
(bump it when you change preprocessing *logic*, e.g. add a normalization mode)
and exposes `compute_hash()`, a content-addressed fingerprint of the exact
resolved parameters. `save_preprocessing_config` persists both alongside a
trained model/manifest so a preprocessed dataset or a served prediction can
always be traced back to the exact preprocessing recipe that produced it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ad_mri.core.exceptions import ConfigurationError

NormalizationMode = Literal["zscore_nonzero", "zscore", "percentile_clip", "min_max"]
QualityControlMode = Literal["raise", "warn"]


class AugmentationConfig(BaseModel):
    """Training-only randomized augmentation — never applied at val/test/inference.

    Ranges default to small, medically-plausible perturbations of head
    positioning/scanner intensity variation (see module docstring in
    `transforms.py` for the rationale behind what is and isn't included here).
    """

    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    probability: float = Field(default=0.5, ge=0.0, le=1.0)

    #: Small-angle rotation, applied independently per axis — simulates head
    #: tilt/positioning variation, not a wholesale reorientation.
    rotation_range_deg: float = Field(default=10.0, ge=0.0, le=25.0)
    #: Small translation, in voxels — simulates imperfect centering.
    translation_range_voxels: float = Field(default=5.0, ge=0.0)
    #: Isotropic scale jitter as a fraction (0.1 = +/-10%) — simulates head-size
    #: variation/FOV differences, not a growth/atrophy simulation.
    scale_range: float = Field(default=0.1, ge=0.0, le=0.3)

    #: Additive intensity shift/scale jitter, fractions of the (already
    #: normalized) intensity scale — simulates scanner/protocol variation.
    intensity_shift_range: float = Field(default=0.1, ge=0.0)
    intensity_scale_range: float = Field(default=0.1, ge=0.0)
    gaussian_noise_std: float = Field(default=0.01, ge=0.0)

    #: Left-right flip is OFF by default: Alzheimer's-related atrophy can be
    #: asymmetric, so mirroring the volume is not guaranteed to produce a
    #: medically valid sample for this task. Enable explicitly only if you
    #: have verified it's appropriate for your dataset/label definition.
    allow_flip: bool = False
    flip_probability: float = Field(default=0.5, ge=0.0, le=1.0)


class PreprocessingConfig(BaseModel):
    """Deterministic preprocessing + (optional) training augmentation parameters."""

    model_config = ConfigDict(extra="ignore")

    version: str = "1.0.0"

    # --- Spatial standardization ---
    target_orientation: str = "RAS"
    target_spacing: tuple[float, float, float] = (1.5, 1.5, 1.5)
    target_shape: tuple[int, int, int] = (128, 128, 128)
    interpolation_mode: Literal["bilinear", "nearest"] = "bilinear"

    # --- Intensity normalization ---
    normalization: NormalizationMode = "zscore_nonzero"
    #: Used only when normalization == "percentile_clip".
    percentile_clip: tuple[float, float] = (0.5, 99.5)

    # --- Runtime input validation (before/around the transform pipeline) ---
    expected_dim_range: tuple[int, int] = (3, 4)
    min_voxel_count: int = 100_000
    #: Security control (Phase 14): rejects a header-declared volume this
    #: large *before* the pipeline ever decompresses/allocates it - the
    #: mitigation for a decompression-bomb-style malicious upload (a small
    #: compressed `.nii.gz` whose header declares an enormous shape). Reading
    #: a NIfTI header is cheap regardless of the file's compressed or
    #: declared-uncompressed size (nibabel only decompresses the first few
    #: hundred bytes to parse it), so this check runs *before* any expensive
    #: full-volume load is attempted - see `datasets.validation
    #: .validate_image_for_pipeline` and docs/security.md "Malicious files".
    #: 100M voxels is a generous ceiling for any real clinical/research MRI
    #: (a 512x512x300 scan is ~79M) while still bounding worst-case memory.
    max_voxel_count: int = 100_000_000

    # --- Optional quality control (post-preprocessing sanity checks) ---
    enable_quality_control: bool = True
    qc_mode: QualityControlMode = "raise"
    qc_min_nonzero_ratio: float = Field(default=0.01, ge=0.0, le=1.0)

    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)

    #: Seeds MONAI's random transforms for reproducible augmentation (e.g. in
    #: tests/demos). Training runs normally rely on the global seed
    #: (`core.seeding.set_global_seed`) plus per-worker DataLoader seeding
    #: instead of pinning this.
    seed: int | None = None

    @field_validator(
        "target_spacing", "target_shape", "percentile_clip", "expected_dim_range", mode="before"
    )
    @classmethod
    def _coerce_tuple(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_ranges(self) -> PreprocessingConfig:
        if len(self.target_spacing) != 3 or any(v <= 0 for v in self.target_spacing):
            raise ConfigurationError("target_spacing must be 3 positive values.")
        if len(self.target_shape) != 3 or any(v <= 0 for v in self.target_shape):
            raise ConfigurationError("target_shape must be 3 positive integers.")
        low, high = self.percentile_clip
        if not (0 <= low < high <= 100):
            raise ConfigurationError("percentile_clip must satisfy 0 <= low < high <= 100.")
        dim_low, dim_high = self.expected_dim_range
        if not (0 < dim_low <= dim_high):
            raise ConfigurationError("expected_dim_range must satisfy 0 < low <= high.")
        if self.max_voxel_count <= self.min_voxel_count:
            raise ConfigurationError("max_voxel_count must be greater than min_voxel_count.")
        return self

    def compute_hash(self) -> str:
        """Content-addressed fingerprint of the fully resolved config."""
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_preprocessing_config(path: Path) -> PreprocessingConfig:
    """Load and validate a preprocessing config YAML file.

    Raises:
        ConfigurationError: the file is missing, unreadable, or fails validation.
    """
    if not path.exists():
        raise ConfigurationError(f"Preprocessing config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    raw.pop("config_hash", None)  # tolerate re-loading a file written by save_preprocessing_config
    try:
        return PreprocessingConfig(**raw)
    except ConfigurationError:
        raise
    except Exception as exc:  # pydantic.ValidationError and friends
        raise ConfigurationError(f"Invalid preprocessing config at {path}: {exc}") from exc


def save_preprocessing_config(config: PreprocessingConfig, path: Path) -> Path:
    """Persist a preprocessing config alongside its content hash, for provenance.

    Written next to a manifest/model artifact so any downstream consumer
    (a training run, a served model) can be traced back to the exact
    preprocessing recipe used to produce it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = config.model_dump(mode="json")
    payload["config_hash"] = config.compute_hash()
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)
    return path
