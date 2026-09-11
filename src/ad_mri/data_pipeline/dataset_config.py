"""Per-dataset configuration schema.

The class taxonomy, folder layout, and label source are dataset-specific (an
ADNI-style export differs from an OASIS-style or a Kaggle-style export) and are
therefore never hardcoded — each dataset gets its own YAML file under
`configs/data/`, validated against this schema. See
docs/TECHNICAL_SPECIFICATION.md §4, §17.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from ad_mri.core.exceptions import ConfigurationError


class DatasetConfig(BaseModel):
    """Everything the ingestion/validation/manifest/split stages need for one dataset."""

    name: str
    dataset_source: str = Field(description="Human-readable provenance, e.g. 'OASIS-1', 'ADNI'.")
    root_dir: Path = Field(description="Directory containing one subfolder per subject.")
    modality: str = "T1w"
    file_extensions: list[str] = Field(default_factory=lambda: [".nii.gz", ".nii"])
    subject_dir_pattern: str = "*"

    labels_csv: Path = Field(description="CSV mapping subject IDs to a raw diagnosis label.")
    subject_id_column: str = "subject_id"
    label_column: str = "label"
    #: Raw label value (as it appears in labels_csv) -> canonical class name.
    #: Intentionally dataset-specific; never a hardcoded taxonomy in code.
    label_map: dict[str, str]

    #: Accepted volume rank: 3 (single 3D volume) and/or 4 (e.g. multi-echo/4D).
    expected_dim_range: tuple[int, int] = (3, 4)
    min_voxel_count: int = 100_000
    #: Security control (Phase 14, mirrors `PreprocessingConfig.max_voxel_count`):
    #: rejects a header-declared volume this large during ingestion, before
    #: any full-volume decompression/load - see that field's docstring.
    max_voxel_count: int = 100_000_000

    convert_dicom: bool = True

    seed: int = 42
    train_ratio: float = 0.7
    val_ratio: float = 0.15

    @field_validator("root_dir", "labels_csv", mode="before")
    @classmethod
    def _coerce_path(cls, value: object) -> object:
        return Path(value) if isinstance(value, str) else value

    @model_validator(mode="after")
    def _validate_ratios(self) -> DatasetConfig:
        if not (0.0 < self.train_ratio < 1.0) or not (0.0 < self.val_ratio < 1.0):
            raise ConfigurationError("train_ratio and val_ratio must each be in (0, 1).")
        if self.train_ratio + self.val_ratio >= 1.0:
            raise ConfigurationError(
                "train_ratio + val_ratio must leave a positive remainder for the test split."
            )
        if self.max_voxel_count <= self.min_voxel_count:
            raise ConfigurationError("max_voxel_count must be greater than min_voxel_count.")
        return self

    @property
    def test_ratio(self) -> float:
        return round(1.0 - self.train_ratio - self.val_ratio, 10)


def load_dataset_config(path: Path) -> DatasetConfig:
    """Load and validate a dataset config YAML file.

    Raises:
        ConfigurationError: the file is missing, unreadable, or fails validation.
    """
    if not path.exists():
        raise ConfigurationError(f"Dataset config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    try:
        return DatasetConfig(**raw)
    except Exception as exc:  # pydantic.ValidationError and friends
        raise ConfigurationError(f"Invalid dataset config at {path}: {exc}") from exc
