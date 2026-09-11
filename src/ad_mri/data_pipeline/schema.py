"""Shared manifest schema and report dataclasses used across the data pipeline.

The manifest is the single, versioned source of truth handed from ingestion to
splitting to `datasets.MRIDataset` — see docs/TECHNICAL_SPECIFICATION.md §4, §17.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Exact column order/names written to the manifest CSV (requirement: subject ID,
#: image path, label, dataset source, modality, dimensions, preprocessing metadata).
MANIFEST_COLUMNS: list[str] = [
    "subject_id",
    "image_path",
    "format",
    "label",
    "label_id",
    "dataset_source",
    "modality",
    "dim_x",
    "dim_y",
    "dim_z",
    "num_dims",
    "voxel_spacing_x",
    "voxel_spacing_y",
    "voxel_spacing_z",
    "dtype",
    "file_size_bytes",
    "sha256",
    "qc_status",
    "ingested_at",
]

#: Columns whose presence/non-null status is required for a row to be "valid".
REQUIRED_METADATA_FIELDS: list[str] = [
    "subject_id",
    "image_path",
    "label",
    "dataset_source",
    "modality",
    "dim_x",
    "dim_y",
    "dim_z",
    "voxel_spacing_x",
    "voxel_spacing_y",
    "voxel_spacing_z",
]

SPLIT_NAMES: tuple[str, str, str] = ("train", "val", "test")


@dataclass
class RowError:
    """One validation/ingestion failure, attributable to a specific subject/row."""

    subject_id: str
    field: str
    message: str


@dataclass
class IngestionReport:
    source_dir: str
    destination_dir: str | None
    subjects_discovered: int = 0
    scans_discovered: int = 0
    dicom_series_converted: int = 0
    errors: list[RowError] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return len(self.errors) == 0


@dataclass
class ValidationResult:
    total_rows: int
    passed_rows: int
    failed_rows: int
    errors: list[RowError] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.failed_rows == 0
