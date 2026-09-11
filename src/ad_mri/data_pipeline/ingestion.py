"""Raw MRI data ingestion: discovers per-subject scans and builds a raw inventory.

Ingestion never mutates the manifest's meaning of "valid" — it only discovers
what exists on disk (NIfTI files, or DICOM series it converts) and reports what
it found or failed to find. Validation (validation.py) is a separate stage.
See docs/TECHNICAL_SPECIFICATION.md §4.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ad_mri.core.exceptions import DataValidationError, PreprocessingError
from ad_mri.core.logging import get_logger
from ad_mri.data_pipeline.conversion import convert_dicom_series_to_nifti
from ad_mri.data_pipeline.dataset_config import DatasetConfig
from ad_mri.data_pipeline.schema import IngestionReport, RowError

logger = get_logger(__name__)

INVENTORY_COLUMNS = ["subject_id", "image_path", "format", "dataset_source", "modality"]


def discover_subject_scans(
    config: DatasetConfig, interim_dir: Path | None = None
) -> tuple[pd.DataFrame, IngestionReport]:
    """Scan `config.root_dir` for one candidate scan per subject subfolder.

    A subject folder is matched by NIfTI extension first; if none is found, a
    DICOM series (`*.dcm`) is looked for and converted to NIfTI under
    `interim_dir` when `config.convert_dicom` is true.

    Returns:
        (inventory_df, report) — inventory_df has columns `INVENTORY_COLUMNS`;
        rows that could not be resolved are omitted and recorded in report.errors.
    """
    report = IngestionReport(
        source_dir=str(config.root_dir),
        destination_dir=str(interim_dir) if interim_dir else None,
    )

    if not config.root_dir.exists():
        report.errors.append(
            RowError("<root>", "root_dir", f"root_dir does not exist: {config.root_dir}")
        )
        logger.error("ingestion_root_missing", root_dir=str(config.root_dir))
        return pd.DataFrame(columns=INVENTORY_COLUMNS), report

    subject_dirs = sorted(p for p in config.root_dir.glob(config.subject_dir_pattern) if p.is_dir())
    report.subjects_discovered = len(subject_dirs)

    rows: list[dict[str, str]] = []
    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        nifti_files = sorted(
            {f for ext in config.file_extensions for f in subject_dir.rglob(f"*{ext}")}
        )

        if nifti_files:
            if len(nifti_files) > 1:
                report.errors.append(
                    RowError(
                        subject_id,
                        "image_path",
                        f"{len(nifti_files)} candidate scans found; using {nifti_files[0].name}",
                    )
                )
            rows.append(
                {
                    "subject_id": subject_id,
                    "image_path": str(nifti_files[0]),
                    "format": "nifti",
                    "dataset_source": config.dataset_source,
                    "modality": config.modality,
                }
            )
            report.scans_discovered += 1
            continue

        dicom_files = sorted(subject_dir.rglob("*.dcm"))
        if dicom_files:
            if not config.convert_dicom:
                rows.append(
                    {
                        "subject_id": subject_id,
                        "image_path": str(subject_dir),
                        "format": "dicom",
                        "dataset_source": config.dataset_source,
                        "modality": config.modality,
                    }
                )
                report.scans_discovered += 1
                continue

            if interim_dir is None:
                report.errors.append(
                    RowError(subject_id, "format", "DICOM series found but no interim_dir given")
                )
                continue

            output_path = interim_dir / config.name / f"{subject_id}.nii.gz"
            try:
                convert_dicom_series_to_nifti(subject_dir, output_path)
            except (DataValidationError, PreprocessingError) as exc:
                report.errors.append(RowError(subject_id, "dicom", str(exc)))
                logger.warning("dicom_conversion_failed", subject_id=subject_id, error=str(exc))
                continue

            rows.append(
                {
                    "subject_id": subject_id,
                    "image_path": str(output_path),
                    "format": "nifti",
                    "dataset_source": config.dataset_source,
                    "modality": config.modality,
                }
            )
            report.dicom_series_converted += 1
            report.scans_discovered += 1
            continue

        report.errors.append(RowError(subject_id, "image_path", "no NIfTI or DICOM scan found"))

    inventory_df = pd.DataFrame(rows, columns=INVENTORY_COLUMNS)
    logger.info(
        "ingestion_complete",
        subjects_discovered=report.subjects_discovered,
        scans_discovered=report.scans_discovered,
        dicom_series_converted=report.dicom_series_converted,
        errors=len(report.errors),
    )
    return inventory_df, report


def ingest_raw_scans(
    config: DatasetConfig, interim_dir: Path, inventory_output_path: Path | None = None
) -> tuple[pd.DataFrame, IngestionReport]:
    """Run discovery and persist the resulting inventory as CSV.

    This is the CLI-facing entry point; `discover_subject_scans` is the
    reusable/testable core.
    """
    inventory_df, report = discover_subject_scans(config, interim_dir=interim_dir)
    if inventory_output_path is not None:
        inventory_output_path.parent.mkdir(parents=True, exist_ok=True)
        inventory_df.to_csv(inventory_output_path, index=False)
        logger.info("inventory_written", path=str(inventory_output_path), rows=len(inventory_df))
    return inventory_df, report
