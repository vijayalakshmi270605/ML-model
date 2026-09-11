"""Dataset validation: file existence, format, corruption, dimensions, metadata,
and label checks. Produces a per-row-annotated DataFrame plus an aggregate
ValidationResult — see docs/TECHNICAL_SPECIFICATION.md §4.

Runs on the inventory produced by `ingestion.discover_subject_scans`.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pandas as pd

from ad_mri.core.exceptions import ConfigurationError
from ad_mri.core.logging import get_logger
from ad_mri.data_pipeline.dataset_config import DatasetConfig
from ad_mri.data_pipeline.schema import REQUIRED_METADATA_FIELDS, RowError, ValidationResult

logger = get_logger(__name__)

_HASH_CHUNK_SIZE = 1024 * 1024  # 1 MiB


def validate_file_exists(path: Path) -> str | None:
    """Return an error message if `path` does not exist, else None."""
    return None if path.exists() else f"file does not exist: {path}"


def validate_file_format(path: Path, allowed_extensions: list[str]) -> str | None:
    """Return an error message if `path`'s extension is not in `allowed_extensions`."""
    name = path.name.lower()
    if any(name.endswith(ext.lower()) for ext in allowed_extensions):
        return None
    return f"unsupported file extension for {path.name} (expected one of {allowed_extensions})"


def compute_sha256(path: Path) -> str:
    """Stream-hash a file without loading it fully into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_nifti_metadata(path: Path) -> dict[str, object]:
    """Read shape/spacing/dtype from a NIfTI header without loading pixel data.

    Raises:
        ValueError: the file cannot be parsed as NIfTI (corrupted/not NIfTI).
    """
    import nibabel as nib

    try:
        # nib.load's declared return type is the abstract FileBasedImage, which
        # doesn't expose these members in nibabel's stubs even though every
        # concrete image class it can return (Nifti1Image, Nifti2Image, ...)
        # does. The extension check upstream already restricts us to NIfTI files.
        image = cast("nib.Nifti1Image", nib.load(str(path)))  # lazy — no voxel read yet
        shape = image.shape
        zooms = image.header.get_zooms()
        dtype = str(image.get_data_dtype())
    except Exception as exc:  # nibabel raises a variety of exception types
        raise ValueError(f"could not read NIfTI header for {path}: {exc}") from exc

    return {
        "num_dims": len(shape),
        "dim_x": int(shape[0]) if len(shape) > 0 else None,
        "dim_y": int(shape[1]) if len(shape) > 1 else None,
        "dim_z": int(shape[2]) if len(shape) > 2 else None,
        "voxel_spacing_x": float(zooms[0]) if len(zooms) > 0 else None,
        "voxel_spacing_y": float(zooms[1]) if len(zooms) > 1 else None,
        "voxel_spacing_z": float(zooms[2]) if len(zooms) > 2 else None,
        "dtype": dtype,
    }


#: Applied when a caller doesn't pass its own `max_voxel_count` - matches
#: `PreprocessingConfig`/`DatasetConfig`'s own default, so `check_shape` is
#: never silently unbounded above even if a future call site forgets to wire
#: one through explicitly. See `PreprocessingConfig.max_voxel_count`'s
#: docstring for the security rationale (Phase 14, docs/security.md).
_DEFAULT_MAX_VOXEL_COUNT = 100_000_000


def check_shape(
    num_dims: object,
    dims: tuple[object, object, object],
    expected_dim_range: tuple[int, int],
    min_voxel_count: int,
    max_voxel_count: int = _DEFAULT_MAX_VOXEL_COUNT,
) -> list[str]:
    """Pure dimensionality/voxel-count check, shared by ingestion-time validation
    (`validate_image_dimensions`, below) and the runtime preprocessing pipeline's
    input check (`datasets.validation`) — one rule, two call sites.

    The upper bound (`max_voxel_count`) is a security control, not just a data
    sanity check: a NIfTI header declares its shape in the first few hundred
    bytes, readable without decompressing the rest of a `.nii.gz` — so this
    check runs, and can reject a maliciously (or corruptly) huge declared
    volume, *before* any caller attempts the expensive full decompression/
    allocation a "decompression bomb" upload is designed to trigger.
    """
    errors: list[str] = []
    low, high = expected_dim_range
    if not isinstance(num_dims, int) or not (low <= num_dims <= high):
        errors.append(f"unexpected dimensionality {num_dims} (expected {low}-{high})")

    known_dims = [d for d in dims if isinstance(d, int)]
    if len(known_dims) == len(dims):
        voxel_count = 1
        for d in known_dims:
            voxel_count *= d
        if voxel_count < min_voxel_count:
            errors.append(f"volume too small: {voxel_count} voxels (minimum {min_voxel_count})")
        elif voxel_count > max_voxel_count:
            errors.append(f"volume too large: {voxel_count} voxels (maximum {max_voxel_count})")
    return errors


def validate_image_dimensions(metadata: dict[str, object], config: DatasetConfig) -> list[str]:
    """Check num_dims range and a minimum/maximum total-voxel-count bound."""
    dims = (metadata.get("dim_x"), metadata.get("dim_y"), metadata.get("dim_z"))
    return check_shape(
        metadata.get("num_dims"),
        dims,
        config.expected_dim_range,
        config.min_voxel_count,
        config.max_voxel_count,
    )


def _load_labels(config: DatasetConfig) -> pd.DataFrame:
    if not config.labels_csv.exists():
        raise ConfigurationError(f"labels_csv not found: {config.labels_csv}")
    labels_df = pd.read_csv(config.labels_csv, dtype=str)
    missing_cols = {config.subject_id_column, config.label_column} - set(labels_df.columns)
    if missing_cols:
        raise ConfigurationError(f"labels_csv is missing required column(s): {missing_cols}")
    return labels_df[[config.subject_id_column, config.label_column]].rename(
        columns={config.subject_id_column: "subject_id", config.label_column: "raw_label"}
    )


def validate_inventory(
    inventory_df: pd.DataFrame, config: DatasetConfig
) -> tuple[pd.DataFrame, ValidationResult]:
    """Validate every row of `inventory_df`, extracting preprocessing metadata.

    Returns the inventory joined with labels + extracted metadata + a
    `qc_status` ("passed"/"failed") column, alongside an aggregate
    ValidationResult listing every row-level failure.
    """
    labels_df = _load_labels(config)
    merged = inventory_df.merge(labels_df, on="subject_id", how="left")

    row_errors: list[RowError] = []
    records: list[dict[str, object]] = []
    now = datetime.now(UTC).isoformat()

    for row in merged.itertuples(index=False):
        subject_id = str(row.subject_id)
        image_path = Path(str(row.image_path))
        record: dict[str, object] = {
            "subject_id": subject_id,
            "image_path": str(image_path),
            "format": row.format,
            "dataset_source": row.dataset_source,
            "modality": row.modality,
            "dim_x": None,
            "dim_y": None,
            "dim_z": None,
            "num_dims": None,
            "voxel_spacing_x": None,
            "voxel_spacing_y": None,
            "voxel_spacing_z": None,
            "dtype": None,
            "file_size_bytes": None,
            "sha256": None,
            "label": None,
            "label_id": None,
            "qc_status": "passed",
            "ingested_at": now,
        }
        row_ok = True

        if (error := validate_file_exists(image_path)) or (
            error := validate_file_format(image_path, config.file_extensions)
        ):
            row_errors.append(RowError(subject_id, "image_path", error))
            row_ok = False
        else:
            record["file_size_bytes"] = image_path.stat().st_size
            try:
                metadata = load_nifti_metadata(image_path)
            except ValueError as exc:
                row_errors.append(RowError(subject_id, "image_path", str(exc)))
                row_ok = False
            else:
                record.update(metadata)
                dim_errors = validate_image_dimensions(metadata, config)
                for msg in dim_errors:
                    row_errors.append(RowError(subject_id, "dimensions", msg))
                    row_ok = False
                record["sha256"] = compute_sha256(image_path)

        raw_label = getattr(row, "raw_label", None)
        if raw_label is None or (isinstance(raw_label, float) and pd.isna(raw_label)):
            row_errors.append(RowError(subject_id, "label", "no label found in labels_csv"))
            row_ok = False
        elif raw_label not in config.label_map:
            row_errors.append(
                RowError(subject_id, "label", f"raw label {raw_label!r} not in label_map")
            )
            row_ok = False
        else:
            canonical_label = config.label_map[raw_label]
            record["label"] = canonical_label
            record["label_id"] = sorted(set(config.label_map.values())).index(canonical_label)

        for field_name in REQUIRED_METADATA_FIELDS:
            if record.get(field_name) in (None, ""):
                row_errors.append(RowError(subject_id, field_name, "required field missing"))
                row_ok = False

        record["qc_status"] = "passed" if row_ok else "failed"
        records.append(record)

    validated_df = pd.DataFrame(records)
    passed = int((validated_df["qc_status"] == "passed").sum()) if len(validated_df) else 0
    result = ValidationResult(
        total_rows=len(validated_df),
        passed_rows=passed,
        failed_rows=len(validated_df) - passed,
        errors=row_errors,
    )

    logger.info(
        "validation_complete",
        total_rows=result.total_rows,
        passed_rows=result.passed_rows,
        failed_rows=result.failed_rows,
    )
    for err in row_errors:
        logger.warning(
            "validation_error", subject_id=err.subject_id, field=err.field, message=err.message
        )

    return validated_df, result
