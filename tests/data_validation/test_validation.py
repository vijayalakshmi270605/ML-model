from __future__ import annotations

from pathlib import Path

from ad_mri.data_pipeline.dataset_config import DatasetConfig
from ad_mri.data_pipeline.ingestion import discover_subject_scans
from ad_mri.data_pipeline.validation import (
    compute_sha256,
    validate_file_exists,
    validate_file_format,
    validate_inventory,
)


def test_validate_file_exists(tmp_path: Path) -> None:
    present = tmp_path / "present.nii.gz"
    present.write_bytes(b"x")
    assert validate_file_exists(present) is None
    assert validate_file_exists(tmp_path / "missing.nii.gz") is not None


def test_validate_file_format(tmp_path: Path) -> None:
    good = tmp_path / "scan.nii.gz"
    bad = tmp_path / "scan.txt"
    assert validate_file_format(good, [".nii.gz", ".nii"]) is None
    assert validate_file_format(bad, [".nii.gz", ".nii"]) is not None


def test_compute_sha256_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "file.bin"
    path.write_bytes(b"some deterministic content")
    assert compute_sha256(path) == compute_sha256(path)


def test_validate_inventory_flags_every_expected_failure_mode(
    synthetic_dataset: DatasetConfig,
) -> None:
    inventory_df, _ = discover_subject_scans(synthetic_dataset)
    validated_df, result = validate_inventory(inventory_df, synthetic_dataset)

    by_subject = validated_df.set_index("subject_id")

    # sub-01, sub-02: valid, correctly labeled
    assert by_subject.loc["sub-01", "qc_status"] == "passed"
    assert by_subject.loc["sub-01", "label"] == "CN"
    assert by_subject.loc["sub-02", "qc_status"] == "passed"
    assert by_subject.loc["sub-02", "label"] == "AD"

    # sub-03: missing from labels.csv
    assert by_subject.loc["sub-03", "qc_status"] == "failed"

    # sub-04: corrupted file
    assert by_subject.loc["sub-04", "qc_status"] == "failed"

    # sub-05: too small (below min_voxel_count)
    assert by_subject.loc["sub-05", "qc_status"] == "failed"

    # sub-06: label not in label_map
    assert by_subject.loc["sub-06", "qc_status"] == "failed"

    assert result.total_rows == 14
    assert result.failed_rows == 4
    assert result.passed_rows == 10
    assert not result.is_valid

    failed_subjects = {err.subject_id for err in result.errors}
    assert {"sub-03", "sub-04", "sub-05", "sub-06"} <= failed_subjects


def test_validate_inventory_extracts_preprocessing_metadata(
    synthetic_dataset: DatasetConfig,
) -> None:
    inventory_df, _ = discover_subject_scans(synthetic_dataset)
    validated_df, _ = validate_inventory(inventory_df, synthetic_dataset)

    row = validated_df.set_index("subject_id").loc["sub-01"]
    assert row["num_dims"] == 3
    assert row["dim_x"] == row["dim_y"] == row["dim_z"] == 20
    assert row["voxel_spacing_x"] == 1.0
    assert row["sha256"] is not None and len(row["sha256"]) == 64
