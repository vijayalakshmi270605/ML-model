"""Shared synthetic-MRI fixtures for the data-pipeline test suite.

All "scans" here are small random NIfTI volumes written with nibabel — no real
patient data is ever used in tests (see docs/TECHNICAL_SPECIFICATION.md §0, §17).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ad_mri.data_pipeline.dataset_config import DatasetConfig

nib = pytest.importorskip("nibabel")


def write_synthetic_nifti(path: Path, shape: tuple[int, ...] = (20, 20, 20)) -> Path:
    """Write a small random NIfTI volume to `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.random.default_rng(0).random(shape).astype(np.float32)
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    return path


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> DatasetConfig:
    """A small end-to-end synthetic dataset covering every validation failure mode:

    - sub-01, sub-02: valid scans with valid labels (CN / AD)
    - sub-03: valid scan, but MISSING from labels.csv
    - sub-04: file that fails to parse as NIfTI (corrupted)
    - sub-05: valid NIfTI but too small (dims below min_voxel_count)
    - sub-06: valid scan, label present but NOT in label_map (invalid label)
    - sub-07..sub-14: valid scans/labels, enough volume for a train/val/test split
    """
    root = tmp_path / "raw"

    write_synthetic_nifti(root / "sub-01" / "scan.nii.gz")
    write_synthetic_nifti(root / "sub-02" / "scan.nii.gz")
    write_synthetic_nifti(root / "sub-03" / "scan.nii.gz")

    corrupted = root / "sub-04" / "scan.nii.gz"
    corrupted.parent.mkdir(parents=True, exist_ok=True)
    corrupted.write_bytes(b"this is not a real nifti file")

    write_synthetic_nifti(root / "sub-05" / "scan.nii.gz", shape=(2, 2, 2))
    write_synthetic_nifti(root / "sub-06" / "scan.nii.gz")

    extra_labels = []
    for i in range(7, 15):
        label = "Nondemented" if i % 2 == 0 else "Demented"
        write_synthetic_nifti(root / f"sub-{i:02d}" / "scan.nii.gz")
        extra_labels.append({"subject_id": f"sub-{i:02d}", "diagnosis": label})

    labels_csv = tmp_path / "labels.csv"
    rows = [
        {"subject_id": "sub-01", "diagnosis": "Nondemented"},
        {"subject_id": "sub-02", "diagnosis": "Demented"},
        # sub-03 intentionally absent
        {"subject_id": "sub-05", "diagnosis": "Nondemented"},
        {"subject_id": "sub-06", "diagnosis": "SomeUnknownLabel"},
        *extra_labels,
    ]
    pd.DataFrame(rows).to_csv(labels_csv, index=False)

    return DatasetConfig(
        name="synthetic",
        dataset_source="unit-test",
        root_dir=root,
        modality="T1w",
        file_extensions=[".nii.gz", ".nii"],
        labels_csv=labels_csv,
        subject_id_column="subject_id",
        label_column="diagnosis",
        label_map={"Nondemented": "CN", "Demented": "AD"},
        expected_dim_range=(3, 4),
        min_voxel_count=1000,
        convert_dicom=True,
        seed=42,
        train_ratio=0.7,
        val_ratio=0.15,
    )
