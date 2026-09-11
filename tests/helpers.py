"""Shared test helpers. Not a test module itself (no `test_` prefix, not collected)."""

from __future__ import annotations

from pathlib import Path


def write_synthetic_nifti(
    path: Path, shape: tuple[int, int, int] = (32, 32, 32), seed: int = 0
) -> Path:
    """Write a small random NIfTI volume to `path`. No real patient data anywhere in tests."""
    import nibabel as nib
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.random.default_rng(seed).random(shape).astype(np.float32)
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    return path


def write_synthetic_manifest(
    directory: Path,
    classes: tuple[str, ...] = ("CN", "AD"),
    num_per_class: int = 3,
    shape: tuple[int, int, int] = (16, 16, 16),
    prefix: str = "subj",
) -> Path:
    """Write `num_per_class` synthetic NIfTI scans per class + a matching manifest CSV
    (using the real `MANIFEST_COLUMNS` schema, so it's a faithful stand-in for a
    Phase 2 `ad-mri data manifest` output)."""
    import pandas as pd

    from ad_mri.data_pipeline.schema import MANIFEST_COLUMNS

    rows = []
    counter = 0
    for label in classes:
        for i in range(num_per_class):
            subject_id = f"{prefix}-{label}-{i}"
            image_path = write_synthetic_nifti(
                directory / f"{subject_id}.nii.gz", shape=shape, seed=counter
            )
            row = dict.fromkeys(MANIFEST_COLUMNS)
            row.update({"subject_id": subject_id, "image_path": str(image_path), "label": label})
            rows.append(row)
            counter += 1

    manifest_path = directory / "manifest.csv"
    pd.DataFrame(rows, columns=MANIFEST_COLUMNS).to_csv(manifest_path, index=False)
    return manifest_path
