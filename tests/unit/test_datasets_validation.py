"""Tests for datasets.validation: single-image, fail-fast checks before preprocessing."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("nibabel")

from ad_mri.core.exceptions import DataValidationError  # noqa: E402
from ad_mri.datasets.preprocessing_config import PreprocessingConfig  # noqa: E402
from ad_mri.datasets.validation import validate_affine, validate_image_for_pipeline  # noqa: E402
from tests.helpers import write_synthetic_nifti  # noqa: E402


def test_validate_image_for_pipeline_accepts_a_good_image(tmp_path: Path) -> None:
    image = write_synthetic_nifti(tmp_path / "scan.nii.gz", shape=(32, 32, 32))
    metadata = validate_image_for_pipeline(image, PreprocessingConfig(min_voxel_count=1000))
    assert metadata["num_dims"] == 3


def test_validate_image_for_pipeline_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DataValidationError):
        validate_image_for_pipeline(tmp_path / "missing.nii.gz", PreprocessingConfig())


def test_validate_image_for_pipeline_rejects_a_corrupted_file(tmp_path: Path) -> None:
    corrupted = tmp_path / "corrupted.nii.gz"
    corrupted.write_bytes(b"not a real nifti file")
    with pytest.raises(DataValidationError):
        validate_image_for_pipeline(corrupted, PreprocessingConfig())


def test_validate_image_for_pipeline_rejects_a_too_small_volume(tmp_path: Path) -> None:
    image = write_synthetic_nifti(tmp_path / "tiny.nii.gz", shape=(2, 2, 2))
    with pytest.raises(DataValidationError):
        validate_image_for_pipeline(image, PreprocessingConfig(min_voxel_count=1000))


def test_validate_image_for_pipeline_rejects_a_decompression_bomb(tmp_path: Path) -> None:
    """Security control (Phase 14, docs/security.md "Malicious files"): a
    NIfTI header can declare an enormous shape while the file itself is tiny
    on disk (the header is readable without decompressing the body) - this
    must be rejected by the *cheap* header check, never reach the full-volume
    load a real attack is trying to trigger.

    Real, not simulated: writes a genuine ~70-byte gzip file whose header
    alone declares a 5000^3 (125 billion) voxel volume - the exact shape of
    a real decompression-bomb-style upload, without needing to actually
    allocate gigabytes to prove the point."""
    import gzip

    import nibabel as nib
    import numpy as np

    header = nib.Nifti1Header()
    header.set_data_shape((5000, 5000, 5000))
    header.set_data_dtype(np.float32)

    bomb = tmp_path / "bomb.nii.gz"
    with gzip.open(bomb, "wb") as f:
        f.write(header.binaryblock)
    assert bomb.stat().st_size < 1000  # tiny on disk - that's the whole attack

    with pytest.raises(DataValidationError, match="too large"):
        validate_image_for_pipeline(bomb, PreprocessingConfig())


def test_validate_image_for_pipeline_accepts_a_volume_within_the_max(tmp_path: Path) -> None:
    """The upper bound doesn't reject a real, merely-large-ish volume."""
    image = write_synthetic_nifti(tmp_path / "large.nii.gz", shape=(64, 64, 64))
    config = PreprocessingConfig(min_voxel_count=1000, max_voxel_count=1_000_000)
    metadata = validate_image_for_pipeline(image, config)
    assert metadata["num_dims"] == 3


def test_validate_affine_flags_non_finite_values() -> None:
    import numpy as np

    affine = np.eye(4)
    affine[0, 0] = np.nan
    assert validate_affine(affine) != []


def test_validate_affine_flags_a_degenerate_matrix() -> None:
    import numpy as np

    affine = np.zeros((4, 4))
    affine[3, 3] = 1.0  # valid homogeneous row, but the linear part is all zero
    assert validate_affine(affine) != []


def test_validate_affine_accepts_a_normal_matrix() -> None:
    import numpy as np

    assert validate_affine(np.eye(4)) == []
