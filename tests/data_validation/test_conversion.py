"""Tests for the optional DICOM -> NIfTI conversion path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")
nib = pytest.importorskip("nibabel")

from ad_mri.core.exceptions import DataValidationError  # noqa: E402
from ad_mri.data_pipeline.conversion import convert_dicom_series_to_nifti  # noqa: E402


def _write_synthetic_dicom_slice(
    path: Path, slice_index: int, shape: tuple[int, int] = (16, 16)
) -> None:
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.PatientID = "TEST001"
    ds.Modality = "MR"
    ds.SeriesInstanceUID = generate_uid()
    ds.StudyInstanceUID = generate_uid()
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.Rows, ds.Columns = shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelSpacing = [1.0, 1.0]
    ds.SliceThickness = 1.0
    ds.ImagePositionPatient = [0.0, 0.0, float(slice_index)]
    ds.InstanceNumber = slice_index + 1
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = 0.0

    pixel_array = (np.random.default_rng(slice_index).random(shape) * 1000).astype(np.uint16)
    ds.PixelData = pixel_array.tobytes()
    ds.save_as(str(path), enforce_file_format=True)


def _write_synthetic_dicom_series(
    directory: Path, num_slices: int = 5, shape: tuple[int, int] = (16, 16)
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(num_slices):
        _write_synthetic_dicom_slice(directory / f"slice_{i:03d}.dcm", i, shape)
    return directory


def test_convert_dicom_series_to_nifti_produces_a_valid_volume(tmp_path: Path) -> None:
    dicom_dir = _write_synthetic_dicom_series(tmp_path / "series", num_slices=5, shape=(16, 16))
    output_path = tmp_path / "converted.nii.gz"

    result_path = convert_dicom_series_to_nifti(dicom_dir, output_path)

    assert result_path == output_path
    assert output_path.exists()
    image = nib.load(str(output_path))
    assert image.shape == (16, 16, 5)


def test_convert_dicom_series_to_nifti_raises_on_empty_directory(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(DataValidationError):
        convert_dicom_series_to_nifti(empty_dir, tmp_path / "out.nii.gz")
