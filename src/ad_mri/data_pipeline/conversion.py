"""DICOM to NIfTI conversion (optional DICOM support).

`pydicom`/`nibabel`/`numpy` are training-phase dependencies (requirements/train.txt),
imported lazily so this module stays importable without them (e.g. by the API image).

This is a deliberately simple, single-series converter: it assumes one axial
series per directory, sorts slices by `ImagePositionPatient` (falling back to
`InstanceNumber`), and builds an axis-aligned affine from pixel spacing + slice
thickness. It does not handle multi-frame DICOM, oblique orientation cosines,
or multi-series studies. For those, a more robust dependency such as
`dicom2nifti` or SimpleITK should replace this in a later hardening pass — this
is a documented trade-off, not an oversight.
"""

from __future__ import annotations

from pathlib import Path

from ad_mri.core.exceptions import DataValidationError, PreprocessingError
from ad_mri.core.logging import get_logger

logger = get_logger(__name__)


def convert_dicom_series_to_nifti(dicom_dir: Path, output_path: Path) -> Path:
    """Convert a single-series DICOM directory into one NIfTI file.

    Raises:
        PreprocessingError: pydicom/nibabel/numpy are not installed.
        DataValidationError: no DICOM files were found, or the series could not
            be assembled into a consistent volume.
    """
    try:
        import numpy as np
        import pydicom
    except ImportError as exc:
        raise PreprocessingError(
            "DICOM conversion requires pydicom/numpy/nibabel " "(pip install -e '.[train]')."
        ) from exc
    import nibabel as nib

    dicom_paths = sorted(dicom_dir.glob("*.dcm"))
    if not dicom_paths:
        raise DataValidationError(f"No DICOM (*.dcm) files found in {dicom_dir}")

    try:
        datasets = [pydicom.dcmread(str(p)) for p in dicom_paths]
    except Exception as exc:
        raise DataValidationError(f"Failed to read DICOM series in {dicom_dir}: {exc}") from exc

    def _sort_key(ds: object) -> float:
        position = getattr(ds, "ImagePositionPatient", None)
        if position is not None:
            return float(position[2])
        return float(getattr(ds, "InstanceNumber", 0))

    datasets.sort(key=_sort_key)

    try:
        slices = [ds.pixel_array.astype(np.float32) for ds in datasets]
        volume = np.stack(slices, axis=-1)  # (rows, cols, num_slices)
    except Exception as exc:
        raise DataValidationError(f"Inconsistent slice geometry in {dicom_dir}: {exc}") from exc

    first = datasets[0]
    slope = float(getattr(first, "RescaleSlope", 1.0))
    intercept = float(getattr(first, "RescaleIntercept", 0.0))
    volume = volume * slope + intercept

    pixel_spacing = [float(x) for x in getattr(first, "PixelSpacing", [1.0, 1.0])]
    slice_thickness = float(getattr(first, "SliceThickness", 1.0))
    affine = np.diag([pixel_spacing[0], pixel_spacing[1], slice_thickness, 1.0]).astype(np.float64)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(volume, affine), str(output_path))

    logger.info(
        "dicom_series_converted",
        source=str(dicom_dir),
        output=str(output_path),
        num_slices=len(datasets),
        shape=list(volume.shape),
    )
    return output_path
