"""Runtime, single-image validation for the preprocessing pipeline.

Fail-fast checks run immediately before the deterministic transform pipeline
touches an image — distinct from `data_pipeline.validation`, which validates
an entire dataset manifest at ingestion time. Both share the same low-level
shape-checking primitive (`data_pipeline.validation.check_shape`) so the rule
is defined once. See docs/TECHNICAL_SPECIFICATION.md §4, §6.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from ad_mri.core.exceptions import DataValidationError
from ad_mri.core.logging import get_logger
from ad_mri.data_pipeline.validation import check_shape, load_nifti_metadata
from ad_mri.datasets.preprocessing_config import PreprocessingConfig

logger = get_logger(__name__)

_DEGENERATE_DETERMINANT_THRESHOLD = 1e-8


def validate_affine(affine: Any) -> list[str]:
    """Sanity-check a NIfTI affine: finite values, non-degenerate (invertible).

    A degenerate/NaN affine means orientation and voxel spacing can't be
    trusted, which would make `Orientationd`/`Spacingd` (see `transforms.py`)
    silently produce a meaningless volume rather than fail — so this is
    checked explicitly, up front.
    """
    import numpy as np

    errors: list[str] = []
    if not np.all(np.isfinite(affine)):
        return ["affine matrix contains non-finite values (NaN/Inf)"]

    determinant = float(np.linalg.det(affine[:3, :3]))
    if abs(determinant) < _DEGENERATE_DETERMINANT_THRESHOLD:
        errors.append(
            f"affine matrix is degenerate (determinant={determinant:.2e}); "
            "orientation/spacing cannot be trusted"
        )
    return errors


def validate_image_for_pipeline(path: Path, config: PreprocessingConfig) -> dict[str, object]:
    """Fail-fast checks before running the preprocessing pipeline on `path`.

    Returns the extracted NIfTI header metadata (shape/spacing/dtype) on success.

    Raises:
        DataValidationError: the file is missing, not a readable NIfTI, has an
            invalid/degenerate affine, or falls outside the configured
            dimension/voxel-count bounds.
    """
    if not path.exists():
        raise DataValidationError(f"Image does not exist: {path}")

    try:
        metadata = load_nifti_metadata(path)
    except ValueError as exc:
        raise DataValidationError(str(exc)) from exc

    dims = (metadata.get("dim_x"), metadata.get("dim_y"), metadata.get("dim_z"))
    shape_errors = check_shape(
        metadata.get("num_dims"),
        dims,
        config.expected_dim_range,
        config.min_voxel_count,
        config.max_voxel_count,
    )

    import nibabel as nib

    # See the equivalent cast in data_pipeline.validation.load_nifti_metadata:
    # nib.load's declared return type is the abstract FileBasedImage, which
    # doesn't expose `.affine` in nibabel's stubs even though every concrete
    # NIfTI image class does.
    image = cast("nib.Nifti1Image", nib.load(str(path)))  # header-only; no voxel data read
    affine_errors = validate_affine(image.affine)

    errors = shape_errors + affine_errors
    if errors:
        raise DataValidationError(
            f"Image failed pre-preprocessing validation for {path}: {'; '.join(errors)}"
        )

    logger.debug("image_validated_for_pipeline", path=str(path), **metadata)
    return metadata
