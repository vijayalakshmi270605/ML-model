"""Visualize the preprocessing pipeline's effect on a single MRI volume.

A debugging/QA utility, not part of the production inference path: renders
original-vs-preprocessed mid-slices (sagittal/coronal/axial) plus intensity
histograms, so a preprocessing change can be sanity-checked visually before
it's trusted. Runs the exact deterministic pipeline (`train=False` — no
augmentation) training/inference use, so what's rendered is what the model
sees. See docs/TECHNICAL_SPECIFICATION.md §4.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from ad_mri.core.logging import get_logger
from ad_mri.datasets.preprocessing_config import PreprocessingConfig

logger = get_logger(__name__)


def _mid_slices(volume: Any) -> tuple[Any, Any, Any]:
    """Return the mid-index 2D slice along each spatial axis (sagittal/coronal/axial).

    Accepts either a plain 3D volume or a channel-first 4D array (1, X, Y, Z).
    """
    import numpy as np

    array = np.asarray(volume)
    if array.ndim == 4:
        array = array[0]
    x, y, z = array.shape[-3:]
    return array[x // 2, :, :], array[:, y // 2, :], array[:, :, z // 2]


def _strip_nifti_suffix(name: str) -> str:
    for ext in (".nii.gz", ".nii"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def plot_preprocessing_sample(
    image_path: Path,
    config: PreprocessingConfig | None = None,
    output_path: Path | None = None,
) -> Path:
    """Render an original-vs-preprocessed comparison figure for one scan.

    Raises:
        DataValidationError: `image_path` fails pre-pipeline validation.
        PreprocessingError: the deterministic pipeline itself fails, or an
            enabled quality-control check fails.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless-safe: never assumes a display is available
    import matplotlib.pyplot as plt
    import nibabel as nib
    import numpy as np

    from ad_mri.datasets.transforms import apply_transforms, build_transforms
    from ad_mri.datasets.validation import validate_image_for_pipeline

    config = config or PreprocessingConfig()
    image_path = Path(image_path)
    output_path = output_path or image_path.with_name(
        f"{_strip_nifti_suffix(image_path.name)}.preprocessing_preview.png"
    )

    validate_image_for_pipeline(image_path, config)

    # See the equivalent cast in data_pipeline.validation.load_nifti_metadata.
    nifti_image = cast("nib.Nifti1Image", nib.load(str(image_path)))
    original = np.asanyarray(nifti_image.dataobj)

    transform = build_transforms(config, train=False)
    processed = apply_transforms(transform, {"image": str(image_path)})["image"]
    processed_array = (
        processed.detach().cpu().numpy() if hasattr(processed, "detach") else np.asarray(processed)
    )

    orig_slices = _mid_slices(original)
    proc_slices = _mid_slices(processed_array)
    axis_titles = ("Sagittal", "Coronal", "Axial")

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for col, (orig_slice, title) in enumerate(zip(orig_slices, axis_titles, strict=True)):
        axes[0, col].imshow(orig_slice.T, cmap="gray", origin="lower")
        axes[0, col].set_title(f"Original — {title}")
        axes[0, col].axis("off")

    for col, (proc_slice, title) in enumerate(zip(proc_slices, axis_titles, strict=True)):
        axes[1, col].imshow(proc_slice.T, cmap="gray", origin="lower")
        axes[1, col].set_title(f"Preprocessed — {title}")
        axes[1, col].axis("off")

    axes[0, 3].hist(original.ravel(), bins=100, color="steelblue")
    axes[0, 3].set_title("Original intensity histogram")
    axes[1, 3].hist(processed_array.ravel(), bins=100, color="darkorange")
    axes[1, 3].set_title("Preprocessed intensity histogram")

    fig.suptitle(
        f"{image_path.name}  |  preprocessing v{config.version} ({config.compute_hash()[:12]})"
    )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)

    logger.info(
        "preprocessing_preview_written", image_path=str(image_path), output_path=str(output_path)
    )
    return output_path
