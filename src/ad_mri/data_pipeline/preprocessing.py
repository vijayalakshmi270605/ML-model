"""Shared MRI preprocessing pipeline.

This pipeline is used identically at training time and inference time to
eliminate train/serve skew — see docs/TECHNICAL_SPECIFICATION.md §4, §6.
Steps: skull-stripping, template registration, bias-field correction,
intensity normalization, resampling.
"""

from __future__ import annotations

from pathlib import Path


def skull_strip(image_path: Path, output_path: Path) -> Path:
    """Raises: NotImplementedError — implemented in the data-pipeline phase."""
    raise NotImplementedError("Preprocessing is implemented in the data-pipeline phase.")


def register_to_template(image_path: Path, template_path: Path, output_path: Path) -> Path:
    """Raises: NotImplementedError — implemented in the data-pipeline phase."""
    raise NotImplementedError("Preprocessing is implemented in the data-pipeline phase.")


def normalize_intensity(image_path: Path, output_path: Path) -> Path:
    """Raises: NotImplementedError — implemented in the data-pipeline phase."""
    raise NotImplementedError("Preprocessing is implemented in the data-pipeline phase.")


def resample(
    image_path: Path, output_path: Path, target_spacing: tuple[float, float, float]
) -> Path:
    """Raises: NotImplementedError — implemented in the data-pipeline phase."""
    raise NotImplementedError("Preprocessing is implemented in the data-pipeline phase.")
