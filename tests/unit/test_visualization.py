"""Tests for the preprocessing sample visualization utility."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("nibabel")
pytest.importorskip("torch")
pytest.importorskip("monai")
pytest.importorskip("matplotlib")

from ad_mri.core.exceptions import DataValidationError  # noqa: E402
from ad_mri.datasets.preprocessing_config import PreprocessingConfig  # noqa: E402
from ad_mri.datasets.visualization import plot_preprocessing_sample  # noqa: E402
from tests.helpers import write_synthetic_nifti  # noqa: E402


def test_plot_preprocessing_sample_writes_a_png(tmp_path: Path) -> None:
    image = write_synthetic_nifti(tmp_path / "scan.nii.gz", shape=(32, 32, 32))
    config = PreprocessingConfig(target_shape=(16, 16, 16), min_voxel_count=1000)

    output_path = plot_preprocessing_sample(image, config=config)

    assert output_path.exists()
    assert output_path.name == "scan.preprocessing_preview.png"
    assert output_path.stat().st_size > 0


def test_plot_preprocessing_sample_respects_explicit_output_path(tmp_path: Path) -> None:
    image = write_synthetic_nifti(tmp_path / "scan.nii.gz", shape=(32, 32, 32))
    output_path = tmp_path / "custom" / "preview.png"
    config = PreprocessingConfig(target_shape=(16, 16, 16), min_voxel_count=1000)

    result = plot_preprocessing_sample(image, config=config, output_path=output_path)

    assert result == output_path
    assert output_path.exists()


def test_plot_preprocessing_sample_raises_for_invalid_image(tmp_path: Path) -> None:
    with pytest.raises(DataValidationError):
        plot_preprocessing_sample(tmp_path / "missing.nii.gz")
