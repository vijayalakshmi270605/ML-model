"""Tests for datasets.transforms: the shared training/inference preprocessing pipeline.

Skipped automatically if torch/monai/nibabel (training-phase extras) aren't installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("nibabel")
torch = pytest.importorskip("torch")
pytest.importorskip("monai")

from ad_mri.core.exceptions import PreprocessingError  # noqa: E402
from ad_mri.datasets.preprocessing_config import PreprocessingConfig  # noqa: E402
from ad_mri.datasets.transforms import (  # noqa: E402
    apply_transforms,
    build_augmentation_transforms,
    build_preprocessing_transforms,
    build_transforms,
)
from tests.helpers import write_synthetic_nifti  # noqa: E402

_SMALL_SHAPE = (32, 32, 32)


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    return write_synthetic_nifti(tmp_path / "scan.nii.gz", shape=_SMALL_SHAPE)


def _config(**overrides: object) -> PreprocessingConfig:
    defaults: dict[str, object] = {"target_shape": (16, 16, 16), "target_spacing": (1.0, 1.0, 1.0)}
    defaults.update(overrides)
    return PreprocessingConfig(**defaults)  # type: ignore[arg-type]


# --- deterministic vs. augmentation separation --------------------------------


def test_eval_pipeline_has_no_augmentation_steps() -> None:
    config = _config()
    deterministic = build_preprocessing_transforms(config)
    eval_compose = build_transforms(config, train=False)
    assert len(eval_compose.transforms) == len(deterministic)


def test_train_pipeline_appends_augmentation_on_top_of_the_same_deterministic_steps() -> None:
    config = _config()
    deterministic = build_preprocessing_transforms(config)
    augmentation = build_augmentation_transforms(config)
    train_compose = build_transforms(config, train=True)

    assert len(train_compose.transforms) == len(deterministic) + len(augmentation)
    # The deterministic prefix is exactly the same step objects' types, in order.
    prefix = train_compose.transforms[: len(deterministic)]
    for step, expected in zip(prefix, deterministic, strict=True):
        assert type(step) is type(expected)


def test_augmentation_disabled_yields_no_extra_steps() -> None:
    config = _config(augmentation={"enabled": False})
    assert build_augmentation_transforms(config) == []
    assert len(build_transforms(config, train=True).transforms) == len(
        build_preprocessing_transforms(config)
    )


def test_flip_is_excluded_unless_explicitly_enabled() -> None:
    without_flip = build_augmentation_transforms(_config())
    with_flip = build_augmentation_transforms(_config(augmentation={"allow_flip": True}))
    assert len(with_flip) == len(without_flip) + 1


# --- determinism / reproducibility ---------------------------------------------


def test_eval_pipeline_is_fully_deterministic(sample_image: Path) -> None:
    config = _config()
    first = build_transforms(config, train=False)({"image": str(sample_image)})["image"]
    second = build_transforms(config, train=False)({"image": str(sample_image)})["image"]
    assert torch.equal(first, second)


def test_train_pipeline_is_reproducible_given_the_same_seed(sample_image: Path) -> None:
    config = _config(augmentation={"probability": 1.0})
    out_a = build_transforms(config, train=True, seed=123)({"image": str(sample_image)})["image"]
    out_b = build_transforms(config, train=True, seed=123)({"image": str(sample_image)})["image"]
    assert torch.equal(out_a, out_b)


def test_train_pipeline_differs_across_seeds(sample_image: Path) -> None:
    config = _config(augmentation={"probability": 1.0})
    out_a = build_transforms(config, train=True, seed=1)({"image": str(sample_image)})["image"]
    out_b = build_transforms(config, train=True, seed=2)({"image": str(sample_image)})["image"]
    assert not torch.equal(out_a, out_b)


# --- normalization modes -------------------------------------------------------


@pytest.mark.parametrize("mode", ["zscore_nonzero", "zscore", "percentile_clip", "min_max"])
def test_every_normalization_mode_produces_a_finite_output(sample_image: Path, mode: str) -> None:
    config = _config(normalization=mode)
    output = build_transforms(config, train=False)({"image": str(sample_image)})["image"]
    assert torch.isfinite(output).all()
    assert tuple(output.shape) == (1, 16, 16, 16)


# --- quality control ------------------------------------------------------------


def test_quality_control_raises_on_an_all_zero_volume(tmp_path: Path) -> None:
    empty_image = write_synthetic_nifti(tmp_path / "empty.nii.gz", shape=_SMALL_SHAPE)
    import nibabel as nib
    import numpy as np

    nib.save(nib.Nifti1Image(np.zeros(_SMALL_SHAPE, dtype=np.float32), np.eye(4)), str(empty_image))

    config = _config(qc_mode="raise")
    with pytest.raises(PreprocessingError):
        apply_transforms(build_transforms(config, train=False), {"image": str(empty_image)})


def test_quality_control_warn_mode_does_not_raise(tmp_path: Path) -> None:
    empty_image = tmp_path / "empty.nii.gz"
    import nibabel as nib
    import numpy as np

    nib.save(nib.Nifti1Image(np.zeros(_SMALL_SHAPE, dtype=np.float32), np.eye(4)), str(empty_image))

    config = _config(qc_mode="warn")
    output = build_transforms(config, train=False)({"image": str(empty_image)})["image"]
    assert tuple(output.shape) == (1, 16, 16, 16)


def test_quality_control_can_be_disabled(tmp_path: Path) -> None:
    empty_image = tmp_path / "empty.nii.gz"
    import nibabel as nib
    import numpy as np

    nib.save(nib.Nifti1Image(np.zeros(_SMALL_SHAPE, dtype=np.float32), np.eye(4)), str(empty_image))

    config = _config(enable_quality_control=False)
    output = build_transforms(config, train=False)({"image": str(empty_image)})["image"]
    assert tuple(output.shape) == (1, 16, 16, 16)


# --- safe loading ----------------------------------------------------------------


def test_safe_load_wraps_a_missing_file_in_preprocessing_error(tmp_path: Path) -> None:
    config = _config()
    transform = build_transforms(config, train=False)
    with pytest.raises(PreprocessingError):
        apply_transforms(transform, {"image": str(tmp_path / "does-not-exist.nii.gz")})


def test_safe_load_wraps_a_corrupted_file_in_preprocessing_error(tmp_path: Path) -> None:
    corrupted = tmp_path / "corrupted.nii.gz"
    corrupted.write_bytes(b"not a real nifti file")

    config = _config()
    transform = build_transforms(config, train=False)
    with pytest.raises(PreprocessingError):
        apply_transforms(transform, {"image": str(corrupted)})
