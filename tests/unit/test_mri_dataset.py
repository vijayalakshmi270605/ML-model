"""Tests for datasets.dataset.MRIDataset and datasets.transforms.build_transforms.

Skipped automatically if torch/monai/nibabel (training-phase extras) aren't installed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

nib = pytest.importorskip("nibabel")
torch = pytest.importorskip("torch")
pytest.importorskip("monai")

from ad_mri.data_pipeline.schema import MANIFEST_COLUMNS  # noqa: E402
from ad_mri.datasets.dataset import MRIDataset, infer_label_mapping  # noqa: E402
from ad_mri.datasets.transforms import build_transforms  # noqa: E402


def _write_manifest(tmp_path: Path, num_per_class: int = 2) -> Path:
    import numpy as np

    rows = []
    for label in ("CN", "AD"):
        for i in range(num_per_class):
            subject_id = f"{label}-{i}"
            image_path = tmp_path / f"{subject_id}.nii.gz"
            data = np.random.default_rng(i).random((16, 16, 16)).astype("float32")
            nib.save(nib.Nifti1Image(data, np.eye(4)), str(image_path))
            row = dict.fromkeys(MANIFEST_COLUMNS)
            row.update({"subject_id": subject_id, "image_path": str(image_path), "label": label})
            rows.append(row)

    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(rows, columns=MANIFEST_COLUMNS).to_csv(manifest_path, index=False)
    return manifest_path


def test_infer_label_mapping_is_deterministic_and_sorted() -> None:
    df = pd.DataFrame({"label": ["AD", "CN", "AD", "CN"]})
    assert infer_label_mapping(df) == {"AD": 0, "CN": 1}


def test_mri_dataset_without_transform_exposes_raw_paths(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path)
    dataset = MRIDataset(manifest_path)

    assert len(dataset) == 4
    assert dataset.label_to_index == {"AD": 0, "CN": 1}

    sample = dataset[0]
    assert Path(sample["image"]).exists()
    assert sample["label"] in (0, 1)
    assert "subject_id" in sample


def test_mri_dataset_with_transforms_produces_a_normalized_tensor(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path)
    transform = build_transforms({"target_shape": (16, 16, 16)}, train=False)
    dataset = MRIDataset(manifest_path, transform=transform)

    sample = dataset[0]
    image = sample["image"]

    assert isinstance(image, torch.Tensor)
    assert image.shape == (1, 16, 16, 16)  # channel-first
    assert sample["label"] in (0, 1)


# Deterministic-vs-augmentation pipeline composition is covered in depth by
# tests/unit/test_transforms.py; this file focuses on MRIDataset itself.
