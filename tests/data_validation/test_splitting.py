from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ad_mri.core.exceptions import DataValidationError
from ad_mri.data_pipeline.dataset_config import DatasetConfig
from ad_mri.data_pipeline.ingestion import discover_subject_scans
from ad_mri.data_pipeline.manifest import build_manifest
from ad_mri.data_pipeline.splitting import assert_no_subject_leakage, create_subject_level_split
from ad_mri.data_pipeline.validation import validate_inventory


@pytest.fixture
def manifest_path(tmp_path: Path, synthetic_dataset: DatasetConfig) -> Path:
    inventory_df, _ = discover_subject_scans(synthetic_dataset)
    validated_df, _ = validate_inventory(inventory_df, synthetic_dataset)
    manifest_df = build_manifest(validated_df)
    path = tmp_path / "manifest.csv"
    manifest_df.to_csv(path, index=False)
    return path


def test_split_covers_every_subject_with_no_leakage(tmp_path: Path, manifest_path: Path) -> None:
    output_dir = tmp_path / "splits"
    paths = create_subject_level_split(
        manifest_path, output_dir, train_ratio=0.7, val_ratio=0.15, seed=42
    )

    assert set(paths) == {"train", "val", "test"}
    dfs = {name: pd.read_csv(path) for name, path in paths.items()}

    manifest_df = pd.read_csv(manifest_path)
    all_split_subjects = set()
    for df in dfs.values():
        all_split_subjects |= set(df["subject_id"])
    assert all_split_subjects == set(manifest_df["subject_id"])

    # No overlap between splits (the leakage guarantee).
    assert_no_subject_leakage(dfs)  # must not raise


def test_split_is_reproducible_given_the_same_seed(tmp_path: Path, manifest_path: Path) -> None:
    paths_a = create_subject_level_split(manifest_path, tmp_path / "a", seed=123)
    paths_b = create_subject_level_split(manifest_path, tmp_path / "b", seed=123)

    for name in ("train", "val", "test"):
        subjects_a = set(pd.read_csv(paths_a[name])["subject_id"])
        subjects_b = set(pd.read_csv(paths_b[name])["subject_id"])
        assert subjects_a == subjects_b


def test_split_differs_across_different_seeds(tmp_path: Path, manifest_path: Path) -> None:
    paths_a = create_subject_level_split(manifest_path, tmp_path / "a", seed=1)
    paths_b = create_subject_level_split(manifest_path, tmp_path / "b", seed=999)

    train_a = set(pd.read_csv(paths_a["train"])["subject_id"])
    train_b = set(pd.read_csv(paths_b["train"])["subject_id"])
    assert train_a != train_b


def test_assert_no_subject_leakage_raises_on_overlap() -> None:
    splits = {
        "train": pd.DataFrame({"subject_id": ["a", "b"]}),
        "val": pd.DataFrame({"subject_id": ["b", "c"]}),  # "b" leaks into both
    }
    with pytest.raises(DataValidationError):
        assert_no_subject_leakage(splits)


def test_create_subject_level_split_rejects_empty_manifest(tmp_path: Path) -> None:
    empty_manifest = tmp_path / "empty.csv"
    pd.DataFrame(columns=["subject_id", "label"]).to_csv(empty_manifest, index=False)
    with pytest.raises(DataValidationError):
        create_subject_level_split(empty_manifest, tmp_path / "splits")


def test_multi_scan_subjects_stay_together_across_split(tmp_path: Path) -> None:
    """A subject with multiple rows (e.g. two sessions) must land entirely in one split."""
    manifest_df = pd.DataFrame(
        {
            "subject_id": ["s1", "s1", "s2", "s2", "s3", "s3", "s4", "s4"],
            "label": ["CN", "CN", "AD", "AD", "CN", "CN", "AD", "AD"],
        }
    )
    manifest_path = tmp_path / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    paths = create_subject_level_split(
        manifest_path, tmp_path / "splits", train_ratio=0.5, val_ratio=0.25, seed=1
    )
    dfs = {name: pd.read_csv(path) for name, path in paths.items()}
    for df in dfs.values():
        # each subject appearing in this split must contribute exactly its 2 rows
        counts = df["subject_id"].value_counts()
        assert (counts == 2).all()

    assert_no_subject_leakage(dfs)
