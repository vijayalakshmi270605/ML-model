"""Tests for data_pipeline.cross_validation: fold generation is subject-level,
leakage-safe, reproducible, and never touches a test set (it has no parameter
that could accept one)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("sklearn")

from ad_mri.core.exceptions import DataValidationError  # noqa: E402
from ad_mri.data_pipeline.cross_validation import (  # noqa: E402
    create_cross_validation_folds,
    discover_cv_folds,
    load_development_pool,
)
from ad_mri.data_pipeline.splitting import assert_no_subject_leakage  # noqa: E402


def _write_manifest(path: Path, subject_labels: dict[str, str]) -> Path:
    rows = [{"subject_id": sid, "label": label} for sid, label in subject_labels.items()]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def dev_pool_manifests(tmp_path: Path) -> tuple[Path, Path]:
    # 10 subjects per class across train+val — enough for 5-fold stratified CV.
    train_labels = {f"CN-{i}": "CN" for i in range(7)} | {f"AD-{i}": "AD" for i in range(7)}
    val_labels = {f"CN-{i}": "CN" for i in range(7, 10)} | {f"AD-{i}": "AD" for i in range(7, 10)}
    train_path = _write_manifest(tmp_path / "train.csv", train_labels)
    val_path = _write_manifest(tmp_path / "val.csv", val_labels)
    return train_path, val_path


def test_load_development_pool_combines_manifests(dev_pool_manifests: tuple[Path, Path]) -> None:
    train_path, val_path = dev_pool_manifests
    pool = load_development_pool([train_path, val_path])
    assert len(pool) == 20
    assert pool["subject_id"].nunique() == 20


def test_load_development_pool_rejects_overlapping_subjects(tmp_path: Path) -> None:
    shared = {"subj-1": "CN", "subj-2": "AD"}
    a = _write_manifest(tmp_path / "a.csv", shared)
    b = _write_manifest(tmp_path / "b.csv", shared)  # same subjects in both
    with pytest.raises(DataValidationError):
        load_development_pool([a, b])


def test_folds_partition_all_subjects_with_no_leakage(
    tmp_path: Path, dev_pool_manifests: tuple[Path, Path]
) -> None:
    train_path, val_path = dev_pool_manifests
    folds = create_cross_validation_folds(
        [train_path, val_path], tmp_path / "cv", n_folds=5, seed=42
    )

    assert len(folds) == 5
    all_val_subjects: set[str] = set()
    for fold in folds:
        train_df = pd.read_csv(fold["train"])
        val_df = pd.read_csv(fold["val"])
        assert_no_subject_leakage({"train": train_df, "val": val_df})  # must not raise
        all_val_subjects |= set(val_df["subject_id"])

    # every subject is the held-out val fold exactly once across all folds
    assert all_val_subjects == {f"CN-{i}" for i in range(10)} | {f"AD-{i}" for i in range(10)}


def test_folds_are_reproducible_given_the_same_seed(
    tmp_path: Path, dev_pool_manifests: tuple[Path, Path]
) -> None:
    train_path, val_path = dev_pool_manifests
    folds_a = create_cross_validation_folds(
        [train_path, val_path], tmp_path / "cv_a", n_folds=5, seed=7
    )
    folds_b = create_cross_validation_folds(
        [train_path, val_path], tmp_path / "cv_b", n_folds=5, seed=7
    )
    for fold_a, fold_b in zip(folds_a, folds_b, strict=True):
        val_a = set(pd.read_csv(fold_a["val"])["subject_id"])
        val_b = set(pd.read_csv(fold_b["val"])["subject_id"])
        assert val_a == val_b


def test_folds_differ_across_different_seeds(
    tmp_path: Path, dev_pool_manifests: tuple[Path, Path]
) -> None:
    train_path, val_path = dev_pool_manifests
    folds_a = create_cross_validation_folds(
        [train_path, val_path], tmp_path / "cv_a", n_folds=5, seed=1
    )
    folds_b = create_cross_validation_folds(
        [train_path, val_path], tmp_path / "cv_b", n_folds=5, seed=999
    )
    val_a = set(pd.read_csv(folds_a[0]["val"])["subject_id"])
    val_b = set(pd.read_csv(folds_b[0]["val"])["subject_id"])
    assert val_a != val_b


def test_stratification_falls_back_gracefully_when_not_statistically_appropriate(
    tmp_path: Path,
) -> None:
    # Only 2 subjects of "AD" but requesting 5 folds — StratifiedKFold cannot
    # place >=1 AD subject in every fold, so this must fall back to plain
    # KFold rather than raising.
    labels = {f"CN-{i}": "CN" for i in range(10)} | {"AD-0": "AD", "AD-1": "AD"}
    manifest = _write_manifest(tmp_path / "manifest.csv", labels)
    folds = create_cross_validation_folds([manifest], tmp_path / "cv", n_folds=5, seed=42)
    assert len(folds) == 5


def test_rejects_fewer_subjects_than_folds(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "manifest.csv", {"s1": "CN", "s2": "AD"})
    with pytest.raises(DataValidationError):
        create_cross_validation_folds([manifest], tmp_path / "cv", n_folds=5)


def test_discover_cv_folds_rediscovers_what_was_written(
    tmp_path: Path, dev_pool_manifests: tuple[Path, Path]
) -> None:
    train_path, val_path = dev_pool_manifests
    written = create_cross_validation_folds(
        [train_path, val_path], tmp_path / "cv", n_folds=3, seed=42
    )
    rediscovered = discover_cv_folds(tmp_path / "cv")
    assert len(rediscovered) == len(written) == 3
    for w, r in zip(written, rediscovered, strict=True):
        assert w["train"] == r["train"]
        assert w["val"] == r["val"]


def test_discover_cv_folds_raises_when_none_exist(tmp_path: Path) -> None:
    with pytest.raises(DataValidationError):
        discover_cv_folds(tmp_path / "does-not-exist")
