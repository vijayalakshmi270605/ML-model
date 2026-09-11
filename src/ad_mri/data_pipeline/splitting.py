"""Subject-level stratified train/val/test split generation (leakage-safe).

Splitting is done on the set of unique subject IDs, never on individual rows —
if a subject has multiple scans, all of them land in the same split. This is
the single most important anti-leakage guarantee in the pipeline; see
docs/TECHNICAL_SPECIFICATION.md §4.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ad_mri.core.exceptions import DataValidationError
from ad_mri.core.logging import get_logger
from ad_mri.data_pipeline.schema import SPLIT_NAMES

logger = get_logger(__name__)


def assert_no_subject_leakage(
    splits: dict[str, pd.DataFrame], subject_column: str = "subject_id"
) -> None:
    """Raise DataValidationError if any subject_id appears in more than one split."""
    seen: dict[str, str] = {}
    for split_name, df in splits.items():
        for subject_id in df[subject_column].unique():
            if subject_id in seen and seen[subject_id] != split_name:
                raise DataValidationError(
                    f"Subject-level data leakage: {subject_id!r} appears in both "
                    f"{seen[subject_id]!r} and {split_name!r}."
                )
            seen[subject_id] = split_name


def create_subject_level_split(
    manifest_path: Path,
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    label_column: str = "label",
    subject_column: str = "subject_id",
    stratify: bool = True,
) -> dict[str, Path]:
    """Split a manifest into train/val/test CSVs at the subject level.

    Reproducible: identical `seed` (and identical input manifest) always
    produces an identical split. Raises DataValidationError if the manifest is
    empty or a leakage check fails.
    """
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(manifest_path)
    if df.empty:
        raise DataValidationError(f"Manifest is empty: {manifest_path}")
    if (
        not (0.0 < train_ratio < 1.0)
        or not (0.0 < val_ratio < 1.0)
        or train_ratio + val_ratio >= 1.0
    ):
        raise DataValidationError("train_ratio/val_ratio must be in (0,1) and sum to < 1.")

    subject_labels = df.groupby(subject_column)[label_column].first().reset_index()
    subject_ids = subject_labels[subject_column].to_numpy()
    labels = subject_labels[label_column].to_numpy()

    test_ratio = 1.0 - train_ratio - val_ratio

    def _split(ids: object, y: object, train_size: float, do_stratify: bool) -> tuple:
        stratify_arg = y if do_stratify else None
        try:
            return train_test_split(
                ids, y, train_size=train_size, random_state=seed, stratify=stratify_arg
            )
        except ValueError as exc:
            if do_stratify:
                logger.warning("stratified_split_failed_falling_back", error=str(exc))
                return train_test_split(ids, y, train_size=train_size, random_state=seed)
            raise

    train_ids, remaining_ids, _, remaining_labels = _split(
        subject_ids, labels, train_ratio, stratify
    )
    relative_val_size = val_ratio / (val_ratio + test_ratio)
    val_ids, test_ids, _, _ = _split(remaining_ids, remaining_labels, relative_val_size, stratify)

    split_id_map = {"train": set(train_ids), "val": set(val_ids), "test": set(test_ids)}
    split_dfs: dict[str, pd.DataFrame] = {
        name: df[df[subject_column].isin(ids)].reset_index(drop=True)
        for name, ids in split_id_map.items()
    }

    assert_no_subject_leakage(split_dfs, subject_column=subject_column)

    output_paths: dict[str, Path] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLIT_NAMES:
        output_path = output_dir / f"{name}.csv"
        split_dfs[name].to_csv(output_path, index=False)
        output_paths[name] = output_path

    logger.info(
        "split_complete",
        seed=seed,
        train_subjects=len(train_ids),
        val_subjects=len(val_ids),
        test_subjects=len(test_ids),
        train_rows=len(split_dfs["train"]),
        val_rows=len(split_dfs["val"]),
        test_rows=len(split_dfs["test"]),
    )
    return output_paths
