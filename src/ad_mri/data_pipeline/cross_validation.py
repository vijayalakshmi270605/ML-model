"""K-fold subject-level cross-validation fold generation.

Operates on the **development pool** (train + val manifests from
`data_pipeline.splitting.create_subject_level_split`) — the held-out test
manifest is never an input here, structurally enforcing "do not optimize
against the test set" (this module has no parameter that could accept it).
See docs/TECHNICAL_SPECIFICATION.md §4, and docs/model_selection.md.

Folding is done on the set of unique subject IDs (never on individual rows),
using the same leakage guarantee as the holdout split
(`splitting.assert_no_subject_leakage`) — a subject with multiple scans always
lands entirely in one fold.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ad_mri.core.exceptions import DataValidationError
from ad_mri.core.logging import get_logger
from ad_mri.data_pipeline.splitting import assert_no_subject_leakage

logger = get_logger(__name__)


def load_development_pool(
    manifest_paths: list[Path], subject_column: str = "subject_id"
) -> pd.DataFrame:
    """Concatenate one or more manifests (typically train.csv + val.csv) into
    one development pool for cross-validation.

    Raises:
        DataValidationError: the same subject appears in more than one input
            manifest (would otherwise silently duplicate/corrupt fold assignment).
    """
    frames = [pd.read_csv(path) for path in manifest_paths]
    if not frames:
        raise DataValidationError("No manifest paths given to load_development_pool.")

    subject_sets = [set(frame[subject_column]) for frame in frames]
    for i, subjects_i in enumerate(subject_sets):
        for j, subjects_j in enumerate(subject_sets):
            if i < j and (overlap := subjects_i & subjects_j):
                raise DataValidationError(
                    f"Subject(s) {sorted(overlap)} appear in both {manifest_paths[i]} "
                    f"and {manifest_paths[j]} — cannot build a leakage-free development pool."
                )

    pool = pd.concat(frames, ignore_index=True)
    logger.info(
        "development_pool_loaded",
        manifests=[str(p) for p in manifest_paths],
        rows=len(pool),
        subjects=pool[subject_column].nunique(),
    )
    return pool


def create_cross_validation_folds(
    manifest_paths: list[Path],
    output_dir: Path,
    n_folds: int = 5,
    seed: int = 42,
    label_column: str = "label",
    subject_column: str = "subject_id",
    stratify: bool = True,
) -> list[dict[str, Path]]:
    """Build `n_folds` subject-level (stratified where possible) folds from the
    development pool and write each fold's train/val manifest CSVs.

    Reproducible: identical `seed` + identical input manifests always produce
    identical folds.

    Returns:
        A list of `{"train": Path, "val": Path}`, one per fold, in fold order.

    Raises:
        DataValidationError: the pool is empty, has fewer subjects than
            `n_folds`, or a leakage check fails.
    """
    from sklearn.model_selection import KFold, StratifiedKFold

    pool = load_development_pool(manifest_paths, subject_column=subject_column)
    if pool.empty:
        raise DataValidationError("Development pool is empty — no folds can be created.")

    subject_labels = pool.groupby(subject_column)[label_column].first().reset_index()
    subject_ids = subject_labels[subject_column].to_numpy()
    labels = subject_labels[label_column].to_numpy()

    if len(subject_ids) < n_folds:
        raise DataValidationError(
            f"Only {len(subject_ids)} subject(s) available but n_folds={n_folds}; "
            "reduce n_folds or grow the development pool."
        )

    class_counts = pd.Series(labels).value_counts()
    can_stratify = bool(stratify and class_counts.min() >= n_folds)
    if stratify and not can_stratify:
        logger.warning(
            "stratified_cv_not_statistically_appropriate_falling_back",
            min_class_count=int(class_counts.min()),
            n_folds=n_folds,
            reason="at least one class has fewer subjects than n_folds",
        )

    if can_stratify:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        split_iterator = splitter.split(subject_ids, labels)
    else:
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        split_iterator = splitter.split(subject_ids)

    output_dir.mkdir(parents=True, exist_ok=True)
    folds: list[dict[str, Path]] = []
    for fold_index, (train_idx, val_idx) in enumerate(split_iterator):
        train_subjects = set(subject_ids[train_idx])
        val_subjects = set(subject_ids[val_idx])

        fold_train_df = pool[pool[subject_column].isin(train_subjects)].reset_index(drop=True)
        fold_val_df = pool[pool[subject_column].isin(val_subjects)].reset_index(drop=True)
        assert_no_subject_leakage(
            {"train": fold_train_df, "val": fold_val_df}, subject_column=subject_column
        )

        fold_dir = output_dir / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_path = fold_dir / "train.csv"
        val_path = fold_dir / "val.csv"
        fold_train_df.to_csv(train_path, index=False)
        fold_val_df.to_csv(val_path, index=False)
        folds.append({"train": train_path, "val": val_path})

        logger.info(
            "cv_fold_written",
            fold=fold_index,
            train_subjects=len(train_subjects),
            val_subjects=len(val_subjects),
            stratified=can_stratify,
        )

    logger.info("cross_validation_folds_complete", n_folds=n_folds, output_dir=str(output_dir))
    return folds


def discover_cv_folds(output_dir: Path) -> list[dict[str, Path]]:
    """Rediscover the `{"train": Path, "val": Path}` list for folds already
    written by `create_cross_validation_folds` — lets a later command (CV
    training, HPO) reuse a fold layout without recomputing it.

    Raises:
        DataValidationError: no `fold_*` directories (or an incomplete one)
            are found under `output_dir`.
    """
    fold_dirs = sorted(output_dir.glob("fold_*"), key=lambda p: int(p.name.split("_")[1]))
    if not fold_dirs:
        raise DataValidationError(
            f"No fold_* directories found under {output_dir}. "
            "Run cross-validation fold generation first."
        )

    folds: list[dict[str, Path]] = []
    for fold_dir in fold_dirs:
        train_path, val_path = fold_dir / "train.csv", fold_dir / "val.csv"
        if not train_path.exists() or not val_path.exists():
            raise DataValidationError(f"{fold_dir} is missing train.csv and/or val.csv.")
        folds.append({"train": train_path, "val": val_path})
    return folds
