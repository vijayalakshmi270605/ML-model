"""Deterministic sampling of preprocessed volumes from a manifest — used for
SHAP's background/reference set and for the multi-sample batch
`compute_global_shap_importance` aggregates over.

Reuses the identical deterministic preprocessing pipeline
(`datasets.transforms.build_transforms(..., train=False)`) training/inference/
evaluation already use — no augmentation, no second preprocessing
implementation (requirement: "keep preprocessing identical to inference
preprocessing"). See docs/explainability.md.

`torch`/`pandas` are training-phase dependencies (requirements/train.txt),
imported lazily so this module stays importable without them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ad_mri.core.exceptions import DataValidationError
from ad_mri.core.logging import get_logger
from ad_mri.datasets.preprocessing_config import PreprocessingConfig

logger = get_logger(__name__)


def load_volume_batch(
    manifest_path: Path,
    preprocessing_config: PreprocessingConfig,
    n_samples: int,
    seed: int = 42,
) -> tuple[Any, list[str]]:
    """Deterministically draw `n_samples` rows from `manifest_path`, preprocess
    each (`train=False`: no augmentation), and stack into one batch tensor.

    Reproducible: identical `manifest_path` + `n_samples` + `seed` always draws
    the same rows in the same order.

    Raises:
        DataValidationError: the manifest has fewer than `n_samples` rows.
    """
    import pandas as pd
    import torch

    from ad_mri.datasets.transforms import apply_transforms, build_transforms

    df = pd.read_csv(manifest_path)
    if len(df) < n_samples:
        raise DataValidationError(
            f"{manifest_path} has {len(df)} row(s) but {n_samples} sample(s) were requested."
        )
    sampled = df.sample(n=n_samples, random_state=seed).reset_index(drop=True)

    transform = build_transforms(preprocessing_config, train=False)
    volumes = []
    subject_ids = []
    for _, row in sampled.iterrows():
        sample = apply_transforms(transform, {"image": str(row["image_path"])})
        volumes.append(sample["image"])
        subject_ids.append(str(row["subject_id"]))

    batch = torch.stack(volumes)
    logger.info(
        "volume_batch_sampled", manifest_path=str(manifest_path), n_samples=n_samples, seed=seed
    )
    return batch, subject_ids
