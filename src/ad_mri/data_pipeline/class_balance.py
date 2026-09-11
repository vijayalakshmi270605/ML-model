"""Class-imbalance handling utilities.

Computes class weights from a manifest/split's label distribution, for use by
the training phase's loss function (`modeling.losses.build_loss`) and/or a
`WeightedRandomSampler`. See docs/TECHNICAL_SPECIFICATION.md §5, §15.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ad_mri.core.logging import get_logger

logger = get_logger(__name__)


def compute_class_weights(
    df: pd.DataFrame, label_column: str = "label", method: str = "inverse_frequency"
) -> dict[str, float]:
    """Compute per-class weights from `df[label_column]`'s distribution.

    `method="inverse_frequency"`: weight(c) = total_count / (num_classes * count(c)),
    normalized so the mean weight is 1.0 — the standard scikit-learn "balanced" formula.
    """
    if method != "inverse_frequency":
        raise ValueError(f"Unsupported class-weighting method: {method!r}")

    counts = df[label_column].value_counts()
    num_classes = len(counts)
    total = int(counts.sum())
    if num_classes == 0 or total == 0:
        return {}

    weights = {str(label): total / (num_classes * int(count)) for label, count in counts.items()}
    logger.info("class_weights_computed", weights=weights, method=method)
    return weights


def build_weighted_sampler(df: pd.DataFrame, label_column: str = "label") -> Any:
    """Build a `torch.utils.data.WeightedRandomSampler` from `df[label_column]`.

    `torch` is a training-phase dependency (requirements/train.txt), imported
    lazily so this module stays importable without it.
    """
    import torch

    weights_by_class = compute_class_weights(df, label_column=label_column)
    sample_weights: list[float] = [weights_by_class[str(label)] for label in df[label_column]]
    return torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
