"""Final dataset manifest: the versioned, single source of truth for training.

Filters the validated inventory down to QC-passed rows, orders/selects the
manifest columns, and reports class distribution (for imbalance handling —
see `class_balance.py`). See docs/TECHNICAL_SPECIFICATION.md §4, §17.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ad_mri.core.logging import get_logger
from ad_mri.data_pipeline.schema import MANIFEST_COLUMNS

logger = get_logger(__name__)

#: If max-class-count / min-class-count exceeds this, imbalance is logged as a warning.
IMBALANCE_WARNING_RATIO = 3.0


def build_manifest(validated_df: pd.DataFrame, *, include_failed: bool = False) -> pd.DataFrame:
    """Filter to QC-passed rows (unless `include_failed`) and select manifest columns."""
    df = validated_df if include_failed else validated_df[validated_df["qc_status"] == "passed"]
    df = df.copy()
    for column in MANIFEST_COLUMNS:
        if column not in df.columns:
            df[column] = None
    return df[MANIFEST_COLUMNS].reset_index(drop=True)


def write_manifest(manifest_df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(output_path, index=False)
    logger.info("manifest_written", path=str(output_path), rows=len(manifest_df))
    return output_path


def summarize_class_distribution(
    manifest_df: pd.DataFrame, label_column: str = "label"
) -> dict[str, int]:
    """Return {class_name: count}, and log a warning if the dataset is imbalanced."""
    counts = manifest_df[label_column].value_counts().to_dict()
    counts = {str(k): int(v) for k, v in counts.items()}

    if counts:
        max_count, min_count = max(counts.values()), min(counts.values())
        ratio = max_count / min_count if min_count > 0 else float("inf")
        if ratio > IMBALANCE_WARNING_RATIO:
            logger.warning(
                "class_imbalance_detected",
                class_counts=counts,
                imbalance_ratio=round(ratio, 2),
                threshold=IMBALANCE_WARNING_RATIO,
            )
        else:
            logger.info("class_distribution", class_counts=counts, imbalance_ratio=round(ratio, 2))
    return counts
