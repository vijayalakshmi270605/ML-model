from __future__ import annotations

from pathlib import Path

import pandas as pd

from ad_mri.data_pipeline.dataset_config import DatasetConfig
from ad_mri.data_pipeline.ingestion import discover_subject_scans
from ad_mri.data_pipeline.manifest import (
    build_manifest,
    summarize_class_distribution,
    write_manifest,
)
from ad_mri.data_pipeline.schema import MANIFEST_COLUMNS
from ad_mri.data_pipeline.validation import validate_inventory


def test_build_manifest_filters_to_passed_rows_and_has_exact_schema(
    synthetic_dataset: DatasetConfig,
) -> None:
    inventory_df, _ = discover_subject_scans(synthetic_dataset)
    validated_df, _ = validate_inventory(inventory_df, synthetic_dataset)

    manifest_df = build_manifest(validated_df)

    assert list(manifest_df.columns) == MANIFEST_COLUMNS
    assert len(manifest_df) == 10  # 14 discovered - 4 failed
    assert (manifest_df["qc_status"] == "passed").all()
    assert set(manifest_df["subject_id"]) == {
        "sub-01",
        "sub-02",
        "sub-07",
        "sub-08",
        "sub-09",
        "sub-10",
        "sub-11",
        "sub-12",
        "sub-13",
        "sub-14",
    }


def test_build_manifest_can_include_failed_rows_for_audit(synthetic_dataset: DatasetConfig) -> None:
    inventory_df, _ = discover_subject_scans(synthetic_dataset)
    validated_df, _ = validate_inventory(inventory_df, synthetic_dataset)

    manifest_df = build_manifest(validated_df, include_failed=True)
    assert len(manifest_df) == 14


def test_write_manifest_round_trips_through_csv(
    tmp_path: Path, synthetic_dataset: DatasetConfig
) -> None:
    inventory_df, _ = discover_subject_scans(synthetic_dataset)
    validated_df, _ = validate_inventory(inventory_df, synthetic_dataset)
    manifest_df = build_manifest(validated_df)

    output_path = tmp_path / "manifest.csv"
    write_manifest(manifest_df, output_path)

    reloaded = pd.read_csv(output_path)
    assert len(reloaded) == len(manifest_df)
    assert list(reloaded.columns) == MANIFEST_COLUMNS


def test_summarize_class_distribution_counts_correctly() -> None:
    df = pd.DataFrame({"label": ["CN", "CN", "AD", "AD", "AD"]})
    counts = summarize_class_distribution(df)
    assert counts == {"CN": 2, "AD": 3}
