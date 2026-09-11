from __future__ import annotations

from pathlib import Path

from ad_mri.data_pipeline.dataset_config import DatasetConfig
from ad_mri.data_pipeline.ingestion import discover_subject_scans, ingest_raw_scans


def test_discover_subject_scans_finds_all_subject_dirs(synthetic_dataset: DatasetConfig) -> None:
    inventory_df, report = discover_subject_scans(synthetic_dataset)

    assert report.subjects_discovered == 14  # sub-01 .. sub-14
    assert report.scans_discovered == 14
    assert len(inventory_df) == 14
    assert set(inventory_df.columns) == {
        "subject_id",
        "image_path",
        "format",
        "dataset_source",
        "modality",
    }
    assert (inventory_df["format"] == "nifti").all()
    assert (inventory_df["dataset_source"] == synthetic_dataset.dataset_source).all()


def test_discover_subject_scans_missing_root_dir_is_reported(
    tmp_path: Path, synthetic_dataset: DatasetConfig
) -> None:
    config = synthetic_dataset.model_copy(update={"root_dir": tmp_path / "does-not-exist"})
    inventory_df, report = discover_subject_scans(config)

    assert inventory_df.empty
    assert report.subjects_discovered == 0
    assert len(report.errors) == 1
    assert "does not exist" in report.errors[0].message


def test_ingest_raw_scans_writes_inventory_csv(
    tmp_path: Path, synthetic_dataset: DatasetConfig
) -> None:
    inventory_path = tmp_path / "inventory.csv"
    inventory_df, report = ingest_raw_scans(
        synthetic_dataset, interim_dir=tmp_path / "interim", inventory_output_path=inventory_path
    )

    assert inventory_path.exists()
    assert len(inventory_df) == report.scans_discovered == 14
