from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ad_mri.core.exceptions import ConfigurationError
from ad_mri.data_pipeline.dataset_config import DatasetConfig, load_dataset_config


def _base_kwargs(tmp_path: Path) -> dict:
    return {
        "name": "example",
        "dataset_source": "unit-test",
        "root_dir": tmp_path / "raw",
        "labels_csv": tmp_path / "labels.csv",
        "label_map": {"Nondemented": "CN", "Demented": "AD"},
    }


def test_dataset_config_computes_test_ratio(tmp_path: Path) -> None:
    config = DatasetConfig(**_base_kwargs(tmp_path), train_ratio=0.7, val_ratio=0.15)
    assert config.test_ratio == pytest.approx(0.15)


def test_dataset_config_rejects_ratios_that_leave_no_room_for_test(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        DatasetConfig(**_base_kwargs(tmp_path), train_ratio=0.8, val_ratio=0.3)


def test_load_dataset_config_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        load_dataset_config(tmp_path / "does-not-exist.yaml")


def test_load_dataset_config_reads_valid_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "dataset.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "example",
                "dataset_source": "unit-test",
                "root_dir": str(tmp_path / "raw"),
                "labels_csv": str(tmp_path / "labels.csv"),
                "label_map": {"Nondemented": "CN", "Demented": "AD"},
            }
        ),
        encoding="utf-8",
    )
    config = load_dataset_config(config_path)
    assert config.name == "example"
    assert config.label_map == {"Nondemented": "CN", "Demented": "AD"}


def test_the_shipped_example_dataset_config_is_valid() -> None:
    """configs/data/dataset_example.yaml is documentation-by-example — it must stay valid."""
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "configs" / "data" / "dataset_example.yaml"
    config = load_dataset_config(config_path)
    assert config.name == "example_mri_dataset"
