"""Shared fixtures for the explainability test suite: a tiny real checkpoint
+ preprocessing config + manifest, so every test exercises the real LIME/SHAP
pipeline against a real (if tiny) trained CNN — no mocks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("lime")
pytest.importorskip("shap")

from tests.helpers import write_synthetic_manifest  # noqa: E402


@pytest.fixture
def manifest(tmp_path: Path) -> Path:
    return write_synthetic_manifest(tmp_path / "data", num_per_class=4, shape=(16, 16, 16))


@pytest.fixture
def checkpoint(tmp_path: Path) -> tuple[Path, Path]:
    """Returns `(checkpoint_path, preprocessing_config_path)` for a tiny, real
    (untrained — weights are whatever `build_model` initializes) 3D CNN. Not
    training a real classifier keeps every test here fast; explanation
    correctness does not depend on the model having converged."""
    from ad_mri.datasets.preprocessing_config import PreprocessingConfig, save_preprocessing_config
    from ad_mri.modeling.checkpoint import save_checkpoint
    from ad_mri.modeling.factory import build_model
    from ad_mri.modeling.model_config import ModelConfig

    model_config = ModelConfig(
        num_classes=2,
        spatial_dims=3,
        input_shape=(16, 16, 16),
        num_blocks=1,
        base_channels=4,
        fc_hidden_dim=8,
    )
    preprocessing_config = PreprocessingConfig(
        target_shape=(16, 16, 16), target_spacing=(1.0, 1.0, 1.0), min_voxel_count=1000
    )

    model = build_model(model_config)
    checkpoint_path = tmp_path / "model.pt"
    # Mirrors the provenance fields a real Trainer.fit() checkpoint carries
    # (see training.trainer.Trainer._build_checkpoint_extra) — a "bare"
    # checkpoint without these would make the service's mismatch-detection
    # checks untestable (nothing to compare against).
    save_checkpoint(
        model,
        model_config,
        checkpoint_path,
        extra={
            "class_names": ["AD", "CN"],
            "preprocessing_config_hash": preprocessing_config.compute_hash(),
            "preprocessing_version": preprocessing_config.version,
        },
    )

    preprocessing_path = save_preprocessing_config(
        preprocessing_config, tmp_path / "preprocessing.yaml"
    )
    return checkpoint_path, preprocessing_path
