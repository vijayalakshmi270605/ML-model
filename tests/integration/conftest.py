"""Shared fixtures for the API integration test suite: a real, tiny
checkpoint + preprocessing config, and a `TestClient` wired to serve it via
environment variables — so every test in this directory exercises the real
FastAPI app end-to-end (no mocks).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("monai")
pytest.importorskip("nibabel")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def checkpoint(tmp_path: Path) -> tuple[Path, Path]:
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
    preprocessing_path = save_preprocessing_config(
        preprocessing_config, tmp_path / "preprocessing.yaml"
    )

    model = build_model(model_config)
    checkpoint_path = tmp_path / "model.pt"
    save_checkpoint(
        model,
        model_config,
        checkpoint_path,
        extra={
            "class_names": ["AD", "CN"],
            "preprocessing_config_hash": preprocessing_config.compute_hash(),
        },
    )
    return checkpoint_path, preprocessing_path


@pytest.fixture
def configured_client(
    monkeypatch: pytest.MonkeyPatch, checkpoint: tuple[Path, Path], tmp_path: Path
) -> TestClient:
    checkpoint_path, preprocessing_path = checkpoint
    monkeypatch.setenv("INFERENCE_MODEL_PATH", str(checkpoint_path))
    monkeypatch.setenv("INFERENCE_PREPROCESSING_CONFIG_PATH", str(preprocessing_path))
    monkeypatch.setenv("INFERENCE_EXPLANATION_OUTPUT_DIR", str(tmp_path / "explanations"))

    from ad_mri.api.main import create_app

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def configured_client_with_audit_db(
    monkeypatch: pytest.MonkeyPatch, checkpoint: tuple[Path, Path], tmp_path: Path
) -> TestClient:
    """Same as `configured_client`, plus a real, schema-migrated SQLite
    database wired in as `DATABASE_URL` - for the handful of tests
    (`test_audit_integration.py`) that assert the API layer actually writes
    audit rows, not only that it survives when it can't."""
    checkpoint_path, preprocessing_path = checkpoint
    monkeypatch.setenv("INFERENCE_MODEL_PATH", str(checkpoint_path))
    monkeypatch.setenv("INFERENCE_PREPROCESSING_CONFIG_PATH", str(preprocessing_path))
    monkeypatch.setenv("INFERENCE_EXPLANATION_OUTPUT_DIR", str(tmp_path / "explanations"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'audit.db'}")

    from ad_mri.db.base import Base
    from ad_mri.db.session import get_engine

    Base.metadata.create_all(get_engine())

    from ad_mri.api.main import create_app

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
