"""Integration tests for GET /api/v1/models — a real FastAPI TestClient
against a real, tiny checkpoint (no mocks)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def test_models_lists_the_configured_checkpoint(
    configured_client: TestClient, checkpoint: tuple[Path, Path]
) -> None:
    from ad_mri.modeling.checkpoint import load_checkpoint

    checkpoint_path, _preprocessing_path = checkpoint
    _model, model_config, _metadata = load_checkpoint(checkpoint_path)

    response = configured_client.get("/api/v1/models")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    entry = body[0]
    assert entry["name"] == model_config.architecture
    assert entry["version"] == model_config.version
    assert entry["model_config_hash"] == model_config.compute_hash()
    assert entry["stage"] == "serving"


def test_models_returns_an_empty_list_when_not_configured(client: TestClient) -> None:
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    assert response.json() == []


def test_models_requires_no_api_key_even_when_configured(
    monkeypatch: pytest.MonkeyPatch, checkpoint: tuple[Path, Path]
) -> None:
    checkpoint_path, preprocessing_path = checkpoint
    monkeypatch.setenv("INFERENCE_MODEL_PATH", str(checkpoint_path))
    monkeypatch.setenv("INFERENCE_PREPROCESSING_CONFIG_PATH", str(preprocessing_path))
    monkeypatch.setenv("API_KEY_REQUIRED", "true")
    monkeypatch.setenv("API_KEY", "the-real-key")

    from ad_mri.api.main import create_app

    with TestClient(create_app()) as client:
        response = client.get("/api/v1/models")  # no X-API-Key header at all
        assert response.status_code == 200
