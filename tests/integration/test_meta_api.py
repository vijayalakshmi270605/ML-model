"""Integration tests for the versioned meta endpoints
(GET /api/v1/health, /ready, /version) — a real FastAPI TestClient (no mocks)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ad_mri import __version__


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_does_not_require_inference_when_not_configured(client: TestClient) -> None:
    """A deployment that never set INFERENCE_MODEL_PATH must not be reported
    "not ready" for a dependency it was never asked to have."""
    response = client.get("/api/v1/ready")
    assert response.status_code in (200, 503)  # 503 only possible via the DB check
    body = response.json()
    assert body["inference_engine"] is None


def test_ready_reports_inference_engine_when_configured(configured_client: TestClient) -> None:
    """Only the inference-engine field is asserted here: whether /ready is
    fully 200 also depends on database reachability (test_health_api.py's
    test_readyz_does_not_crash_without_a_database already covers that this
    suite runs with no real PostgreSQL available), which is orthogonal to
    what this test is checking."""
    response = configured_client.get("/api/v1/ready")
    assert response.status_code in (200, 503)
    body = response.json()
    assert body["inference_engine"] == "reachable"


def test_ready_reports_unreachable_and_503_for_a_broken_checkpoint_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("INFERENCE_MODEL_PATH", str(tmp_path / "does-not-exist.pt"))

    from ad_mri.api.main import create_app

    with TestClient(create_app()) as client:
        response = client.get("/api/v1/ready")
        assert response.status_code == 503
        body = response.json()
        assert body["inference_engine"] == "unreachable"


def test_version_reports_the_real_app_version(client: TestClient) -> None:
    response = client.get("/api/v1/version")
    assert response.status_code == 200
    body = response.json()
    assert body["version"] == __version__
    assert body["api_version"] == "v1"


def test_unversioned_and_versioned_health_probes_agree(client: TestClient) -> None:
    assert client.get("/healthz").json() == client.get("/api/v1/health").json()
