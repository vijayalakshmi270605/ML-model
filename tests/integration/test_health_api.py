from __future__ import annotations

from fastapi.testclient import TestClient


def test_healthz_returns_ok(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_does_not_crash_without_a_database(client: TestClient) -> None:
    response = client.get("/readyz")
    assert response.status_code in (200, 503)


def test_metrics_endpoint_is_exposed(client: TestClient) -> None:
    client.get("/healthz")  # warm up a request so the counter has data
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"ad_mri_http_requests_total" in response.content
