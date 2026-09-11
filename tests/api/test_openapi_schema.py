"""OpenAPI documentation smoke tests — real schema generation, no mocks.

Superset of the old `test_routers_smoke.py`: `/explain/{prediction_id}` and
`/models` were Phase-1-era 501 stubs; both are real endpoints now
(`/api/v1/explanations`, `/api/v1/models` — see tests/integration/) and are no
longer tested here as stubs.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_openapi_schema_is_generated(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert response.json()["info"]["title"] == "ad-mri"


def test_every_required_endpoint_is_present(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    paths = schema["paths"]
    assert set(paths["/api/v1/predictions"]) == {"post"}
    assert set(paths["/api/v1/explanations"]) == {"post"}
    assert "get" in paths["/api/v1/models"]
    assert "get" in paths["/api/v1/health"]
    assert "get" in paths["/api/v1/ready"]
    assert "get" in paths["/api/v1/version"]
    assert set(paths["/api/v1/monitoring/ground-truth"]) == {"post"}


def test_api_key_security_scheme_is_documented(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    schemes = schema["components"]["securitySchemes"]
    assert schemes["APIKeyHeader"] == {"type": "apiKey", "in": "header", "name": "X-API-Key"}

    predictions_security = schema["paths"]["/api/v1/predictions"]["post"].get("security")
    assert predictions_security == [{"APIKeyHeader": []}]
    ground_truth_security = schema["paths"]["/api/v1/monitoring/ground-truth"]["post"].get(
        "security"
    )
    assert ground_truth_security == [{"APIKeyHeader": []}]
    # read-only endpoints stay unauthenticated
    models_security = schema["paths"]["/api/v1/models"]["get"].get("security")
    assert not models_security


def test_tags_are_documented(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    tag_names = {tag["name"] for tag in schema.get("tags", [])}
    assert {"predictions", "explanations", "models", "monitoring", "meta", "health"} <= tag_names
    for tag in schema["tags"]:
        assert tag["description"]  # every declared tag has a real description
