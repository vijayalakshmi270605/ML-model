"""Integration tests for POST /api/v1/predictions — a real FastAPI TestClient
against a real, tiny checkpoint (no mocks)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.helpers import write_synthetic_nifti


def _upload(image_path: Path) -> dict[str, tuple[str, bytes, str]]:
    return {"file": (image_path.name, image_path.read_bytes(), "application/octet-stream")}


def test_predictions_returns_201_with_a_valid_response(
    configured_client: TestClient, tmp_path: Path
) -> None:
    image = write_synthetic_nifti(tmp_path / "scan.nii.gz", shape=(32, 32, 32))
    response = configured_client.post("/api/v1/predictions", files=_upload(image))

    assert response.status_code == 201
    body = response.json()
    assert set(body) == {
        "request_id",
        "model_version",
        "preprocessing_version",
        "prediction",
        "confidence",
        "probabilities",
        "processing_time_ms",
        "explanation_available",
        "explanation_id",
        "timestamp",
        "disclaimer",
    }
    # Phase 14 ML-safety control: every response carries this inline, not
    # only in documentation - see core.disclaimers.
    assert "not a clinical diagnosis" in body["disclaimer"]
    assert body["prediction"] in {"AD", "CN"}
    assert set(body["probabilities"]) == {"AD", "CN"}
    assert body["explanation_available"] is False
    assert body["explanation_id"] is None

    # request_id correlates with the X-Request-ID response header
    assert response.headers["X-Request-ID"] == body["request_id"]


def test_predictions_response_never_contains_the_uploaded_filename(
    configured_client: TestClient, tmp_path: Path
) -> None:
    image = write_synthetic_nifti(
        tmp_path / "sensitive_patient_name_1957.nii.gz", shape=(32, 32, 32)
    )
    response = configured_client.post("/api/v1/predictions", files=_upload(image))
    assert response.status_code == 201
    assert "sensitive_patient_name" not in response.text


def test_predictions_rejects_an_invalid_upload_with_422(configured_client: TestClient) -> None:
    response = configured_client.post(
        "/api/v1/predictions",
        files={
            "file": ("garbage.nii.gz", b"this is not a real nifti file", "application/octet-stream")
        },
    )
    assert response.status_code == 422
    body = response.json()
    assert body["error_code"] == "VALIDATION_FAILED"
    assert "garbage.nii.gz" not in body["detail"]


def test_predictions_rejects_an_oversized_upload_with_413(
    monkeypatch: pytest.MonkeyPatch, checkpoint: tuple[Path, Path], tmp_path: Path
) -> None:
    checkpoint_path, preprocessing_path = checkpoint
    monkeypatch.setenv("INFERENCE_MODEL_PATH", str(checkpoint_path))
    monkeypatch.setenv("INFERENCE_PREPROCESSING_CONFIG_PATH", str(preprocessing_path))
    monkeypatch.setenv("MAX_UPLOAD_SIZE_MB", "1")  # 1 MB cap for this test only

    from ad_mri.api.main import create_app

    with TestClient(create_app()) as client:
        oversized_payload = b"0" * (2 * 1024 * 1024)  # 2 MB, over the 1 MB cap
        response = client.post(
            "/api/v1/predictions",
            files={"file": ("scan.nii.gz", oversized_payload, "application/octet-stream")},
        )
        assert response.status_code == 413


def test_predictions_returns_503_when_engine_not_configured(
    client: TestClient, tmp_path: Path
) -> None:
    image = write_synthetic_nifti(tmp_path / "scan.nii.gz", shape=(32, 32, 32))
    response = client.post("/api/v1/predictions", files=_upload(image))
    assert response.status_code == 503


def test_predictions_requires_an_api_key_when_configured(
    monkeypatch: pytest.MonkeyPatch, checkpoint: tuple[Path, Path], tmp_path: Path
) -> None:
    checkpoint_path, preprocessing_path = checkpoint
    monkeypatch.setenv("INFERENCE_MODEL_PATH", str(checkpoint_path))
    monkeypatch.setenv("INFERENCE_PREPROCESSING_CONFIG_PATH", str(preprocessing_path))
    monkeypatch.setenv("API_KEY_REQUIRED", "true")
    monkeypatch.setenv("API_KEY", "the-real-key")

    from ad_mri.api.main import create_app

    with TestClient(create_app()) as client:
        image = write_synthetic_nifti(tmp_path / "scan.nii.gz", shape=(32, 32, 32))

        no_key_response = client.post("/api/v1/predictions", files=_upload(image))
        assert no_key_response.status_code == 401

        wrong_key_response = client.post(
            "/api/v1/predictions", files=_upload(image), headers={"X-API-Key": "wrong"}
        )
        assert wrong_key_response.status_code == 401

        correct_key_response = client.post(
            "/api/v1/predictions", files=_upload(image), headers={"X-API-Key": "the-real-key"}
        )
        assert correct_key_response.status_code == 201


def test_predictions_get_is_not_allowed(configured_client: TestClient) -> None:
    response = configured_client.get("/api/v1/predictions")
    assert response.status_code == 405


def test_cors_headers_present_for_an_allowed_origin(
    monkeypatch: pytest.MonkeyPatch, checkpoint: tuple[Path, Path], tmp_path: Path
) -> None:
    checkpoint_path, preprocessing_path = checkpoint
    monkeypatch.setenv("INFERENCE_MODEL_PATH", str(checkpoint_path))
    monkeypatch.setenv("INFERENCE_PREPROCESSING_CONFIG_PATH", str(preprocessing_path))
    monkeypatch.setenv("ALLOWED_ORIGINS", "http://localhost:3000")

    from ad_mri.api.main import create_app

    with TestClient(create_app()) as client:
        # The preflight (OPTIONS) response only ever carries
        # Allow-Origin/Allow-Methods/Allow-Headers — never Expose-Headers,
        # which CORS only attaches to the actual response (see
        # starlette.middleware.cors.CORSMiddleware.preflight_response vs.
        # .simple_response). Check both, each against the response that
        # actually carries it.
        preflight = client.options(
            "/api/v1/predictions",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert preflight.headers.get("access-control-allow-origin") == "http://localhost:3000"

        image = write_synthetic_nifti(tmp_path / "scan.nii.gz", shape=(16, 16, 16))
        actual = client.post(
            "/api/v1/predictions", files=_upload(image), headers={"Origin": "http://localhost:3000"}
        )
        assert actual.headers.get("access-control-allow-origin") == "http://localhost:3000"
        assert "X-Request-ID" in actual.headers.get("access-control-expose-headers", "")
