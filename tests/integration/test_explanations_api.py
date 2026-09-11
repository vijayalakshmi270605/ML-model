"""Integration tests for POST /api/v1/explanations — a real FastAPI TestClient
against a real, tiny checkpoint (no mocks)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.helpers import write_synthetic_nifti

pytest.importorskip("lime")


def _upload(image_path: Path) -> dict[str, tuple[str, bytes, str]]:
    return {"file": (image_path.name, image_path.read_bytes(), "application/octet-stream")}


def test_explanations_returns_201_with_a_valid_response(
    configured_client: TestClient, tmp_path: Path
) -> None:
    image = write_synthetic_nifti(tmp_path / "scan.nii.gz", shape=(32, 32, 32))
    response = configured_client.post(
        "/api/v1/explanations", files=_upload(image), params={"methods": "lime"}
    )

    assert response.status_code == 201
    body = response.json()
    assert set(body) == {
        "request_id",
        "model_version",
        "preprocessing_version",
        "prediction",
        "confidence",
        "probabilities",
        "explanation_id",
        "methods",
        "processing_time_ms",
        "timestamp",
        "disclaimer",
        "narrative_summary",
    }
    # Phase 14 ML-safety control: explanation responses get the
    # causality-specific wording, not the plain prediction one - see
    # core.disclaimers.
    assert "causal" in body["disclaimer"]
    # narrative_summary is None here: OPENAI_API_KEY is forced blank for this
    # whole suite (tests/conftest.py) so no test ever makes a real network
    # call - see test_llm_summary.py for the feature itself.
    assert body["narrative_summary"] is None
    assert body["prediction"] in {"AD", "CN"}
    assert body["methods"] == ["lime"]
    assert body["explanation_id"] == body["request_id"]
    assert response.headers["X-Request-ID"] == body["request_id"]


def test_explanations_can_target_a_specific_class(
    configured_client: TestClient, tmp_path: Path
) -> None:
    image = write_synthetic_nifti(tmp_path / "scan.nii.gz", shape=(32, 32, 32))
    response = configured_client.post(
        "/api/v1/explanations",
        files=_upload(image),
        params={"methods": "lime", "target_class": "CN"},
    )
    assert response.status_code == 201


def test_explanations_rejects_an_unknown_target_class(
    configured_client: TestClient, tmp_path: Path
) -> None:
    image = write_synthetic_nifti(tmp_path / "scan.nii.gz", shape=(32, 32, 32))
    response = configured_client.post(
        "/api/v1/explanations",
        files=_upload(image),
        params={"methods": "lime", "target_class": "not-a-real-class"},
    )
    assert response.status_code == 500
    assert response.json()["error_code"] == "CONFIGURATION_ERROR"


def test_explanations_fails_loudly_when_shap_requested_without_background(
    configured_client: TestClient, tmp_path: Path
) -> None:
    """Unlike the best-effort inline path on /predictions, /explanations is
    the dedicated endpoint for producing an explanation — a failure here must
    be a real error, not a silently "unavailable" success."""
    image = write_synthetic_nifti(tmp_path / "scan.nii.gz", shape=(32, 32, 32))
    response = configured_client.post(
        "/api/v1/explanations", files=_upload(image), params={"methods": "shap"}
    )
    assert response.status_code == 500
    assert response.json()["error_code"] == "CONFIGURATION_ERROR"


def test_explanations_rejects_an_invalid_upload_with_422(configured_client: TestClient) -> None:
    response = configured_client.post(
        "/api/v1/explanations",
        files={
            "file": ("garbage.nii.gz", b"this is not a real nifti file", "application/octet-stream")
        },
    )
    assert response.status_code == 422


def test_explanations_returns_503_when_engine_not_configured(
    client: TestClient, tmp_path: Path
) -> None:
    image = write_synthetic_nifti(tmp_path / "scan.nii.gz", shape=(32, 32, 32))
    response = client.post("/api/v1/explanations", files=_upload(image))
    assert response.status_code == 503


def test_explanations_attaches_a_narrative_summary_when_openai_is_configured(
    monkeypatch: pytest.MonkeyPatch, checkpoint: tuple[Path, Path], tmp_path: Path
) -> None:
    """The one deliberate exception to this file's "no mocks" rule: only the
    third-party `openai` SDK's transport (the actual external network
    boundary `explainability.llm_summary` isolates - see its module
    docstring) is stubbed, so this test can prove the *router*'s wiring
    (settings -> generate_narrative_summary -> response field) end-to-end
    without a real, billed network call. `test_llm_summary.py` covers the
    module's own payload-construction/error-handling contract directly.

    Builds its own `TestClient` (rather than the shared `configured_client`
    fixture) so `OPENAI_API_KEY` is set *before* `create_app()`'s first
    `get_settings()` call populates the process-wide cache - see
    `test_predictions_rejects_an_oversized_upload_with_413` for the same
    pattern."""
    checkpoint_path, preprocessing_path = checkpoint
    monkeypatch.setenv("INFERENCE_MODEL_PATH", str(checkpoint_path))
    monkeypatch.setenv("INFERENCE_PREPROCESSING_CONFIG_PATH", str(preprocessing_path))
    monkeypatch.setenv("INFERENCE_EXPLANATION_OUTPUT_DIR", str(tmp_path / "explanations"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")

    class _Message:
        content = "The model's output was most sensitive to a small set of regions."

    class _Choice:
        message = _Message()

    class _Response:
        choices = [_Choice()]

    class _StubClient:
        def __init__(self, **_kwargs: object) -> None:
            self.chat = self

        class completions:
            @staticmethod
            async def create(**_kwargs: object) -> _Response:
                return _Response()

    import openai

    monkeypatch.setattr(openai, "AsyncOpenAI", _StubClient)

    from ad_mri.api.main import create_app

    with TestClient(create_app()) as client:
        image = write_synthetic_nifti(tmp_path / "scan.nii.gz", shape=(32, 32, 32))
        response = client.post(
            "/api/v1/explanations", files=_upload(image), params={"methods": "lime"}
        )
    assert response.status_code == 201
    assert response.json()["narrative_summary"] == (
        "The model's output was most sensitive to a small set of regions."
    )


def test_explanations_writes_a_real_manifest(configured_client: TestClient, tmp_path: Path) -> None:
    image = write_synthetic_nifti(tmp_path / "scan.nii.gz", shape=(32, 32, 32))
    response = configured_client.post(
        "/api/v1/explanations", files=_upload(image), params={"methods": "lime"}
    )
    explanation_id = response.json()["explanation_id"]
    manifest_path = tmp_path / "explanations" / explanation_id / "manifest.json"
    assert manifest_path.exists()
