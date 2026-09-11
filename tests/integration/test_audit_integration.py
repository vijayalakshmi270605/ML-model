"""Integration tests proving the audit trail (Phase 11) is actually wired
into the real FastAPI app, end-to-end - a real `TestClient` request against a
real, tiny checkpoint, read back from a real SQLite database. No mocks."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ad_mri.db.models import (
    ExplanationRecord,
    ModelVersionRecord,
    PredictionRequestRecord,
    PredictionResultRecord,
    SystemEventRecord,
)
from ad_mri.db.session import get_sessionmaker
from tests.helpers import write_synthetic_nifti

pytest.importorskip("lime")


def _upload(image_path: Path) -> dict[str, tuple[str, bytes, str]]:
    return {"file": (image_path.name, image_path.read_bytes(), "application/octet-stream")}


def test_a_successful_prediction_is_durably_recorded(
    configured_client_with_audit_db: TestClient, tmp_path: Path
) -> None:
    image = write_synthetic_nifti(tmp_path / "scan.nii.gz", shape=(16, 16, 16))
    response = configured_client_with_audit_db.post("/api/v1/predictions", files=_upload(image))
    assert response.status_code == 201
    body = response.json()

    session = get_sessionmaker()()
    try:
        model_versions = session.query(ModelVersionRecord).all()
        assert len(model_versions) == 1

        requests = session.query(PredictionRequestRecord).all()
        assert len(requests) == 1
        assert requests[0].external_request_id == body["request_id"]
        assert requests[0].model_version_id == model_versions[0].id
        # The redacted image_ref is a hash, never the uploaded filename.
        assert "scan" not in requests[0].image_ref

        results = session.query(PredictionResultRecord).all()
        assert len(results) == 1
        assert results[0].predicted_class == body["prediction"]
        assert results[0].request_id == requests[0].id
    finally:
        session.close()


def test_a_successful_explanation_is_durably_recorded(
    configured_client_with_audit_db: TestClient, tmp_path: Path
) -> None:
    image = write_synthetic_nifti(tmp_path / "scan.nii.gz", shape=(16, 16, 16))
    response = configured_client_with_audit_db.post(
        "/api/v1/explanations", files=_upload(image), params={"methods": "lime"}
    )
    assert response.status_code == 201
    body = response.json()

    session = get_sessionmaker()()
    try:
        records = session.query(ExplanationRecord).all()
        assert len(records) == 1
        assert records[0].external_explanation_id == body["explanation_id"]
        assert records[0].methods == "lime"
        assert records[0].artifact_uri.endswith("manifest.json")
    finally:
        session.close()


def test_a_mapped_error_response_also_writes_a_system_event(
    configured_client_with_audit_db: TestClient,
) -> None:
    response = configured_client_with_audit_db.post(
        "/api/v1/predictions",
        files={"file": ("garbage.nii.gz", b"not a real nifti file", "application/octet-stream")},
    )
    assert response.status_code == 422

    session = get_sessionmaker()()
    try:
        events = (
            session.query(SystemEventRecord)
            .filter_by(event_type="prediction_validation_failed")
            .all()
        )
        assert len(events) == 1
        assert events[0].severity == "warning"
        # The event's sanitized detail must never contain the uploaded filename.
        assert "garbage" not in str(events[0].detail)
    finally:
        session.close()


def test_a_prediction_still_succeeds_when_the_audit_database_is_unreachable(
    configured_client: TestClient, tmp_path: Path
) -> None:
    """`configured_client` (unlike `configured_client_with_audit_db`) never
    sets `DATABASE_URL`, so this exercises the real default (an unreachable
    local PostgreSQL) - the response must be unaffected."""
    image = write_synthetic_nifti(tmp_path / "scan.nii.gz", shape=(16, 16, 16))
    response = configured_client.post("/api/v1/predictions", files=_upload(image))
    assert response.status_code == 201
