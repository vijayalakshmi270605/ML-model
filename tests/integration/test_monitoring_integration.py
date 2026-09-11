"""Integration tests proving the Phase 13 monitoring wiring is real,
end-to-end - a real `TestClient` request against a real, tiny checkpoint,
read back from a real SQLite database. No mocks."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from ad_mri.db.models import InputStatisticsRecord
from ad_mri.db.session import get_sessionmaker
from tests.helpers import write_synthetic_nifti


def _upload(image_path: Path) -> dict[str, tuple[str, bytes, str]]:
    return {"file": (image_path.name, image_path.read_bytes(), "application/octet-stream")}


def test_a_prediction_records_real_input_statistics(
    configured_client_with_audit_db: TestClient, tmp_path: Path
) -> None:
    image = write_synthetic_nifti(tmp_path / "scan.nii.gz", shape=(16, 16, 16))
    response = configured_client_with_audit_db.post("/api/v1/predictions", files=_upload(image))
    assert response.status_code == 201

    session = get_sessionmaker()()
    try:
        rows = session.query(InputStatisticsRecord).all()
        assert len(rows) == 1
        assert rows[0].voxel_count > 0
    finally:
        session.close()


def test_a_prediction_updates_prometheus_model_metrics(
    configured_client_with_audit_db: TestClient, tmp_path: Path
) -> None:
    before = REGISTRY.get_sample_value("ad_mri_prediction_confidence_count")

    image = write_synthetic_nifti(tmp_path / "scan.nii.gz", shape=(16, 16, 16))
    response = configured_client_with_audit_db.post("/api/v1/predictions", files=_upload(image))
    assert response.status_code == 201
    predicted_class = response.json()["prediction"]

    after = REGISTRY.get_sample_value("ad_mri_prediction_confidence_count")
    assert after == (before or 0) + 1

    class_count = REGISTRY.get_sample_value(
        "ad_mri_predicted_class_total", {"predicted_class": predicted_class}
    )
    assert class_count is not None and class_count >= 1

    mean_intensity_count = REGISTRY.get_sample_value("ad_mri_input_mean_intensity_count")
    assert mean_intensity_count is not None and mean_intensity_count >= 1


def test_the_metrics_endpoint_includes_system_resource_metrics(
    configured_client_with_audit_db: TestClient,
) -> None:
    response = configured_client_with_audit_db.get("/metrics")
    assert response.status_code == 200
    body = response.text
    assert "ad_mri_process_memory_bytes" in body
    assert "ad_mri_system_memory_percent" in body


def test_record_ground_truth_via_the_api_and_read_it_back(
    configured_client_with_audit_db: TestClient, tmp_path: Path
) -> None:
    image = write_synthetic_nifti(tmp_path / "scan.nii.gz", shape=(16, 16, 16))
    prediction = configured_client_with_audit_db.post(
        "/api/v1/predictions", files=_upload(image)
    ).json()

    response = configured_client_with_audit_db.post(
        "/api/v1/monitoring/ground-truth",
        json={
            "request_id": prediction["request_id"],
            "true_class": "AD",
            "label_source": "clinician_review",
        },
    )
    assert response.status_code == 201
    assert response.json()["true_class"] == "AD"

    from ad_mri.db.models import GroundTruthLabelRecord

    session = get_sessionmaker()()
    try:
        rows = session.query(GroundTruthLabelRecord).all()
        assert len(rows) == 1
        assert rows[0].true_class == "AD"
    finally:
        session.close()


def test_record_ground_truth_for_an_unknown_request_returns_404(
    configured_client_with_audit_db: TestClient,
) -> None:
    response = configured_client_with_audit_db.post(
        "/api/v1/monitoring/ground-truth",
        json={
            "request_id": "not-a-real-request-id",
            "true_class": "AD",
            "label_source": "clinician_review",
        },
    )
    assert response.status_code == 404
