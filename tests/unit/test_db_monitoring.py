"""Tests for the Phase 13 monitoring DB layer - real CRUD/joins against a
real (SQLite-backed) database, no mocks."""

from __future__ import annotations

import pytest
from sqlalchemy.orm import sessionmaker

from ad_mri.db.monitoring_service import MonitoringService
from ad_mri.db.repositories import (
    GroundTruthLabelRepository,
    InputStatisticsRepository,
    ModelVersionRepository,
    PredictionRequestRepository,
    PredictionResultRepository,
)

_MODEL_VERSION_KWARGS = dict(
    architecture="ad-mri-convnet3d",
    model_version="1.0.0",
    model_config_hash="cfg-hash-a",
    preprocessing_version="1.0.0",
    preprocessing_config_hash="prep-hash-a",
    checkpoint_uri="file:///artifacts/checkpoints/model.pt",
    stage="Production",
    device="cpu",
)
_REAL_IMAGE_REF = "0123456789abcdef"


def _make_prediction(
    session, *, external_request_id: str, predicted_class: str, model_version_id=None
):
    request = PredictionRequestRepository(session).create(
        external_request_id=external_request_id,
        model_version_id=model_version_id,
        image_ref=_REAL_IMAGE_REF,
        content_type=None,
        api_key_hash=None,
    )
    PredictionResultRepository(session).create(
        request_id=request.id,
        predicted_class=predicted_class,
        confidence=0.9,
        probabilities={"AD": 0.9, "CN": 0.1},
        processing_time_ms=1.0,
        explanation_available=False,
    )
    session.commit()
    return request


class TestModelVersionRepositorySetBaselineMetrics:
    def test_sets_and_persists_baseline_metrics(
        self, db_session, db_sessionmaker: sessionmaker
    ) -> None:
        repo = ModelVersionRepository(db_session)
        version = repo.create(**_MODEL_VERSION_KWARGS)
        db_session.commit()

        repo.set_baseline_metrics(version.id, {"accuracy": 0.9, "f1_macro": 0.88})
        db_session.commit()

        verify = db_sessionmaker()
        try:
            fetched = repo.__class__(verify).get(version.id)
            assert fetched.baseline_metrics == {"accuracy": 0.9, "f1_macro": 0.88}
        finally:
            verify.close()

    def test_raises_for_an_unknown_model_version(self, db_session) -> None:
        import uuid

        repo = ModelVersionRepository(db_session)
        with pytest.raises(ValueError, match="No model_versions row"):
            repo.set_baseline_metrics(uuid.uuid4(), {"accuracy": 0.9})


class TestInputStatisticsRepository:
    def test_create_and_list_between(self, db_session) -> None:
        request = _make_prediction(db_session, external_request_id="req-1", predicted_class="AD")
        repo = InputStatisticsRepository(db_session)
        repo.create(
            request_id=request.id,
            mean_intensity=0.5,
            std_intensity=0.1,
            p01_intensity=0.0,
            p50_intensity=0.5,
            p99_intensity=1.0,
            nonzero_fraction=0.8,
            voxel_count=1000,
        )
        db_session.commit()

        rows = repo.list_between()
        assert len(rows) == 1
        assert rows[0].mean_intensity == 0.5


class TestGroundTruthLabelRepository:
    def test_create_or_update_creates_then_updates(self, db_session) -> None:
        request = _make_prediction(db_session, external_request_id="req-2", predicted_class="AD")
        repo = GroundTruthLabelRepository(db_session)

        first = repo.create_or_update(
            request_id=request.id, true_class="AD", label_source="clinician_review"
        )
        db_session.commit()
        assert first.true_class == "AD"

        second = repo.create_or_update(
            request_id=request.id, true_class="CN", label_source="follow_up_diagnosis"
        )
        db_session.commit()

        assert second.id == first.id  # updated, not duplicated
        assert second.true_class == "CN"
        assert repo.get_by_request_id(request.id).true_class == "CN"

    def test_list_labeled_predictions_for_model_version(self, db_session) -> None:
        model_repo = ModelVersionRepository(db_session)
        version = model_repo.create(**_MODEL_VERSION_KWARGS)
        db_session.commit()

        req_a = _make_prediction(
            db_session,
            external_request_id="req-a",
            predicted_class="AD",
            model_version_id=version.id,
        )
        req_b = _make_prediction(
            db_session,
            external_request_id="req-b",
            predicted_class="CN",
            model_version_id=version.id,
        )
        # A prediction from a *different* model version - must not appear in the join.
        other_version = model_repo.create(
            **{**_MODEL_VERSION_KWARGS, "model_config_hash": "cfg-hash-b"}
        )
        db_session.commit()
        req_other = _make_prediction(
            db_session,
            external_request_id="req-other",
            predicted_class="AD",
            model_version_id=other_version.id,
        )

        gt_repo = GroundTruthLabelRepository(db_session)
        gt_repo.create_or_update(
            request_id=req_a.id, true_class="AD", label_source="clinician_review"
        )
        gt_repo.create_or_update(
            request_id=req_b.id, true_class="AD", label_source="clinician_review"
        )
        gt_repo.create_or_update(
            request_id=req_other.id, true_class="AD", label_source="clinician_review"
        )
        db_session.commit()

        pairs = gt_repo.list_labeled_predictions_for_model_version(version.id)
        assert sorted(pairs) == sorted([("AD", "AD"), ("CN", "AD")])


class TestMonitoringService:
    def test_record_input_statistics_no_ops_for_an_unknown_request(self, db_session) -> None:
        service = MonitoringService(db_session)
        result = service.record_input_statistics(
            external_request_id="never-recorded",
            mean_intensity=0.5,
            std_intensity=0.1,
            p01_intensity=0.0,
            p50_intensity=0.5,
            p99_intensity=1.0,
            nonzero_fraction=0.8,
            voxel_count=1000,
        )
        assert result is None

    def test_record_input_statistics_writes_a_real_row(
        self, db_session, db_sessionmaker: sessionmaker
    ) -> None:
        _make_prediction(db_session, external_request_id="req-svc-1", predicted_class="AD")
        service = MonitoringService(db_session)

        result = service.record_input_statistics(
            external_request_id="req-svc-1",
            mean_intensity=0.5,
            std_intensity=0.1,
            p01_intensity=0.0,
            p50_intensity=0.5,
            p99_intensity=1.0,
            nonzero_fraction=0.8,
            voxel_count=1000,
        )
        assert result is not None

        verify = db_sessionmaker()
        try:
            window = InputStatisticsRepository(verify).list_between()
            assert len(window) == 1
        finally:
            verify.close()

    def test_record_ground_truth_raises_for_an_unknown_request(self, db_session) -> None:
        service = MonitoringService(db_session)
        with pytest.raises(ValueError, match="No recorded prediction"):
            service.record_ground_truth(
                external_request_id="never-recorded",
                true_class="AD",
                label_source="clinician_review",
            )

    def test_record_ground_truth_and_get_labeled_predictions_round_trip(
        self, db_session, db_sessionmaker: sessionmaker
    ) -> None:
        model_repo = ModelVersionRepository(db_session)
        version = model_repo.create(**_MODEL_VERSION_KWARGS)
        db_session.commit()

        _make_prediction(
            db_session,
            external_request_id="req-svc-2",
            predicted_class="AD",
            model_version_id=version.id,
        )
        service = MonitoringService(db_session)
        service.record_ground_truth(
            external_request_id="req-svc-2", true_class="AD", label_source="clinician_review"
        )

        pairs = service.get_labeled_predictions(version.id)
        assert pairs == [("AD", "AD")]

    def test_set_baseline_metrics_via_service(self, db_session) -> None:
        model_repo = ModelVersionRepository(db_session)
        version = model_repo.create(**_MODEL_VERSION_KWARGS)
        db_session.commit()

        service = MonitoringService(db_session)
        updated = service.set_baseline_metrics(version.id, {"accuracy": 0.95})
        assert updated.baseline_metrics == {"accuracy": 0.95}

    def test_get_predicted_class_counts(self, db_session) -> None:
        _make_prediction(db_session, external_request_id="req-c1", predicted_class="AD")
        _make_prediction(db_session, external_request_id="req-c2", predicted_class="AD")
        _make_prediction(db_session, external_request_id="req-c3", predicted_class="CN")

        service = MonitoringService(db_session)
        counts = service.get_predicted_class_counts()
        assert counts == {"AD": 2, "CN": 1}
