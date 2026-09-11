"""Tests for `db.audit_service.AuditService` - real transactions against a
real (SQLite-backed) database, no mocks. Every "one transaction" claim in
this module's docstrings is checked here by actually reading the database
back with a *second*, independent `Session`."""

from __future__ import annotations

import pytest
from sqlalchemy.orm import sessionmaker

from ad_mri.db.audit_service import AuditService
from ad_mri.db.models import (
    ExplanationRecord,
    ModelVersionRecord,
    PredictionRequestRecord,
    PredictionResultRecord,
)

_MODEL_VERSION_KWARGS = dict(
    architecture="ad-mri-convnet3d",
    model_version="1.0.0",
    model_config_hash="cfg-hash-a",
    preprocessing_version="1.0.0",
    preprocessing_config_hash="prep-hash-a",
    checkpoint_uri="file:///artifacts/checkpoints/model.pt",
    device="cpu",
)
_REAL_IMAGE_REF = "0123456789abcdef"


class TestRecordPrediction:
    def test_writes_model_version_request_and_result_in_one_transaction(
        self, db_session, db_sessionmaker: sessionmaker
    ) -> None:
        service = AuditService(db_session)
        result = service.record_prediction(
            request_id="caller-facing-request-id",
            image_ref=_REAL_IMAGE_REF,
            content_type="application/octet-stream",
            api_key="the-real-key",
            predicted_class="AD",
            confidence=0.91,
            probabilities={"AD": 0.91, "CN": 0.09},
            processing_time_ms=55.5,
            explanation_available=False,
            **_MODEL_VERSION_KWARGS,
        )

        # Read back with an independent session - proves the write is
        # actually durable (committed), not just visible within db_session.
        verify = db_sessionmaker()
        try:
            assert verify.get(PredictionResultRecord, result.id) is not None
            model_versions = verify.query(ModelVersionRecord).all()
            assert len(model_versions) == 1
            requests = verify.query(PredictionRequestRecord).all()
            assert len(requests) == 1
            assert requests[0].external_request_id == "caller-facing-request-id"
            assert requests[0].model_version_id == model_versions[0].id
            # The presented API key is never stored - only its hash.
            assert requests[0].api_key_hash is not None
            assert requests[0].api_key_hash != "the-real-key"
        finally:
            verify.close()

    def test_reuses_the_existing_model_version_row_on_a_second_call(
        self, db_session, db_sessionmaker: sessionmaker
    ) -> None:
        service = AuditService(db_session)
        for i in range(2):
            service.record_prediction(
                request_id=f"req-{i}",
                image_ref=_REAL_IMAGE_REF,
                content_type=None,
                api_key=None,
                predicted_class="AD",
                confidence=0.5,
                probabilities={"AD": 0.5, "CN": 0.5},
                processing_time_ms=1.0,
                explanation_available=False,
                **_MODEL_VERSION_KWARGS,
            )

        verify = db_sessionmaker()
        try:
            assert verify.query(ModelVersionRecord).count() == 1
            assert verify.query(PredictionRequestRecord).count() == 2
        finally:
            verify.close()

    def test_a_failed_write_rolls_back_the_entire_transaction(
        self, db_session, db_sessionmaker: sessionmaker
    ) -> None:
        """A bad `image_ref` fails inside `PredictionRequestRepository.create`
        - *after* the model-version row was already added to the session in
        the same call. The whole transaction, including that model-version
        insert, must be rolled back, not partially committed."""
        service = AuditService(db_session)

        with pytest.raises(ValueError, match="image_ref"):
            service.record_prediction(
                request_id="req-bad",
                image_ref="not-a-redacted-hash.nii.gz",
                content_type=None,
                api_key=None,
                predicted_class="AD",
                confidence=0.5,
                probabilities={"AD": 0.5, "CN": 0.5},
                processing_time_ms=1.0,
                explanation_available=False,
                **_MODEL_VERSION_KWARGS,
            )

        verify = db_sessionmaker()
        try:
            assert verify.query(ModelVersionRecord).count() == 0
            assert verify.query(PredictionRequestRecord).count() == 0
        finally:
            verify.close()

        # And the session itself is usable again afterwards - rollback()
        # really cleared the failed transaction, not just logged it.
        service.record_prediction(
            request_id="req-good",
            image_ref=_REAL_IMAGE_REF,
            content_type=None,
            api_key=None,
            predicted_class="AD",
            confidence=0.5,
            probabilities={"AD": 0.5, "CN": 0.5},
            processing_time_ms=1.0,
            explanation_available=False,
            **_MODEL_VERSION_KWARGS,
        )
        verify2 = db_sessionmaker()
        try:
            assert verify2.query(PredictionRequestRecord).count() == 1
        finally:
            verify2.close()


class TestRecordExplanation:
    def test_writes_explanation_and_model_version_together(
        self, db_session, db_sessionmaker: sessionmaker
    ) -> None:
        service = AuditService(db_session)
        record = service.record_explanation(
            explanation_id="expl-external-1",
            request_id=None,
            methods="lime",
            target_class="AD",
            artifact_uri="file:///artifacts/explanations/expl-1/manifest.json",
            **_MODEL_VERSION_KWARGS,
        )

        verify = db_sessionmaker()
        try:
            fetched = verify.get(ExplanationRecord, record.id)
            assert fetched is not None
            assert fetched.external_explanation_id == "expl-external-1"
            assert fetched.request_id is None
            assert verify.query(ModelVersionRecord).count() == 1
        finally:
            verify.close()

    def test_params_hash_differs_for_different_target_classes(self, db_session) -> None:
        service = AuditService(db_session)
        first = service.record_explanation(
            explanation_id="expl-a",
            request_id=None,
            methods="lime",
            target_class="AD",
            artifact_uri="file:///a/manifest.json",
            **_MODEL_VERSION_KWARGS,
        )
        second = service.record_explanation(
            explanation_id="expl-b",
            request_id=None,
            methods="lime",
            target_class="CN",
            artifact_uri="file:///b/manifest.json",
            **_MODEL_VERSION_KWARGS,
        )
        assert first.params_hash != second.params_hash


class TestRecordModelVersion:
    def test_get_or_create_semantics_are_exposed_directly(
        self, db_session, db_sessionmaker: sessionmaker
    ) -> None:
        service = AuditService(db_session)
        first = service.record_model_version(stage="Production", **_MODEL_VERSION_KWARGS)
        second = service.record_model_version(stage="Production", **_MODEL_VERSION_KWARGS)
        assert first.id == second.id

        verify = db_sessionmaker()
        try:
            assert verify.query(ModelVersionRecord).count() == 1
        finally:
            verify.close()


class TestRecordEvent:
    def test_persists_a_sanitized_event(self, db_session, db_sessionmaker: sessionmaker) -> None:
        service = AuditService(db_session)
        event = service.record_event(
            event_type="rate_limit_exceeded",
            severity="warning",
            request_id="caller-request-id",
            detail={"limit": 5, "window_seconds": 60},
        )

        verify = db_sessionmaker()
        try:
            from ad_mri.db.models import SystemEventRecord

            fetched = verify.get(SystemEventRecord, event.id)
            assert fetched is not None
            assert fetched.event_type == "rate_limit_exceeded"
            assert fetched.detail == {"limit": 5, "window_seconds": 60}
        finally:
            verify.close()

    def test_accepts_a_non_uuid_shaped_request_id(self, db_session) -> None:
        service = AuditService(db_session)
        event = service.record_event(event_type="auth_failed", request_id="totally-not-a-uuid")
        assert event.request_id == "totally-not-a-uuid"
