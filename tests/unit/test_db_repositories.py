"""Tests for `db.repositories` - real CRUD against a real (SQLite-backed)
`Session`, no mocks. Each repository method is exercised directly; the
transaction-management behaviour lives in `db.audit_service` and is covered
by `test_audit_service.py` instead."""

from __future__ import annotations

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ad_mri.db.repositories import (
    ExplanationRepository,
    ModelVersionRepository,
    PredictionRequestRepository,
    PredictionResultRepository,
    SystemEventRepository,
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

_REAL_IMAGE_REF = "0123456789abcdef"  # 16 lowercase hex chars - the redact_image_reference() shape


class TestModelVersionRepository:
    def test_create_and_get_round_trip(self, db_session: Session) -> None:
        repo = ModelVersionRepository(db_session)
        created = repo.create(**_MODEL_VERSION_KWARGS)
        db_session.commit()

        fetched = repo.get(created.id)
        assert fetched is not None
        assert fetched.model_config_hash == "cfg-hash-a"

    def test_get_or_create_is_idempotent_for_the_same_config_hash_pair(
        self, db_session: Session
    ) -> None:
        repo = ModelVersionRepository(db_session)
        first = repo.get_or_create(**_MODEL_VERSION_KWARGS)
        second = repo.get_or_create(**_MODEL_VERSION_KWARGS)
        db_session.commit()

        assert first.id == second.id
        assert repo.list_recent(limit=10).__len__() == 1

    def test_create_twice_with_the_same_config_hash_pair_violates_the_unique_constraint(
        self, db_session: Session
    ) -> None:
        repo = ModelVersionRepository(db_session)
        repo.create(**_MODEL_VERSION_KWARGS)
        db_session.commit()

        with pytest.raises(IntegrityError):
            repo.create(**_MODEL_VERSION_KWARGS)
            db_session.commit()

    def test_get_by_config_hash_returns_none_for_an_unknown_pair(self, db_session: Session) -> None:
        repo = ModelVersionRepository(db_session)
        assert repo.get_by_config_hash("no-such-hash", "no-such-hash") is None


class TestPredictionRequestRepository:
    def test_create_rejects_a_non_redacted_image_ref(self, db_session: Session) -> None:
        repo = PredictionRequestRepository(db_session)
        with pytest.raises(ValueError, match="image_ref"):
            repo.create(
                external_request_id="req-1",
                model_version_id=None,
                image_ref="john_doe_1957.nii.gz",
                content_type="application/octet-stream",
                api_key_hash=None,
            )

    def test_create_and_get_round_trip(self, db_session: Session) -> None:
        repo = PredictionRequestRepository(db_session)
        created = repo.create(
            external_request_id="req-external-1",
            model_version_id=None,
            image_ref=_REAL_IMAGE_REF,
            content_type="application/octet-stream",
            api_key_hash=None,
        )
        db_session.commit()

        fetched = repo.get(created.id)
        assert fetched is not None
        assert fetched.external_request_id == "req-external-1"
        assert fetched.image_ref == _REAL_IMAGE_REF

    def test_get_by_external_request_id(self, db_session: Session) -> None:
        repo = PredictionRequestRepository(db_session)
        repo.create(
            external_request_id="caller-facing-id",
            model_version_id=None,
            image_ref=_REAL_IMAGE_REF,
            content_type=None,
            api_key_hash=None,
        )
        db_session.commit()

        found = repo.get_by_external_request_id("caller-facing-id")
        assert found is not None
        assert found.external_request_id == "caller-facing-id"


class TestPredictionResultRepository:
    def test_create_requires_an_existing_request_row(self, db_session: Session) -> None:
        """A real PostgreSQL foreign key would reject this - SQLite is
        configured (see `tests/unit/conftest.py`) to enforce it too."""
        import uuid

        repo = PredictionResultRepository(db_session)
        # `create()` flushes immediately, so the foreign-key violation
        # surfaces here rather than at a later `commit()`.
        with pytest.raises(IntegrityError):
            repo.create(
                request_id=uuid.uuid4(),  # no matching prediction_requests row
                predicted_class="AD",
                confidence=0.9,
                probabilities={"AD": 0.9, "CN": 0.1},
                processing_time_ms=12.3,
                explanation_available=False,
            )

    def test_create_and_get_by_request_id(self, db_session: Session) -> None:
        request_repo = PredictionRequestRepository(db_session)
        result_repo = PredictionResultRepository(db_session)

        request = request_repo.create(
            external_request_id="req-2",
            model_version_id=None,
            image_ref=_REAL_IMAGE_REF,
            content_type=None,
            api_key_hash=None,
        )
        result_repo.create(
            request_id=request.id,
            predicted_class="AD",
            confidence=0.87,
            probabilities={"AD": 0.87, "CN": 0.13},
            processing_time_ms=42.0,
            explanation_available=True,
        )
        db_session.commit()

        fetched = result_repo.get_by_request_id(request.id)
        assert fetched is not None
        assert fetched.predicted_class == "AD"
        assert fetched.probabilities == {"AD": 0.87, "CN": 0.13}


class TestExplanationRepository:
    def test_create_with_no_request_row_is_allowed(self, db_session: Session) -> None:
        """`/explanations` never creates a `PredictionRequestRecord` - a
        standalone explanation must still be recordable."""
        repo = ExplanationRepository(db_session)
        created = repo.create(
            external_explanation_id="expl-external-1",
            request_id=None,
            model_version_id=None,
            methods="lime",
            target_class="AD",
            artifact_uri="file:///artifacts/explanations/expl-1/manifest.json",
            params_hash="hash-x",
        )
        db_session.commit()

        fetched = repo.get(created.id)
        assert fetched is not None
        assert fetched.request_id is None
        assert fetched.methods == "lime"


class TestSystemEventRepository:
    def test_create_and_list_recent(self, db_session: Session) -> None:
        repo = SystemEventRepository(db_session)
        repo.create(event_type="rate_limit_exceeded", severity="warning", detail={"limit": 5})
        repo.create(event_type="auth_failed", severity="warning", detail=None)
        db_session.commit()

        events = repo.list_recent(limit=10)
        assert {e.event_type for e in events} == {"rate_limit_exceeded", "auth_failed"}

    def test_list_recent_filters_by_event_type(self, db_session: Session) -> None:
        repo = SystemEventRepository(db_session)
        repo.create(event_type="rate_limit_exceeded", severity="warning")
        repo.create(event_type="auth_failed", severity="warning")
        db_session.commit()

        events = repo.list_recent(limit=10, event_type="auth_failed")
        assert len(events) == 1
        assert events[0].event_type == "auth_failed"

    def test_request_id_accepts_a_non_uuid_shaped_caller_value(self, db_session: Session) -> None:
        """`X-Request-ID` is caller-suppliable and not required to be a real
        UUID (`RequestIDMiddleware` accepts it verbatim) - this column must
        tolerate that."""
        repo = SystemEventRepository(db_session)
        repo.create(event_type="inference_failed", request_id="not-a-uuid-at-all")
        db_session.commit()

        events = repo.list_recent(limit=10)
        assert events[0].request_id == "not-a-uuid-at-all"
