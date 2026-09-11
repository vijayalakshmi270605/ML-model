"""Service layer: orchestrates the repository layer into whole, transactional
units of work - "record a completed prediction" writes its
`PredictionRequestRecord` and `PredictionResultRecord` (and, if the model
version is new, its `ModelVersionRecord`) as one commit, never a partially
written pair. Every public method here is **loud**: a database failure
raises (after rolling back), it is never swallowed. This mirrors
`inference.predictor.InferenceEngine.generate_explanation` (loud) versus
`_try_generate_explanation` (best-effort) - the best-effort wrapper for the
audit trail is `api.audit`, one layer up, not here. Keeping this layer loud
is what makes it possible to unit-test with a real (SQLite) database and
trust that "no exception raised" really means "the row is durably there".
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy.orm import Session

from ad_mri.db.models import (
    ExplanationRecord,
    ModelVersionRecord,
    PredictionResultRecord,
    SystemEventRecord,
)
from ad_mri.db.privacy import hash_value
from ad_mri.db.repositories import (
    ExplanationRepository,
    ModelVersionRepository,
    PredictionRequestRepository,
    PredictionResultRepository,
    SystemEventRepository,
)


class AuditService:
    """One instance per request-scoped `Session` (see `api.deps.get_db`).
    Each method is a self-contained transaction, run through `_transaction()`:
    it commits on success and rolls back (then re-raises) on *any* failure -
    including one raised by a repository call itself (e.g. a privacy-guard
    `ValueError`), not only one raised by the final `commit()` - so a partial
    write attempt never leaves the session holding a flushed-but-uncommitted
    row from an earlier step in the same method."""

    def __init__(self, session: Session) -> None:
        self._session = session
        self._model_versions = ModelVersionRepository(session)
        self._prediction_requests = PredictionRequestRepository(session)
        self._prediction_results = PredictionResultRepository(session)
        self._explanations = ExplanationRepository(session)
        self._system_events = SystemEventRepository(session)

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        try:
            yield
            self._session.commit()
        except Exception:
            self._session.rollback()
            raise

    def record_model_version(
        self,
        *,
        architecture: str,
        model_version: str,
        model_config_hash: str,
        preprocessing_version: str,
        preprocessing_config_hash: str,
        checkpoint_uri: str,
        stage: str,
        device: str,
    ) -> ModelVersionRecord:
        """Idempotent: registers this (model_config_hash,
        preprocessing_config_hash) pair once; a repeat call for the same pair
        returns the existing row rather than inserting a duplicate."""
        with self._transaction():
            record = self._model_versions.get_or_create(
                architecture=architecture,
                model_version=model_version,
                model_config_hash=model_config_hash,
                preprocessing_version=preprocessing_version,
                preprocessing_config_hash=preprocessing_config_hash,
                checkpoint_uri=checkpoint_uri,
                stage=stage,
                device=device,
            )
        return record

    def record_prediction(
        self,
        *,
        request_id: str,
        architecture: str,
        model_version: str,
        model_config_hash: str,
        preprocessing_version: str,
        preprocessing_config_hash: str,
        checkpoint_uri: str,
        device: str,
        image_ref: str,
        content_type: str | None,
        api_key: str | None,
        predicted_class: str,
        confidence: float,
        probabilities: dict[str, float],
        processing_time_ms: float,
        explanation_available: bool,
    ) -> PredictionResultRecord:
        """Writes the model-version row (idempotent get-or-create), the
        request row, and its result row as one transaction - `Requirements
        5-10`: a UUID request id (this row's own `id`; `request_id` here is
        the caller-facing value, stored as `external_request_id`), the model
        version used, the received timestamp (server-default), processing
        time, and the prediction result, all durable together or not at
        all."""
        with self._transaction():
            model_version_record = self._model_versions.get_or_create(
                architecture=architecture,
                model_version=model_version,
                model_config_hash=model_config_hash,
                preprocessing_version=preprocessing_version,
                preprocessing_config_hash=preprocessing_config_hash,
                checkpoint_uri=checkpoint_uri,
                stage="Production",
                device=device,
            )
            request_record = self._prediction_requests.create(
                external_request_id=request_id,
                model_version_id=model_version_record.id,
                image_ref=image_ref,
                content_type=content_type,
                api_key_hash=hash_value(api_key) if api_key else None,
            )
            result = self._prediction_results.create(
                request_id=request_record.id,
                predicted_class=predicted_class,
                confidence=confidence,
                probabilities=probabilities,
                processing_time_ms=processing_time_ms,
                explanation_available=explanation_available,
            )
        return result

    def record_explanation(
        self,
        *,
        explanation_id: str,
        request_id: uuid.UUID | None,
        architecture: str,
        model_version: str,
        model_config_hash: str,
        preprocessing_version: str,
        preprocessing_config_hash: str,
        checkpoint_uri: str,
        device: str,
        methods: str,
        target_class: str | None,
        artifact_uri: str,
    ) -> ExplanationRecord:
        """Requirement 11 (explanation metadata): method(s) used, the target
        class, and a reference (never the arrays themselves) to the written
        manifest - plus the model-version row (idempotent get-or-create), all
        in one transaction."""
        with self._transaction():
            model_version_record = self._model_versions.get_or_create(
                architecture=architecture,
                model_version=model_version,
                model_config_hash=model_config_hash,
                preprocessing_version=preprocessing_version,
                preprocessing_config_hash=preprocessing_config_hash,
                checkpoint_uri=checkpoint_uri,
                stage="Production",
                device=device,
            )
            params_hash = hash_value(f"{methods}|{target_class or ''}")
            record = self._explanations.create(
                external_explanation_id=explanation_id,
                request_id=request_id,
                model_version_id=model_version_record.id,
                methods=methods,
                target_class=target_class,
                artifact_uri=artifact_uri,
                params_hash=params_hash,
            )
        return record

    def record_event(
        self,
        *,
        event_type: str,
        severity: str = "info",
        request_id: str | None = None,
        detail: dict[str, object] | None = None,
    ) -> SystemEventRecord:
        """Requirement 12 (audit events): a generic, append-only event -
        engine startup/failure, a mapped error response, an auth/rate-limit
        rejection. `detail` must already be sanitized by the caller (see
        `db.models.SystemEventRecord`'s docstring) - this method does not
        inspect it."""
        with self._transaction():
            record = self._system_events.create(
                event_type=event_type, severity=severity, request_id=request_id, detail=detail
            )
        return record
