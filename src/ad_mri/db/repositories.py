"""Repository layer: typed CRUD around each ORM model.

No business logic and no transaction management lives here - every method
only `add()`s/`get()`s/queries against the `Session` it was given and, for a
write, `flush()`es (so a caller can immediately read back a generated
default like an id) without `commit()`ing. The caller
(`db.audit_service.AuditService`/`db.monitoring_service.MonitoringService`)
owns the transaction boundary: one service method equals one transaction,
committed or rolled back as a whole - never a sequence of independently-
committed repository calls that could leave the audit trail half-written.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ad_mri.db.models import (
    ExplanationRecord,
    GroundTruthLabelRecord,
    InputStatisticsRecord,
    ModelVersionRecord,
    PredictionRequestRecord,
    PredictionResultRecord,
    SystemEventRecord,
)
from ad_mri.db.privacy import ensure_redacted_image_reference


class ModelVersionRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, model_version_id: uuid.UUID) -> ModelVersionRecord | None:
        return self._session.get(ModelVersionRecord, model_version_id)

    def set_baseline_metrics(
        self, model_version_id: uuid.UUID, metrics: dict[str, float]
    ) -> ModelVersionRecord:
        """Requirement (delayed model-performance monitoring): the fixed
        comparison point `monitoring.model_drift` measures production
        performance against - typically this version's own held-out
        evaluation metrics, set once at registration time."""
        record = self._session.get(ModelVersionRecord, model_version_id)
        if record is None:
            raise ValueError(f"No model_versions row with id={model_version_id}")
        record.baseline_metrics = metrics
        self._session.flush()
        return record

    def get_by_config_hash(
        self, model_config_hash: str, preprocessing_config_hash: str
    ) -> ModelVersionRecord | None:
        stmt = select(ModelVersionRecord).where(
            ModelVersionRecord.model_config_hash == model_config_hash,
            ModelVersionRecord.preprocessing_config_hash == preprocessing_config_hash,
        )
        return self._session.execute(stmt).scalar_one_or_none()

    def create(
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
        record = ModelVersionRecord(
            architecture=architecture,
            model_version=model_version,
            model_config_hash=model_config_hash,
            preprocessing_version=preprocessing_version,
            preprocessing_config_hash=preprocessing_config_hash,
            checkpoint_uri=checkpoint_uri,
            stage=stage,
            device=device,
        )
        self._session.add(record)
        self._session.flush()
        return record

    def get_or_create(
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
        """Idempotent registration keyed by (model_config_hash,
        preprocessing_config_hash) - a version is registered once, the first
        time it is used, not re-inserted on every prediction."""
        existing = self.get_by_config_hash(model_config_hash, preprocessing_config_hash)
        if existing is not None:
            return existing
        return self.create(
            architecture=architecture,
            model_version=model_version,
            model_config_hash=model_config_hash,
            preprocessing_version=preprocessing_version,
            preprocessing_config_hash=preprocessing_config_hash,
            checkpoint_uri=checkpoint_uri,
            stage=stage,
            device=device,
        )

    def list_recent(self, limit: int = 50) -> Sequence[ModelVersionRecord]:
        stmt = (
            select(ModelVersionRecord)
            .order_by(ModelVersionRecord.first_seen_at.desc())
            .limit(limit)
        )
        return self._session.execute(stmt).scalars().all()


class PredictionRequestRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self,
        *,
        external_request_id: str,
        model_version_id: uuid.UUID | None,
        image_ref: str,
        content_type: str | None,
        api_key_hash: str | None,
    ) -> PredictionRequestRecord:
        ensure_redacted_image_reference(image_ref)
        record = PredictionRequestRecord(
            external_request_id=external_request_id,
            model_version_id=model_version_id,
            image_ref=image_ref,
            content_type=content_type,
            api_key_hash=api_key_hash,
        )
        self._session.add(record)
        self._session.flush()
        return record

    def get(self, request_id: uuid.UUID) -> PredictionRequestRecord | None:
        return self._session.get(PredictionRequestRecord, request_id)

    def get_by_external_request_id(
        self, external_request_id: str
    ) -> PredictionRequestRecord | None:
        stmt = select(PredictionRequestRecord).where(
            PredictionRequestRecord.external_request_id == external_request_id
        )
        return self._session.execute(stmt).scalar_one_or_none()

    def list_recent(self, limit: int = 50) -> Sequence[PredictionRequestRecord]:
        stmt = (
            select(PredictionRequestRecord)
            .order_by(PredictionRequestRecord.received_at.desc())
            .limit(limit)
        )
        return self._session.execute(stmt).scalars().all()


class PredictionResultRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self,
        *,
        request_id: uuid.UUID,
        predicted_class: str,
        confidence: float,
        probabilities: dict[str, float],
        processing_time_ms: float,
        explanation_available: bool,
    ) -> PredictionResultRecord:
        record = PredictionResultRecord(
            request_id=request_id,
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities=probabilities,
            processing_time_ms=processing_time_ms,
            explanation_available=explanation_available,
        )
        self._session.add(record)
        self._session.flush()
        return record

    def get_by_request_id(self, request_id: uuid.UUID) -> PredictionResultRecord | None:
        stmt = select(PredictionResultRecord).where(PredictionResultRecord.request_id == request_id)
        return self._session.execute(stmt).scalar_one_or_none()

    def count_by_predicted_class(
        self, start: datetime | None = None, end: datetime | None = None
    ) -> dict[str, int]:
        """`{predicted_class: count}` over a time window - the current-window
        half of `monitoring.drift`'s class-distribution PSI check (the
        reference half is `DriftReference.class_counts`)."""
        from sqlalchemy import func

        stmt = (
            select(PredictionResultRecord.predicted_class, func.count())
            .join(
                PredictionRequestRecord,
                PredictionResultRecord.request_id == PredictionRequestRecord.id,
            )
            .group_by(PredictionResultRecord.predicted_class)
        )
        if start is not None:
            stmt = stmt.where(PredictionRequestRecord.received_at >= start)
        if end is not None:
            stmt = stmt.where(PredictionRequestRecord.received_at <= end)
        rows = self._session.execute(stmt).all()
        return {predicted_class: count for predicted_class, count in rows}


class ExplanationRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self,
        *,
        external_explanation_id: str,
        request_id: uuid.UUID | None,
        model_version_id: uuid.UUID | None,
        methods: str,
        target_class: str | None,
        artifact_uri: str,
        params_hash: str,
    ) -> ExplanationRecord:
        record = ExplanationRecord(
            external_explanation_id=external_explanation_id,
            request_id=request_id,
            model_version_id=model_version_id,
            methods=methods,
            target_class=target_class,
            artifact_uri=artifact_uri,
            params_hash=params_hash,
        )
        self._session.add(record)
        self._session.flush()
        return record

    def get(self, explanation_id: uuid.UUID) -> ExplanationRecord | None:
        return self._session.get(ExplanationRecord, explanation_id)


class SystemEventRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self,
        *,
        event_type: str,
        severity: str = "info",
        request_id: str | None = None,
        detail: dict[str, object] | None = None,
    ) -> SystemEventRecord:
        record = SystemEventRecord(
            event_type=event_type, severity=severity, request_id=request_id, detail=detail
        )
        self._session.add(record)
        self._session.flush()
        return record

    def list_recent(
        self, limit: int = 100, event_type: str | None = None
    ) -> Sequence[SystemEventRecord]:
        stmt = select(SystemEventRecord).order_by(SystemEventRecord.created_at.desc())
        if event_type is not None:
            stmt = stmt.where(SystemEventRecord.event_type == event_type)
        return self._session.execute(stmt.limit(limit)).scalars().all()


class InputStatisticsRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self,
        *,
        request_id: uuid.UUID,
        mean_intensity: float,
        std_intensity: float,
        p01_intensity: float,
        p50_intensity: float,
        p99_intensity: float,
        nonzero_fraction: float,
        voxel_count: int,
    ) -> InputStatisticsRecord:
        record = InputStatisticsRecord(
            request_id=request_id,
            mean_intensity=mean_intensity,
            std_intensity=std_intensity,
            p01_intensity=p01_intensity,
            p50_intensity=p50_intensity,
            p99_intensity=p99_intensity,
            nonzero_fraction=nonzero_fraction,
            voxel_count=voxel_count,
        )
        self._session.add(record)
        self._session.flush()
        return record

    def list_between(
        self, start: datetime | None = None, end: datetime | None = None, limit: int = 100_000
    ) -> Sequence[InputStatisticsRecord]:
        """The "current window" `monitoring.drift.generate_data_drift_report`
        compares against a `DriftReference` - unbounded on either side by
        default (the caller decides the window; this only applies it)."""
        stmt = select(InputStatisticsRecord).order_by(InputStatisticsRecord.recorded_at.desc())
        if start is not None:
            stmt = stmt.where(InputStatisticsRecord.recorded_at >= start)
        if end is not None:
            stmt = stmt.where(InputStatisticsRecord.recorded_at <= end)
        return self._session.execute(stmt.limit(limit)).scalars().all()


class GroundTruthLabelRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get_by_request_id(self, request_id: uuid.UUID) -> GroundTruthLabelRecord | None:
        stmt = select(GroundTruthLabelRecord).where(GroundTruthLabelRecord.request_id == request_id)
        return self._session.execute(stmt).scalar_one_or_none()

    def create_or_update(
        self, *, request_id: uuid.UUID, true_class: str, label_source: str
    ) -> GroundTruthLabelRecord:
        """A correction (e.g. an initial follow-up read later revised)
        overwrites the existing row for this request rather than
        accumulating a second, conflicting label."""
        existing = self.get_by_request_id(request_id)
        if existing is not None:
            existing.true_class = true_class
            existing.label_source = label_source
            self._session.flush()
            return existing
        record = GroundTruthLabelRecord(
            request_id=request_id, true_class=true_class, label_source=label_source
        )
        self._session.add(record)
        self._session.flush()
        return record

    def list_labeled_predictions_for_model_version(
        self, model_version_id: uuid.UUID, since: datetime | None = None, limit: int = 10_000
    ) -> Sequence[Any]:
        """`(predicted_class, true_class)` pairs for every prediction made by
        `model_version_id` that has since received a ground-truth label -
        exactly the join `monitoring.model_drift` needs to compute delayed
        performance metrics, done once in the database rather than three
        separate round trips and an application-level join."""
        stmt = (
            select(PredictionResultRecord.predicted_class, GroundTruthLabelRecord.true_class)
            .join(
                PredictionRequestRecord,
                PredictionResultRecord.request_id == PredictionRequestRecord.id,
            )
            .join(
                GroundTruthLabelRecord,
                GroundTruthLabelRecord.request_id == PredictionRequestRecord.id,
            )
            .where(PredictionRequestRecord.model_version_id == model_version_id)
        )
        if since is not None:
            stmt = stmt.where(GroundTruthLabelRecord.recorded_at >= since)
        stmt = stmt.order_by(GroundTruthLabelRecord.recorded_at.desc()).limit(limit)
        return self._session.execute(stmt).all()
