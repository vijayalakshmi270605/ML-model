"""SQLAlchemy ORM models for the prediction/explanation audit trail and the
monitoring data it feeds.

The five Phase 11 tables - see docs/database.md and
docs/TECHNICAL_SPECIFICATION.md §12:

- `model_versions`      - which checkpoint served a prediction/explanation
                           (Phase 13 adds `baseline_metrics`, see below).
- `prediction_requests` - one row per inbound request (minted at receipt).
- `prediction_results`  - one row per completed prediction, 1:1 with its request.
- `explanation_records` - one row per generated LIME/SHAP explanation.
- `system_events`       - a generic, append-only audit log for security- and
                           operations-relevant occurrences not tied to any
                           one prediction (engine startup, auth failures, ...) -
                           also where a detected drift alert (Phase 13,
                           `monitoring.drift`/`monitoring.model_drift`) is
                           durably recorded.

Two more, added in Phase 13 for monitoring - see docs/monitoring.md:

- `input_statistics`    - per-request preprocessed-image statistics (mean/std/
                           percentile intensity, nonzero fraction) - the real-
                           time data-drift signal, computed on the volume the
                           model actually sees, never the raw file.
- `ground_truth_labels` - a true class attached to a *past* prediction once it
                           becomes available (requirement: "do not assume
                           ground-truth labels are immediately available in
                           production") - the delayed-model-performance-
                           monitoring input.

Privacy is enforced at this layer, not just by convention: nothing here has a
column wide enough for a raw MRI file, and `image_ref` is validated (in
`db.repositories`, via `db.privacy.ensure_redacted_image_reference`) to be a
one-way hash before it can ever be written. Large artifacts (explanation
manifests, model checkpoints) are referenced by URI (`storage.object_storage`) - their
bytes never enter PostgreSQL. `input_statistics` holds only aggregate,
non-identifying voxel statistics (a mean/std/percentile is not
re-identifiable) - never the array itself. See `db.privacy` for the
enforcement layer.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ad_mri.db.base import Base
from ad_mri.db.types import GUID


class ModelVersionRecord(Base):
    """One row per distinct (model_config_hash, preprocessing_config_hash)
    pair ever served - a durable, DB-backed answer to "which model versions
    has this deployment ever run predictions with", independent of (and not
    a replacement for) the MLflow Model Registry (`model_registry.client`),
    which is not yet wired to a real tracking server. Registered once, via
    `db.audit_service.AuditService.record_model_version`'s get-or-create, the
    first time a version is used - not on every prediction.
    """

    __tablename__ = "model_versions"
    __table_args__ = (
        UniqueConstraint(
            "model_config_hash",
            "preprocessing_config_hash",
            name="uq_model_versions_config_hash",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    architecture: Mapped[str] = mapped_column(String(64), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    model_config_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    preprocessing_version: Mapped[str] = mapped_column(String(64), nullable=False)
    preprocessing_config_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    #: An object-storage/filesystem URI (see `storage.object_storage`) pointing at the
    #: checkpoint - never the weights themselves.
    checkpoint_uri: Mapped[str] = mapped_column(String(512), nullable=False)
    stage: Mapped[str] = mapped_column(String(16), nullable=False, default="None")
    device: Mapped[str] = mapped_column(String(16), nullable=False)
    #: `{metric_name: value}` from this version's own held-out evaluation
    #: (`evaluation.metrics.compute_classification_metrics`'s output, or a
    #: subset of it) at training/registration time - the comparison baseline
    #: `monitoring.model_drift` measures delayed, ground-truth-backed
    #: production performance against. `None` until explicitly set (e.g. via
    #: `ad-mri monitor register-baseline`) - a version with no baseline can
    #: still serve predictions, it just can't have model-drift computed yet.
    baseline_metrics: Mapped[dict[str, float] | None] = mapped_column(JSON, nullable=True)
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    prediction_requests: Mapped[list[PredictionRequestRecord]] = relationship(
        back_populates="model_version"
    )
    explanations: Mapped[list[ExplanationRecord]] = relationship(back_populates="model_version")


class PredictionRequestRecord(Base):
    """One row per inbound `POST /api/v1/predictions` (or `/explanations`,
    which also produces a prediction) call.

    `id` is a server-generated UUID (requirement 5) - deliberately *not* the
    same value as the caller-facing `X-Request-ID`/`PredictionResult.request_id`,
    because that value can be supplied by the caller (`RequestIDMiddleware`
    honors an inbound `X-Request-ID` header verbatim) and is therefore not
    guaranteed to even be UUID-shaped. `external_request_id` holds that
    original value instead, indexed for log/response correlation, without
    forcing every column that references "the request" in this schema to
    tolerate arbitrary caller-supplied text.
    """

    __tablename__ = "prediction_requests"
    __table_args__ = (
        Index("ix_prediction_requests_received_at", "received_at"),
        Index("ix_prediction_requests_external_request_id", "external_request_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    #: The `request_id` a client sees (response body, `X-Request-ID` header,
    #: log lines) - free text, not this row's identity.
    external_request_id: Mapped[str] = mapped_column(String(128), nullable=False)
    model_version_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("model_versions.id"), nullable=True
    )
    #: `inference.predictor.redact_image_reference()` output - a one-way
    #: hash, never the uploaded file's path or name. Enforced by
    #: `db.repositories.PredictionRequestRepository.create`.
    image_ref: Mapped[str] = mapped_column(String(16), nullable=False)
    content_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    #: sha256 of the presented API key (`db.privacy.hash_value`) - never the
    #: key itself. `None` when the deployment does not require one.
    api_key_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    model_version: Mapped[ModelVersionRecord | None] = relationship(
        back_populates="prediction_requests"
    )
    result: Mapped[PredictionResultRecord | None] = relationship(
        back_populates="request", uselist=False
    )
    explanations: Mapped[list[ExplanationRecord]] = relationship(back_populates="request")


class PredictionResultRecord(Base):
    """One row per *completed* prediction - absent for a request that failed
    before a result existed (see `system_events` for that case instead).
    1:1 with its `PredictionRequestRecord` (`request_id` is unique)."""

    __tablename__ = "prediction_results"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    request_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("prediction_requests.id"), nullable=False, unique=True
    )
    predicted_class: Mapped[str] = mapped_column(String(64), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    #: `{class_name: probability}` - a handful of floats, not the volume.
    probabilities: Mapped[dict[str, float]] = mapped_column(JSON, nullable=False)
    processing_time_ms: Mapped[float] = mapped_column(Float, nullable=False)
    explanation_available: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    request: Mapped[PredictionRequestRecord] = relationship(back_populates="result")


class ExplanationRecord(Base):
    """One row per `POST /api/v1/explanations` call that actually produced an
    explanation.

    `id` is a server-generated UUID (requirement 5) - not
    `ExplanationBundle.input_id` directly, for the same reason
    `PredictionRequestRecord.id` is not the caller-facing request id: that
    value can trace back to a caller-supplied `X-Request-ID` header (or, if
    neither was ever set, to `image_path.stem`) and is not guaranteed to be
    UUID-shaped. `external_explanation_id` holds the value the API actually
    returns as `explanation_id` instead. `request_id` stays nullable and
    unset for a standalone `/explanations` call, which never creates its own
    `PredictionRequestRecord` - a foreign key here must only ever point at a
    row that really exists."""

    __tablename__ = "explanation_records"
    __table_args__ = (
        Index("ix_explanation_records_created_at", "created_at"),
        Index("ix_explanation_records_external_explanation_id", "external_explanation_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    #: The `explanation_id` a client sees - free text, not this row's identity.
    external_explanation_id: Mapped[str] = mapped_column(String(128), nullable=False)
    request_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("prediction_requests.id"), nullable=True
    )
    model_version_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("model_versions.id"), nullable=True
    )
    #: Comma-separated method names actually produced, e.g. `"lime"` or
    #: `"lime,shap"` - mirrors `ExplanationResponse.methods`.
    methods: Mapped[str] = mapped_column(String(32), nullable=False)
    target_class: Mapped[str | None] = mapped_column(String(64), nullable=True)
    #: An object-storage/filesystem URI (see `storage.object_storage`) pointing at the
    #: explanation's manifest.json - never the saliency arrays themselves.
    artifact_uri: Mapped[str] = mapped_column(String(512), nullable=False)
    params_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    request: Mapped[PredictionRequestRecord | None] = relationship(back_populates="explanations")
    model_version: Mapped[ModelVersionRecord | None] = relationship(back_populates="explanations")


class SystemEventRecord(Base):
    """A generic, append-only audit-event log for occurrences that are not
    themselves a prediction/explanation - engine startup/failure, a mapped
    domain-exception response, an auth failure, a rate-limit rejection.
    `request_id` is a bare, free-text column (not a foreign key and not a
    `GUID`): many events (process startup) have no request at all, an event
    must still be recordable for a request whose own row failed to write,
    and - as with `PredictionRequestRecord.external_request_id` - the
    caller-facing request id is not guaranteed to be UUID-shaped.

    `detail` must only ever hold already-sanitized, small, structured data
    (status codes, error-type names, counts) - never a raw exception message,
    file path, or secret. Nothing enforces that here beyond convention and
    review, the same boundary `core.logging`'s module docstring already
    documents for every other structured log call in this codebase.
    """

    __tablename__ = "system_events"
    __table_args__ = (Index("ix_system_events_event_type_created_at", "event_type", "created_at"),)

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False, default="info")
    request_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    detail: Mapped[dict[str, object] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class InputStatisticsRecord(Base):
    """One row per completed prediction - aggregate statistics of the
    *preprocessed* volume the model actually saw (never the raw file, never
    the array itself). The real-time half of data-drift monitoring
    (`monitoring.drift`): a rolling window of these rows, compared against a
    `monitoring.drift.DriftReference` built from the training distribution,
    is what `ad-mri monitor drift-report` reads."""

    __tablename__ = "input_statistics"
    __table_args__ = (Index("ix_input_statistics_recorded_at", "recorded_at"),)

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    request_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("prediction_requests.id"), nullable=False, unique=True
    )
    mean_intensity: Mapped[float] = mapped_column(Float, nullable=False)
    std_intensity: Mapped[float] = mapped_column(Float, nullable=False)
    p01_intensity: Mapped[float] = mapped_column(Float, nullable=False)
    p50_intensity: Mapped[float] = mapped_column(Float, nullable=False)
    p99_intensity: Mapped[float] = mapped_column(Float, nullable=False)
    nonzero_fraction: Mapped[float] = mapped_column(Float, nullable=False)
    voxel_count: Mapped[int] = mapped_column(BigInteger, nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class GroundTruthLabelRecord(Base):
    """A true class attached to a *past* prediction, recorded whenever it
    actually becomes available - hours, days, or weeks after the prediction
    itself (requirement: never assume it arrives immediately). One row per
    `PredictionRequestRecord` (`request_id` is unique - a correction
    overwrites via `AuditService`-style get by request then update, it does
    not accumulate duplicate labels for one request).

    `label_source` is a small, fixed vocabulary describing *how* the label
    was obtained (e.g. `"clinician_review"`, `"follow_up_diagnosis"`) - never
    a free-text field, which would risk becoming a place PHI accidentally
    gets typed in."""

    __tablename__ = "ground_truth_labels"
    __table_args__ = (Index("ix_ground_truth_labels_recorded_at", "recorded_at"),)

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    request_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("prediction_requests.id"), nullable=False, unique=True
    )
    true_class: Mapped[str] = mapped_column(String(64), nullable=False)
    label_source: Mapped[str] = mapped_column(String(64), nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
