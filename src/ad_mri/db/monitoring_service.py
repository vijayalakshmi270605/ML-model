"""Service layer for the two Phase 13 monitoring tables
(`input_statistics`, `ground_truth_labels`) plus a model version's
`baseline_metrics` - a sibling to `db.audit_service.AuditService`, kept
separate because it is a distinct concern (drift/quality monitoring, not
compliance/traceability) even though it shares the same database.

Same conventions as `AuditService`: one instance per `Session`, every public
method is one whole transaction (`_transaction()`), and every method is
**loud** - a database failure raises, it is never swallowed here. The
best-effort boundary for real-time (per-request) calls is `api.monitoring`,
one layer up, mirroring `api.audit`.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime

from sqlalchemy.orm import Session

from ad_mri.db.models import GroundTruthLabelRecord, InputStatisticsRecord, ModelVersionRecord
from ad_mri.db.repositories import (
    GroundTruthLabelRepository,
    InputStatisticsRepository,
    ModelVersionRepository,
    PredictionRequestRepository,
    PredictionResultRepository,
)


class MonitoringService:
    def __init__(self, session: Session) -> None:
        self._session = session
        self._input_statistics = InputStatisticsRepository(session)
        self._ground_truth = GroundTruthLabelRepository(session)
        self._model_versions = ModelVersionRepository(session)
        self._prediction_requests = PredictionRequestRepository(session)
        self._prediction_results = PredictionResultRepository(session)

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        try:
            yield
            self._session.commit()
        except Exception:
            self._session.rollback()
            raise

    def record_input_statistics(
        self,
        *,
        external_request_id: str,
        mean_intensity: float,
        std_intensity: float,
        p01_intensity: float,
        p50_intensity: float,
        p99_intensity: float,
        nonzero_fraction: float,
        voxel_count: int,
    ) -> InputStatisticsRecord | None:
        """Requirement (data drift): real-time per-request image statistics.
        Looks up the `PredictionRequestRecord` this belongs to by its
        caller-facing id - returns `None` (no-op, not an error) if that
        request was never recorded (e.g. `AUDIT_ENABLED=false`), since a
        statistics row with no request to join against is meaningless."""
        with self._transaction():
            request = self._prediction_requests.get_by_external_request_id(external_request_id)
            if request is None:
                return None
            record = self._input_statistics.create(
                request_id=request.id,
                mean_intensity=mean_intensity,
                std_intensity=std_intensity,
                p01_intensity=p01_intensity,
                p50_intensity=p50_intensity,
                p99_intensity=p99_intensity,
                nonzero_fraction=nonzero_fraction,
                voxel_count=voxel_count,
            )
        return record

    def record_ground_truth(
        self, *, external_request_id: str, true_class: str, label_source: str
    ) -> GroundTruthLabelRecord:
        """Requirement (delayed model-performance monitoring): attach a true
        class to a past prediction, whenever it actually becomes available.

        Raises:
            ValueError: no `PredictionRequestRecord` exists for
                `external_request_id` - a ground-truth label must reference a
                real, already-recorded prediction.
        """
        with self._transaction():
            request = self._prediction_requests.get_by_external_request_id(external_request_id)
            if request is None:
                raise ValueError(
                    f"No recorded prediction with request_id={external_request_id!r} - "
                    "cannot attach a ground-truth label to a prediction that was never audited."
                )
            record = self._ground_truth.create_or_update(
                request_id=request.id, true_class=true_class, label_source=label_source
            )
        return record

    def set_baseline_metrics(
        self, model_version_id: uuid.UUID, metrics: dict[str, float]
    ) -> ModelVersionRecord:
        with self._transaction():
            record = self._model_versions.set_baseline_metrics(model_version_id, metrics)
        return record

    def get_labeled_predictions(
        self, model_version_id: uuid.UUID, since: datetime | None = None, limit: int = 10_000
    ) -> Sequence[tuple[str, str]]:
        """Read-only - no transaction needed. Returns `(predicted_class,
        true_class)` pairs, the direct input to
        `monitoring.model_drift.compute_delayed_performance`."""
        rows = self._ground_truth.list_labeled_predictions_for_model_version(
            model_version_id, since=since, limit=limit
        )
        return [(row.predicted_class, row.true_class) for row in rows]

    def get_input_statistics_window(
        self, start: datetime | None = None, end: datetime | None = None, limit: int = 100_000
    ) -> Sequence[InputStatisticsRecord]:
        """Read-only - no transaction needed."""
        return self._input_statistics.list_between(start=start, end=end, limit=limit)

    def get_predicted_class_counts(
        self, start: datetime | None = None, end: datetime | None = None
    ) -> dict[str, int]:
        """Read-only - no transaction needed. The current-window half of
        `monitoring.drift`'s class-distribution PSI check."""
        return self._prediction_results.count_by_predicted_class(start=start, end=end)

    def get_model_version(self, model_version_id: uuid.UUID) -> ModelVersionRecord | None:
        """Read-only - no transaction needed. Used by
        `ad-mri monitor model-drift-report` to fetch `baseline_metrics`."""
        return self._model_versions.get(model_version_id)
