"""Reflects the most recent *offline* drift-report verdict into the live
`DRIFT_ALERT_ACTIVE`/`MODEL_DRIFT_ALERT_ACTIVE` Prometheus gauges.

Same pull-based pattern `monitoring.system_metrics` uses, for the same
reason: `ad-mri monitor drift-report`/`model-drift-report` are short-lived
batch-job processes that are never themselves scraped, so their verdict has
to be durably recorded somewhere a long-running process (the API) can read
it back from - `system_events` (`db.audit_service.AuditService.record_event`,
written by those CLI commands), read here on every `/metrics` scrape and
mirrored into the two gauges.

A verdict older than `_RECENCY_WINDOW` is treated as stale and cleared back
to "no active alert" - a drift report that was never re-run recently
shouldn't leave a permanent, increasingly-misleading red light on a
dashboard.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from ad_mri.core.logging import get_logger
from ad_mri.monitoring.drift import FEATURE_NAMES
from ad_mri.monitoring.metrics import DRIFT_ALERT_ACTIVE, MODEL_DRIFT_ALERT_ACTIVE
from ad_mri.monitoring.model_drift import MONITORED_METRICS

if TYPE_CHECKING:
    from ad_mri.db.models import SystemEventRecord

logger = get_logger(__name__)

_RECENCY_WINDOW = timedelta(hours=24)
_DATA_DRIFT_EVENT_TYPE = "data_drift_detected"
_MODEL_DRIFT_EVENT_TYPE = "model_drift_detected"


def sample_alert_state() -> None:
    """Never raises - a failed sample is a gap in the series, not an
    application error (mirrors `system_metrics.sample_system_metrics`)."""
    try:
        _sample()
    except Exception as exc:  # noqa: BLE001 - a metrics scrape must never fail the request
        logger.warning("alert_state_sample_failed", error_type=type(exc).__name__)


def _sample() -> None:
    from ad_mri.db.repositories import SystemEventRepository
    from ad_mri.db.session import get_sessionmaker

    session = get_sessionmaker()()
    try:
        repo = SystemEventRepository(session)
        latest_data_event = _most_recent_within_window(
            repo.list_recent(limit=1, event_type=_DATA_DRIFT_EVENT_TYPE)
        )
        latest_model_event = _most_recent_within_window(
            repo.list_recent(limit=1, event_type=_MODEL_DRIFT_EVENT_TYPE)
        )
    finally:
        session.close()

    drifted_features: set[str] = set()
    class_distribution_drifted = False
    if latest_data_event is not None and latest_data_event.detail:
        raw_features = latest_data_event.detail.get("drifted_features", [])
        if isinstance(raw_features, list):
            drifted_features = set(raw_features)
        class_distribution_drifted = bool(
            latest_data_event.detail.get("class_distribution_drifted")
        )
    for feature in FEATURE_NAMES:
        DRIFT_ALERT_ACTIVE.labels(feature).set(1 if feature in drifted_features else 0)
    DRIFT_ALERT_ACTIVE.labels("class_distribution").set(1 if class_distribution_drifted else 0)

    degraded_metrics: set[str] = set()
    if latest_model_event is not None and latest_model_event.detail:
        raw_metrics = latest_model_event.detail.get("degraded_metrics", [])
        if isinstance(raw_metrics, list):
            degraded_metrics = set(raw_metrics)
    for metric in MONITORED_METRICS:
        MODEL_DRIFT_ALERT_ACTIVE.labels(metric).set(1 if metric in degraded_metrics else 0)


def _most_recent_within_window(
    events: Sequence[SystemEventRecord],
) -> SystemEventRecord | None:
    if not events:
        return None
    latest = events[0]
    created_at = latest.created_at
    # SQLite (used by this project's own tests, see docs/database.md §3)
    # returns a naive datetime for a `DateTime(timezone=True)` column even
    # though PostgreSQL returns a real tz-aware one - assume UTC (what
    # `server_default=func.now()` actually wrote) rather than let a naive-
    # vs-aware comparison raise.
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    if created_at < datetime.now(UTC) - _RECENCY_WINDOW:
        return None
    return latest
