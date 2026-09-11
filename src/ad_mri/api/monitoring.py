"""Best-effort persistence for the Phase 13 monitoring tables
(`input_statistics`, `ground_truth_labels`) - the API-layer boundary for
`db.monitoring_service.MonitoringService`, mirroring `api.audit` exactly:
each function opens its own short-lived `Session`, calls the loud service
layer, and swallows any failure (logging `error_type` only) rather than
letting an unreachable monitoring database fail a real request. See
`api.audit`'s module docstring for the full rationale - it applies here
unchanged.

`record_ground_truth` is the one exception to "never raises": it is called
directly (not via `asyncio.to_thread`-wrapped fire-and-forget) from
`POST /api/v1/monitoring/ground-truth`, whose entire purpose is recording a
label - a failure there must be a real 4xx/5xx to the caller, not a silent
no-op that leaves them believing the label was recorded when it wasn't. See
that router for how the two failure modes it can raise
(`ValueError` for an unknown request, anything else for "the database
itself is unreachable") are mapped to HTTP status codes.
"""

from __future__ import annotations

from ad_mri.core.config import get_settings
from ad_mri.core.logging import get_logger
from ad_mri.db.monitoring_service import MonitoringService
from ad_mri.db.session import get_sessionmaker
from ad_mri.monitoring.drift import ImageStatistics

logger = get_logger(__name__)


def record_input_statistics(*, external_request_id: str, image_statistics: ImageStatistics) -> None:
    """Fire-and-forget: persists one request's `ImageStatistics` (Phase 13
    data drift). No-ops when `Settings.audit_enabled` is False (the same
    flag `api.audit` uses - one switch turns off all optional DB writes)."""
    settings = get_settings()
    if not settings.audit_enabled:
        return
    session = get_sessionmaker()()
    try:
        MonitoringService(session).record_input_statistics(
            external_request_id=external_request_id,
            mean_intensity=image_statistics.mean_intensity,
            std_intensity=image_statistics.std_intensity,
            p01_intensity=image_statistics.p01_intensity,
            p50_intensity=image_statistics.p50_intensity,
            p99_intensity=image_statistics.p99_intensity,
            nonzero_fraction=image_statistics.nonzero_fraction,
            voxel_count=image_statistics.voxel_count,
        )
    except Exception as exc:  # noqa: BLE001 - an audit-write failure must never fail the request
        logger.warning("monitoring_record_input_statistics_failed", error_type=type(exc).__name__)
    finally:
        session.close()


def record_ground_truth(*, external_request_id: str, true_class: str, label_source: str) -> None:
    """**Loud** (unlike every other function in this module): the entire
    point of the caller (`POST /api/v1/monitoring/ground-truth`) is
    recording this label, so a failure must reach the caller as a real
    error, not be swallowed. Opens its own session, same as the
    fire-and-forget functions above - not `api.deps.get_db`, so a request
    handler that also needs a request-scoped session for something else is
    unaffected by this one's lifecycle.

    Raises:
        ValueError: no recorded prediction exists for `external_request_id`.
        Exception: the monitoring database itself is unreachable, or any
            other real failure - propagated unchanged.
    """
    session = get_sessionmaker()()
    try:
        MonitoringService(session).record_ground_truth(
            external_request_id=external_request_id,
            true_class=true_class,
            label_source=label_source,
        )
    finally:
        session.close()
