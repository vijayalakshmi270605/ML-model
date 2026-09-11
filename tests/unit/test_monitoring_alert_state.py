"""Tests for `monitoring.alert_state` - real DB writes/reads (SQLite) and
real Prometheus gauge assertions, no mocks."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from prometheus_client import REGISTRY

from ad_mri.core.config import get_settings
from ad_mri.db.audit_service import AuditService
from ad_mri.db.base import Base
from ad_mri.db.session import get_engine, reset_engine_cache
from ad_mri.monitoring.alert_state import sample_alert_state
from ad_mri.monitoring.drift import FEATURE_NAMES
from ad_mri.monitoring.metrics import DRIFT_ALERT_ACTIVE, MODEL_DRIFT_ALERT_ACTIVE
from ad_mri.monitoring.model_drift import MONITORED_METRICS


def _gauge_value(gauge, **labels) -> float | None:  # type: ignore[no-untyped-def]
    return REGISTRY.get_sample_value(gauge._name, labels)


@pytest.fixture
def real_sqlite_database(monkeypatch: pytest.MonkeyPatch, tmp_path):  # type: ignore[no-untyped-def]
    db_path = tmp_path / "alert_state.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    get_settings.cache_clear()
    reset_engine_cache()
    Base.metadata.create_all(get_engine())
    yield
    reset_engine_cache()
    get_settings.cache_clear()


def test_no_events_means_every_gauge_is_zero(real_sqlite_database: None) -> None:
    sample_alert_state()
    for feature in (*FEATURE_NAMES, "class_distribution"):
        assert _gauge_value(DRIFT_ALERT_ACTIVE, feature=feature) == 0
    for metric in MONITORED_METRICS:
        assert _gauge_value(MODEL_DRIFT_ALERT_ACTIVE, metric=metric) == 0


def test_a_recent_data_drift_event_sets_the_matching_gauges(real_sqlite_database: None) -> None:
    from ad_mri.db.session import get_sessionmaker

    session = get_sessionmaker()()
    AuditService(session).record_event(
        event_type="data_drift_detected",
        severity="warning",
        detail={"drifted_features": ["mean_intensity"], "class_distribution_drifted": True},
    )

    sample_alert_state()

    assert _gauge_value(DRIFT_ALERT_ACTIVE, feature="mean_intensity") == 1
    assert _gauge_value(DRIFT_ALERT_ACTIVE, feature="std_intensity") == 0
    assert _gauge_value(DRIFT_ALERT_ACTIVE, feature="class_distribution") == 1


def test_a_recent_model_drift_event_sets_the_matching_gauge(real_sqlite_database: None) -> None:
    from ad_mri.db.session import get_sessionmaker

    session = get_sessionmaker()()
    AuditService(session).record_event(
        event_type="model_drift_detected",
        severity="critical",
        detail={"model_version_id": "v1", "degraded_metrics": ["accuracy"]},
    )

    sample_alert_state()

    assert _gauge_value(MODEL_DRIFT_ALERT_ACTIVE, metric="accuracy") == 1
    assert _gauge_value(MODEL_DRIFT_ALERT_ACTIVE, metric="f1_macro") == 0


def test_a_stale_event_is_treated_as_no_active_alert(real_sqlite_database: None) -> None:
    from ad_mri.db.repositories import SystemEventRepository
    from ad_mri.db.session import get_sessionmaker

    session = get_sessionmaker()()
    repo = SystemEventRepository(session)
    record = repo.create(
        event_type="data_drift_detected",
        severity="warning",
        detail={"drifted_features": ["mean_intensity"]},
    )
    # Backdate it past the 24h recency window - directly, since
    # created_at is a server_default the repository layer never lets a
    # caller set explicitly.
    record.created_at = datetime.now(UTC) - timedelta(hours=48)
    session.commit()

    sample_alert_state()

    assert _gauge_value(DRIFT_ALERT_ACTIVE, feature="mean_intensity") == 0


def test_never_raises_when_the_database_is_unreachable(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    bad_path = tmp_path / "missing-dir" / "db.sqlite"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{bad_path}")
    get_settings.cache_clear()
    reset_engine_cache()
    try:
        sample_alert_state()  # must not raise
    finally:
        reset_engine_cache()
        get_settings.cache_clear()
