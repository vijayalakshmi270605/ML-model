"""Tests for `db.session` - connection pooling and the lazy engine cache.
No mocks: real `Engine` objects (never actually connected to a database) so
the real pool class/sizing each URL produces can be inspected."""

from __future__ import annotations

from sqlalchemy.pool import QueuePool

from ad_mri.core.config import Settings
from ad_mri.db import session as db_session_module
from ad_mri.db.session import build_engine, get_engine, get_sessionmaker, reset_engine_cache


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def test_sqlite_urls_use_a_pool_that_accepts_no_sizing_kwargs() -> None:
    """QueuePool-specific kwargs (`pool_size`, `max_overflow`, ...) are
    rejected outright by SQLite's usual pool class - this only passes if
    `build_engine` actually branched away from passing them."""
    settings = _settings(database_url="sqlite:///:memory:")
    engine = build_engine(settings)
    try:
        assert not isinstance(engine.pool, QueuePool)
    finally:
        engine.dispose()


def test_postgresql_urls_get_a_queuepool_sized_from_settings() -> None:
    settings = _settings(
        database_url="postgresql+psycopg2://user:pw@example-host:5432/ad_mri",
        db_pool_size=7,
        db_max_overflow=3,
    )
    engine = build_engine(settings)
    try:
        assert isinstance(engine.pool, QueuePool)
        assert engine.pool.size() == 7
        assert engine.pool._max_overflow == 3
    finally:
        engine.dispose()


def test_postgresql_engine_uses_pre_ping() -> None:
    settings = _settings(database_url="postgresql+psycopg2://user:pw@example-host:5432/ad_mri")
    engine = build_engine(settings)
    try:
        assert engine.pool._pre_ping is True
    finally:
        engine.dispose()


def test_get_engine_is_cached_across_calls() -> None:
    reset_engine_cache()
    try:
        first = get_engine()
        second = get_engine()
        assert first is second
    finally:
        reset_engine_cache()


def test_reset_engine_cache_disposes_and_forces_a_rebuild() -> None:
    reset_engine_cache()
    try:
        first = get_engine()
        reset_engine_cache()
        second = get_engine()
        assert first is not second
    finally:
        reset_engine_cache()


def test_get_sessionmaker_is_cached_and_bound_to_the_cached_engine() -> None:
    reset_engine_cache()
    try:
        factory = get_sessionmaker()
        assert factory.kw["bind"] is get_engine()
        assert get_sessionmaker() is factory
    finally:
        reset_engine_cache()


def test_reset_engine_cache_is_safe_to_call_before_any_engine_was_built() -> None:
    reset_engine_cache()
    assert db_session_module._engine is None
    assert db_session_module._session_factory is None
