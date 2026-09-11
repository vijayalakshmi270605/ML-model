"""SQLAlchemy engine/session management.

The engine is created lazily (on first use, not at import time) so importing
this module never requires a reachable database - connections are only
attempted when a request actually needs one. See docs/database.md.
"""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from ad_mri.core.config import Settings, get_settings

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def build_engine(settings: Settings) -> Engine:
    """Construct an `Engine` for `settings.get_database_url()`.

    Connection pooling (`pool_size`/`max_overflow`/`pool_timeout`/
    `pool_recycle`) is QueuePool-specific - SQLAlchemy's default pool for
    PostgreSQL, but not one SQLite (used by the test suite in place of a
    locally-running PostgreSQL) accepts, so those kwargs are only passed for
    non-SQLite URLs. `pool_pre_ping=True` in both cases: a connection that
    went stale (e.g. a database restart, an idle timeout) is detected and
    transparently replaced rather than surfacing as an opaque error on a
    request that happened to draw it from the pool.
    """
    url = make_url(settings.get_database_url())
    if url.get_backend_name() == "sqlite":
        return create_engine(url, pool_pre_ping=True, future=True)
    return create_engine(
        url,
        pool_pre_ping=True,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_timeout=settings.db_pool_timeout,
        pool_recycle=settings.db_pool_recycle,
        connect_args={"connect_timeout": settings.db_connect_timeout_seconds},
        future=True,
    )


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = build_engine(get_settings())
    return _engine


def get_sessionmaker() -> sessionmaker[Session]:
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(
            bind=get_engine(), autoflush=False, autocommit=False, future=True
        )
    return _session_factory


def get_db_session() -> Generator[Session, None, None]:
    """FastAPI-dependency-friendly generator yielding a request-scoped session."""
    session_factory = get_sessionmaker()
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


def reset_engine_cache() -> None:
    """Test-only: forces the next `get_engine()`/`get_sessionmaker()` call to
    rebuild - mirrors `core.config.get_settings.cache_clear()`. Disposes the
    old engine first so no pooled connection outlives it."""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None
