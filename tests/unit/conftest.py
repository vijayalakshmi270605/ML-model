"""Fixtures shared by the `db.*` unit tests.

Exercises the exact same SQLAlchemy models/repositories/service code the
application runs against PostgreSQL, against a real (file-backed, not
in-memory - so `sessionmaker()`'s per-call `Session()` all see the same
data) SQLite database instead. See `db.types.GUID`'s docstring for why that
substitution is possible without touching a single line of `db.models`.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from ad_mri.db import models  # noqa: F401 - ensures every table is registered on Base.metadata
from ad_mri.db.base import Base


@pytest.fixture
def db_engine(tmp_path: Path) -> Iterator[Engine]:
    engine = create_engine(f"sqlite:///{tmp_path / 'audit_test.db'}", future=True)

    # SQLite ignores foreign keys unless a connection explicitly turns
    # enforcement on - without this, a test proving a bad foreign key is
    # rejected would pass here and then fail for real against PostgreSQL.
    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection: object, _connection_record: object) -> None:
        cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def db_session(db_engine: Engine) -> Iterator[Session]:
    factory = sessionmaker(bind=db_engine, autoflush=False, autocommit=False, future=True)
    session = factory()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def db_sessionmaker(db_engine: Engine) -> sessionmaker[Session]:
    """For tests that need to open more than one independent `Session`
    against the same database (e.g. proving a committed write is visible
    from a fresh session, or that a rolled-back one is not)."""
    return sessionmaker(bind=db_engine, autoflush=False, autocommit=False, future=True)
