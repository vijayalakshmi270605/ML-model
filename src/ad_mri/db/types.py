"""Portable column types shared by every ORM model in `db.models`.

`GUID` is the standard SQLAlchemy recipe for a UUID column that is a native
`UUID` on PostgreSQL (the only database this project ships migrations for)
but degrades to a 32-char hex string everywhere else — which is what lets
`tests/unit/test_db_*` exercise the exact same models/repositories/service
layer against a real (if disposable) SQLite file instead of requiring a
locally running PostgreSQL server for every contributor and CI run.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import CHAR
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.engine import Dialect
from sqlalchemy.types import CHAR as SA_CHAR
from sqlalchemy.types import TypeDecorator


class GUID(TypeDecorator[uuid.UUID]):
    """Platform-independent UUID column.

    Uses PostgreSQL's native `UUID` type when bound to a `postgresql`
    dialect; otherwise stores a 32-character hex string (no dashes) in a
    `CHAR(32)` column. Values in and out of Python code are always a real
    `uuid.UUID`.
    """

    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect: Dialect) -> Any:
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        return dialect.type_descriptor(SA_CHAR(32))

    def process_bind_param(self, value: Any, dialect: Dialect) -> Any:
        if value is None:
            return None
        if dialect.name == "postgresql":
            return str(value)
        if not isinstance(value, uuid.UUID):
            value = uuid.UUID(str(value))
        return value.hex

    def process_result_value(self, value: Any, dialect: Dialect) -> uuid.UUID | None:
        if value is None:
            return None
        if isinstance(value, uuid.UUID):
            return value
        return uuid.UUID(value)
