"""Tests for `db.types.GUID` - a real round trip through the type's own bind/
result processing for both dialects it supports, no mocks."""

from __future__ import annotations

import uuid

from sqlalchemy.dialects import postgresql, sqlite

from ad_mri.db.types import GUID


def test_binds_as_a_plain_string_on_postgresql() -> None:
    value = uuid.uuid4()
    dialect = postgresql.dialect()
    bound = GUID().process_bind_param(value, dialect)
    assert bound == str(value)


def test_binds_as_hex_on_sqlite() -> None:
    value = uuid.uuid4()
    dialect = sqlite.dialect()
    bound = GUID().process_bind_param(value, dialect)
    assert bound == value.hex
    assert len(bound) == 32


def test_round_trips_through_sqlite_bind_and_result() -> None:
    value = uuid.uuid4()
    dialect = sqlite.dialect()
    bound = GUID().process_bind_param(value, dialect)
    restored = GUID().process_result_value(bound, dialect)
    assert restored == value


def test_round_trips_through_postgresql_bind_and_result() -> None:
    value = uuid.uuid4()
    dialect = postgresql.dialect()
    bound = GUID().process_bind_param(value, dialect)
    restored = GUID().process_result_value(bound, dialect)
    assert restored == value


def test_none_stays_none() -> None:
    dialect = sqlite.dialect()
    assert GUID().process_bind_param(None, dialect) is None
    assert GUID().process_result_value(None, dialect) is None


def test_accepts_a_string_uuid_on_bind() -> None:
    """A caller that already has a UUID as text (e.g. from a query param)
    should not need to construct a `uuid.UUID` first."""
    value = uuid.uuid4()
    dialect = sqlite.dialect()
    bound = GUID().process_bind_param(str(value), dialect)
    assert bound == value.hex


def test_result_value_already_a_uuid_is_returned_unchanged() -> None:
    """PostgreSQL's own driver already returns a `uuid.UUID`, not a string -
    the type must not double-process it."""
    value = uuid.uuid4()
    dialect = postgresql.dialect()
    assert GUID().process_result_value(value, dialect) == value


def test_load_dialect_impl_uses_native_uuid_on_postgresql() -> None:
    impl = GUID().load_dialect_impl(postgresql.dialect())
    # SQLAlchemy's own postgresql-dialect UUID type (its class is internally
    # named `PGUuid` in this SQLAlchemy version) - not the CHAR(32) fallback.
    assert isinstance(impl, postgresql.UUID)


def test_load_dialect_impl_uses_char32_elsewhere() -> None:
    impl = GUID().load_dialect_impl(sqlite.dialect())
    assert impl.length == 32
