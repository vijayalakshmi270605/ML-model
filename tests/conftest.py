"""Shared pytest fixtures.

Sets safe, non-secret environment defaults *before* any app module is
imported, so the test suite never depends on a local .env file or a running
database.
"""

from __future__ import annotations

import os

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("LOG_FORMAT", "console")
os.environ.setdefault("SECRET_KEY", "test-secret-key")
# The optional LLM narrative-summary feature (explainability.llm_summary) must
# never make a real, billed network call as a side effect of running this
# suite, regardless of a real key sitting in this sandbox's own .env - see
# this module's docstring. test_llm_summary.py covers the feature itself
# (mocked) plus its own manually-invoked live check, deliberately not part of
# this automated suite.
os.environ.setdefault("OPENAI_API_KEY", "")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from ad_mri.core.config import get_settings  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    """Settings, the inference engine, and the DB engine/sessionmaker are all
    process-cached module globals; clear all three around each test so one
    test's env vars/model/DATABASE_URL never leak into the next (the
    inference engine is cache-cleared here, not shut down — tests that build
    a real one are responsible for their own cleanup; the DB engine is
    disposed by `reset_engine_cache()` itself)."""
    from ad_mri.api.deps import clear_inference_engine_cache
    from ad_mri.db.session import reset_engine_cache

    get_settings.cache_clear()
    clear_inference_engine_cache()
    reset_engine_cache()
    yield
    get_settings.cache_clear()
    clear_inference_engine_cache()
    reset_engine_cache()


@pytest.fixture
def settings():
    return get_settings()


@pytest.fixture
def client():
    from ad_mri.api.main import create_app

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
