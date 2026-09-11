"""Tests for `api.exception_handlers._handle_unexpected_error` - the
catch-all for anything that is not one of this project's own `AppError`
subclasses. Real FastAPI app, real middleware stack, no mocks - this is
specifically what proves the real fix for the gap
`tests/unit/test_api_security_headers.py` documents (a bare-`Exception`
handler is dispatched by `ServerErrorMiddleware`, outside every
`app.add_middleware`-registered middleware)."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ad_mri.api.exception_handlers import register_exception_handlers
from ad_mri.api.middleware.security_headers import SecurityHeadersMiddleware


def _build_app() -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)
    app.add_middleware(SecurityHeadersMiddleware)

    @app.get("/boom")
    async def boom() -> None:
        raise KeyError("something genuinely unexpected")

    return app


def test_an_unexpected_exception_returns_a_clean_500_not_a_raw_traceback() -> None:
    client = TestClient(_build_app(), raise_server_exceptions=False)
    response = client.get("/boom")

    assert response.status_code == 500
    body = response.json()
    assert body == {"detail": "Internal error.", "error_code": "INTERNAL_ERROR"}
    # The exception's own message/type is never echoed back to the caller.
    assert "something genuinely unexpected" not in response.text
    assert "KeyError" not in response.text


def test_an_unexpected_exception_still_gets_security_headers() -> None:
    """The whole point of `_handle_unexpected_error` calling
    `apply_security_headers` directly, rather than relying on
    `SecurityHeadersMiddleware` (which never runs for this response shape -
    see that test module)."""
    client = TestClient(_build_app(), raise_server_exceptions=False)
    response = client.get("/boom")

    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert (
        response.headers["Content-Security-Policy"] == "default-src 'none'; frame-ancestors 'none'"
    )


def test_a_mapped_app_error_still_gets_the_normal_handler_not_the_catch_all() -> None:
    """The catch-all must never shadow the specific handlers above it -
    registration order in `register_exception_handlers` puts `Exception`
    last precisely so FastAPI picks the most specific match."""
    from ad_mri.core.exceptions import DataValidationError

    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/invalid")
    async def invalid() -> None:
        raise DataValidationError("some path detail that must never leak")

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/invalid")

    assert response.status_code == 422
    assert response.json()["error_code"] == "VALIDATION_FAILED"
    assert "some path detail" not in response.text
