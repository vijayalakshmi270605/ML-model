"""Tests for `api.middleware.security_headers` - real HTTP responses via a
real Starlette app + TestClient, no mocks."""

from __future__ import annotations

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from ad_mri.api.middleware.security_headers import SecurityHeadersMiddleware


async def _ok(request: object) -> JSONResponse:
    return JSONResponse({"status": "ok"})


async def _boom(request: object) -> JSONResponse:
    raise ValueError("boom")


def _build_app() -> Starlette:
    app = Starlette(routes=[Route("/api/v1/predictions", _ok), Route("/docs", _ok)])
    app.add_middleware(SecurityHeadersMiddleware)
    return app


def test_every_response_carries_the_baseline_security_headers() -> None:
    client = TestClient(_build_app())
    response = client.get("/api/v1/predictions")

    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "no-referrer"
    assert "geolocation=()" in response.headers["Permissions-Policy"]
    assert "max-age=" in response.headers["Strict-Transport-Security"]


def test_api_responses_get_the_strict_content_security_policy() -> None:
    client = TestClient(_build_app())
    response = client.get("/api/v1/predictions")
    assert (
        response.headers["Content-Security-Policy"] == "default-src 'none'; frame-ancestors 'none'"
    )


def test_docs_pages_get_a_looser_csp_allowing_their_own_cdn() -> None:
    client = TestClient(_build_app())
    response = client.get("/docs")
    csp = response.headers["Content-Security-Policy"]
    assert "cdn.jsdelivr.net" in csp
    assert csp != "default-src 'none'; frame-ancestors 'none'"


def test_a_genuinely_unhandled_exception_never_reaches_this_middlewares_dispatch() -> None:
    """Real, verified architecture, not an assumption: per `Starlette
    .build_middleware_stack`, a handler registered for bare `Exception`
    becomes `ServerErrorMiddleware`'s own handler, and `ServerErrorMiddleware`
    is the true outermost layer - added *before* any `app.add_middleware`
    call, so this middleware's `dispatch` never even runs for that response.
    `api.exception_handlers._handle_unexpected_error` (tested directly in
    `tests/unit/test_api_exception_handlers.py`) is what actually closes this
    gap, by calling `apply_security_headers` on its own response instead of
    relying on this middleware - see both modules' docstrings."""
    app = Starlette(routes=[Route("/boom", _boom)])
    app.add_middleware(SecurityHeadersMiddleware)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/boom")
    assert response.status_code == 500
    assert "X-Content-Type-Options" not in response.headers
