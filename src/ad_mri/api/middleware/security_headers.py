"""Secure HTTP response headers — applied to every response, including
error responses (this middleware sits close to the outside of the stack;
see `api.main`'s module docstring for the full ordering rationale).

Two policies, not one: this API is almost entirely JSON (no page ever
renders untrusted content), so every response gets the strictest practical
`Content-Security-Policy` (`default-src 'none'`) — except the two paths
FastAPI itself serves as real HTML (`/docs`, `/redoc`, Swagger UI/ReDoc),
which need to load their own JS/CSS from jsdelivr's CDN (FastAPI's own
default asset URLs — see `fastapi.openapi.docs`) and therefore get a
deliberately looser, docs-scoped policy instead. Every other header below
applies unconditionally — none of them need the JSON/HTML distinction.

**One real gap this middleware cannot close by itself**: a genuinely
unhandled exception (nothing registered via `app.add_exception_handler` for
its type) is caught by Starlette's own `ServerErrorMiddleware` — which,
per `Starlette.build_middleware_stack`, is added *outside* every
user-registered middleware (including this one), specifically so a broken
middleware can never suppress the framework's own last-resort error
handling. A response built there never passes back through this
middleware's `dispatch`, so it never gets these headers via `call_next`.
`api.exception_handlers._handle_unexpected_error` (the catch-all handler
registered for bare `Exception`, which *is* what lands in
`ServerErrorMiddleware`) works around this the direct way: it calls
`apply_security_headers` on its own response before returning it, rather
than relying on this middleware to do it. See that handler for the full
explanation — this is not a theoretical concern, it was found and verified
by a real test (`tests/unit/test_api_security_headers.py`) during Phase 14's
security review, not assumed safe.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

#: Bound to `Response` but preserves the caller's exact subtype (e.g.
#: `JSONResponse`) - `apply_security_headers` mutates and returns the same
#: object it was given, so callers with a more specific return-type
#: annotation (like `_handle_unexpected_error` -> `JSONResponse`) don't need
#: an unnecessary cast.
_R = TypeVar("_R", bound=Response)

#: FastAPI's own default docs pages — the only responses this app ever
#: serves that are real, browser-rendered HTML with third-party script/style
#: dependencies. `/openapi.json` is plain JSON data, not HTML — it gets the
#: strict policy like every other endpoint.
_DOCS_PATHS = ("/docs", "/redoc")

_STRICT_CSP = "default-src 'none'; frame-ancestors 'none'"
#: `'unsafe-inline'` is a real, deliberate relaxation - Swagger UI/ReDoc's
#: own bundles inject inline `<style>`/init `<script>` blocks that a strict
#: CSP would otherwise block, breaking the docs page itself. Scoped to only
#: these two paths - every actual API response keeps the strict policy above.
_DOCS_CSP = (
    "default-src 'self'; "
    "script-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; "
    "style-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; "
    "img-src 'self' https://fastapi.tiangolo.com data:; "
    "connect-src 'self'; "
    "frame-ancestors 'none'"
)


def apply_security_headers(response: _R, *, path: str) -> _R:
    """The one place every header is actually set — called from
    `SecurityHeadersMiddleware.dispatch` for every normal response, and
    directly from `api.exception_handlers._handle_unexpected_error` for the
    one response shape that never reaches this middleware's `dispatch` at
    all (see this module's docstring). Mutates `response` in place and
    returns it, so both call sites can use it identically."""
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = (
        "geolocation=(), microphone=(), camera=(), payment=(), usb=()"
    )
    # Ignored by a browser talking to this app over plain HTTP (local
    # dev) — harmless to send unconditionally; a real deployment
    # terminates TLS in front of this process (docs/deployment.md), at
    # which point this header takes effect for real.
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = _DOCS_CSP if path in _DOCS_PATHS else _STRICT_CSP
    return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        return apply_security_headers(response, path=request.url.path)
