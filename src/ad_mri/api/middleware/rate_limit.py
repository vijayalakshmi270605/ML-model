"""Rate-limiting architecture: an in-memory, per-process sliding-window
limiter — real and enforced by default (`Settings.rate_limit_per_minute`,
already existed unused before this phase), with a documented seam for a
distributed backend.

This implementation is correct for a **single API process**. A multi-instance
deployment shares no state between processes, so each instance would enforce
the limit independently (effectively `limit * instance_count`) — acceptable
for a first deployment, not for a scaled one. The seam to close that gap is
narrow and deliberate: swap `_InMemoryLimiter` below for a Redis-backed
counter (`INCR` + `EXPIRE`, or a sorted-set sliding window) — `REDIS_URL` is
already configured (see `.env.example`, used today by Celery) — without
touching `RateLimitMiddleware` or any router.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from ad_mri.core.constants import API_KEY_HEADER
from ad_mri.core.logging import get_logger

logger = get_logger(__name__)

_WINDOW_SECONDS = 60.0
#: Liveness/readiness/metrics probes are infra traffic, not API consumption —
#: exempt so a monitoring system's own polling can never trip a client's limit
#: (they share no key with real clients) or get rate-limited itself.
_EXEMPT_PATH_SUFFIXES = ("/healthz", "/readyz", "/metrics", "/health", "/ready")


class _InMemoryLimiter:
    """Sliding-window counter per key, bounded memory (each key holds at most
    `limit` timestamps). Thread-safe — Starlette may run middleware dispatch
    for concurrent requests on different threads under some ASGI servers."""

    def __init__(self, limit: int, window_seconds: float = _WINDOW_SECONDS) -> None:
        self._limit = limit
        self._window_seconds = window_seconds
        self._buckets: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        cutoff = now - self._window_seconds
        with self._lock:
            bucket = self._buckets.setdefault(key, deque())
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= self._limit:
                return False
            bucket.append(now)
            return True

    def reset(self) -> None:
        """Test-only: clear all state."""
        with self._lock:
            self._buckets.clear()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """`requests_per_minute <= 0` disables enforcement entirely (useful for
    tests and for a deployment that fronts this API with its own limiter)."""

    def __init__(
        self, app: ASGIApp, requests_per_minute: int, api_key_header_name: str = API_KEY_HEADER
    ) -> None:
        super().__init__(app)
        self._enabled = requests_per_minute > 0
        self._limiter = _InMemoryLimiter(requests_per_minute) if self._enabled else None
        #: Security review finding (Phase 14): this used to hardcode
        #: "X-API-Key" regardless of a reconfigured `API_KEY_HEADER_NAME`
        #: (the same class of bug `api.deps.require_api_key` was already
        #: fixed for, in Phase 10) - a deployment that renamed its API-key
        #: header would silently fall back to IP-based limiting instead of
        #: per-key limiting. Not itself an auth bypass (IP-based limiting is
        #: *more* conservative, not less), but a real inconsistency, fixed
        #: to actually take the configured header name.
        self._api_key_header_name = api_key_header_name

    def _client_key(self, request: Request) -> str:
        """API key if presented (so one client's own limit follows them
        regardless of source IP — e.g. behind a shared NAT/proxy), else the
        connecting IP."""
        api_key = request.headers.get(self._api_key_header_name)
        if api_key:
            return f"key:{api_key}"
        client = request.client
        return f"ip:{client.host}" if client else "ip:unknown"

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if not self._enabled or request.url.path.endswith(_EXEMPT_PATH_SUFFIXES):
            return await call_next(request)

        assert self._limiter is not None  # narrows for mypy; guaranteed by _enabled
        key = self._client_key(request)
        if not self._limiter.allow(key):
            logger.warning("rate_limit_exceeded", path=request.url.path)
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded.", "error_code": "RATE_LIMIT_EXCEEDED"},
                headers={"Retry-After": str(int(_WINDOW_SECONDS))},
            )
        return await call_next(request)
