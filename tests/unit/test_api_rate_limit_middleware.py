"""Tests for api.middleware.rate_limit — a real sliding-window limiter under
a real Starlette app + TestClient (no mocks)."""

from __future__ import annotations

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from ad_mri.api.middleware.rate_limit import RateLimitMiddleware, _InMemoryLimiter


async def _ok(request: object) -> JSONResponse:
    return JSONResponse({"status": "ok"})


def _build_app(requests_per_minute: int) -> Starlette:
    app = Starlette(
        routes=[
            Route("/api/v1/predictions", _ok, methods=["POST"]),
            Route("/healthz", _ok, methods=["GET"]),
            Route("/api/v1/health", _ok, methods=["GET"]),
        ]
    )
    app.add_middleware(RateLimitMiddleware, requests_per_minute=requests_per_minute)
    return app


# --- _InMemoryLimiter: pure unit tests, no HTTP -----------------------------


def test_limiter_allows_up_to_the_limit() -> None:
    limiter = _InMemoryLimiter(limit=3)
    assert limiter.allow("client-a") is True
    assert limiter.allow("client-a") is True
    assert limiter.allow("client-a") is True


def test_limiter_rejects_beyond_the_limit() -> None:
    limiter = _InMemoryLimiter(limit=3)
    for _ in range(3):
        assert limiter.allow("client-a") is True
    assert limiter.allow("client-a") is False


def test_limiter_tracks_keys_independently() -> None:
    limiter = _InMemoryLimiter(limit=1)
    assert limiter.allow("client-a") is True
    assert limiter.allow("client-b") is True  # a different key has its own budget
    assert limiter.allow("client-a") is False
    assert limiter.allow("client-b") is False


def test_limiter_reset_clears_all_state() -> None:
    limiter = _InMemoryLimiter(limit=1)
    limiter.allow("client-a")
    assert limiter.allow("client-a") is False
    limiter.reset()
    assert limiter.allow("client-a") is True


def test_limiter_window_expiry_frees_budget() -> None:
    """A request older than the window no longer counts against the limit."""
    # A generous window/sleep margin (not a tight race) — this only needs to
    # prove expiry happens at all, not measure its precision.
    limiter = _InMemoryLimiter(limit=1, window_seconds=0.2)
    assert limiter.allow("client-a") is True
    assert limiter.allow("client-a") is False
    import time

    time.sleep(0.4)
    assert limiter.allow("client-a") is True


# --- RateLimitMiddleware under a real ASGI app ------------------------------


def test_requests_within_the_limit_all_succeed() -> None:
    client = TestClient(_build_app(requests_per_minute=5))
    for _ in range(5):
        response = client.post("/api/v1/predictions")
        assert response.status_code == 200


def test_requests_beyond_the_limit_get_429() -> None:
    client = TestClient(_build_app(requests_per_minute=3))
    for _ in range(3):
        assert client.post("/api/v1/predictions").status_code == 200
    response = client.post("/api/v1/predictions")
    assert response.status_code == 429
    assert response.json()["error_code"] == "RATE_LIMIT_EXCEEDED"
    assert "Retry-After" in response.headers


def test_disabled_when_limit_is_non_positive() -> None:
    client = TestClient(_build_app(requests_per_minute=0))
    for _ in range(50):
        assert client.post("/api/v1/predictions").status_code == 200


def test_health_endpoints_are_exempt_from_rate_limiting() -> None:
    client = TestClient(_build_app(requests_per_minute=2))
    # exhaust the limit on the real endpoint
    for _ in range(2):
        client.post("/api/v1/predictions")
    assert client.post("/api/v1/predictions").status_code == 429

    # health endpoints still succeed despite the same client having no budget left
    assert client.get("/healthz").status_code == 200
    assert client.get("/api/v1/health").status_code == 200


def test_different_api_keys_get_independent_budgets() -> None:
    client = TestClient(_build_app(requests_per_minute=1))
    response_a1 = client.post("/api/v1/predictions", headers={"X-API-Key": "key-a"})
    response_b1 = client.post("/api/v1/predictions", headers={"X-API-Key": "key-b"})
    response_a2 = client.post("/api/v1/predictions", headers={"X-API-Key": "key-a"})

    assert response_a1.status_code == 200
    assert response_b1.status_code == 200  # different key, independent budget
    assert response_a2.status_code == 429  # key-a's single request/minute is used up


def test_a_reconfigured_api_key_header_name_is_honored() -> None:
    """Security review finding (Phase 14): this middleware used to hardcode
    "X-API-Key" regardless of `Settings.api_key_header_name` - a client
    presenting its key under the reconfigured header name must still get
    per-key (not per-IP) rate limiting."""
    app = Starlette(routes=[Route("/api/v1/predictions", _ok, methods=["POST"])])
    app.add_middleware(
        RateLimitMiddleware, requests_per_minute=1, api_key_header_name="X-Custom-Key"
    )
    client = TestClient(app)

    first = client.post("/api/v1/predictions", headers={"X-Custom-Key": "key-a"})
    second = client.post("/api/v1/predictions", headers={"X-Custom-Key": "key-a"})
    different_key = client.post("/api/v1/predictions", headers={"X-Custom-Key": "key-b"})

    assert first.status_code == 200
    assert second.status_code == 429  # same key under the custom header, budget used up
    assert different_key.status_code == 200  # a different key, independent budget
