"""Response contracts for `GET /api/v1/health`, `/ready`, and `/version`
(plus the equivalent unversioned `/healthz`/`/readyz` probes — same bodies,
same underlying checks, see `api.routers.health`/`api.routers.meta`)."""

from __future__ import annotations

from pydantic import BaseModel


class LivenessResponse(BaseModel):
    status: str  # "ok"


class ReadinessResponse(BaseModel):
    status: str  # "ok" | "unavailable"
    database: str  # "reachable" | "unreachable"
    #: Omitted (not `null`) when no model is configured on this deployment —
    #: readiness does not require one; see `api.routers.meta`.
    inference_engine: str | None = None  # "reachable" | "unreachable"


class VersionResponse(BaseModel):
    app_name: str
    version: str
    app_env: str
    api_version: str  # the API_V1_PREFIX segment, e.g. "v1"
