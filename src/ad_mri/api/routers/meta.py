"""Versioned meta endpoints: `GET /api/v1/health` (liveness), `GET
/api/v1/ready` (readiness), `GET /api/v1/version`.

Same underlying checks as the unversioned `/healthz`/`/readyz` probes
(`api.routers.health`) — infra (Kubernetes probes, load balancers) typically
expects those at a fixed, unversioned root path, so both are kept; there is
one implementation of each check either way (`monitoring.health_checks`),
never two.

Readiness here additionally reports the inference engine's health — but only
when one is actually *configured* (`INFERENCE_MODEL_PATH` set): a deployment
that intentionally runs without a model yet (e.g. API-only, or before a
checkpoint is pushed) must not be reported "not ready" for a dependency it
was never asked to have.
"""

from __future__ import annotations

from fastapi import APIRouter, Response, status

from ad_mri import __version__
from ad_mri.api.deps import get_inference_engine
from ad_mri.api.schemas.health import LivenessResponse, ReadinessResponse, VersionResponse
from ad_mri.core.config import get_settings
from ad_mri.db.session import get_engine
from ad_mri.monitoring.health_checks import check_database, check_inference_engine

router = APIRouter(tags=["meta"])


@router.get("/health", summary="Liveness probe (versioned)")
def health() -> LivenessResponse:
    """Always 200 if the process is running and able to handle requests."""
    return LivenessResponse(status="ok")


@router.get("/ready", summary="Readiness probe (versioned)")
def ready(response: Response) -> ReadinessResponse:
    """200 only if every *configured* dependency is reachable."""
    database_ok = check_database(get_engine())

    settings = get_settings()
    inference_status: str | None = None
    inference_ok = True
    if settings.inference_model_path is not None:
        inference_ok = check_inference_engine(get_inference_engine())
        inference_status = "reachable" if inference_ok else "unreachable"

    overall_ok = database_ok and inference_ok
    if not overall_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(
        status="ok" if overall_ok else "unavailable",
        database="reachable" if database_ok else "unreachable",
        inference_engine=inference_status,
    )


@router.get("/version", summary="API/application version")
def version() -> VersionResponse:
    settings = get_settings()
    return VersionResponse(
        app_name=settings.app_name,
        version=__version__,
        app_env=str(settings.app_env),
        api_version=settings.api_v1_prefix.rsplit("/", 1)[-1],
    )
