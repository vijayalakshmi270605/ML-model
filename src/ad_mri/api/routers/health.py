"""Liveness/readiness probes and Prometheus metrics exposition."""

from __future__ import annotations

from fastapi import APIRouter, Response, status

from ad_mri.db.session import get_engine
from ad_mri.monitoring.health_checks import check_database
from ad_mri.monitoring.metrics import render_latest_metrics

router = APIRouter(tags=["health"])


@router.get("/healthz", summary="Liveness probe")
def liveness() -> dict[str, str]:
    """Always returns 200 if the process is running and able to handle requests."""
    return {"status": "ok"}


@router.get("/readyz", summary="Readiness probe")
def readiness(response: Response) -> dict[str, str]:
    """Returns 200 only if downstream dependencies (currently: the database) are reachable."""
    if check_database(get_engine()):
        return {"status": "ok", "database": "reachable"}
    response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "unavailable", "database": "unreachable"}


@router.get("/metrics", summary="Prometheus metrics")
def metrics() -> Response:
    payload, content_type = render_latest_metrics()
    return Response(content=payload, media_type=content_type)
