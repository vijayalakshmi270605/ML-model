"""`POST /api/v1/monitoring/ground-truth`: the delayed-model-performance-
monitoring input - attaches a true class to a *past* prediction once it
becomes available. No inference/business logic lives here; this is a thin
wrapper around `api.monitoring.record_ground_truth`
(`db.monitoring_service.MonitoringService`), same layering as
`api.routers.predictions`/`explanations`.

Unlike those two, this endpoint's own domain error (an unknown
`request_id` - no such prediction was ever recorded) is handled directly
here as a `404`, not routed through `api.exception_handlers`: it is a
router-level precondition on this endpoint's one input, not one of the
project's typed `core.exceptions.AppError` subclasses (mirrors
`api.routers.predictions`'s direct `HTTPException(503)` for "engine not
configured" - see that router's docstring for the same reasoning).
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, status

from ad_mri.api import monitoring
from ad_mri.api.deps import require_scope
from ad_mri.api.schemas.monitoring import GroundTruthRequest, GroundTruthResponse
from ad_mri.core.constants import ApiScope
from ad_mri.core.logging import get_logger

router = APIRouter(prefix="/monitoring", tags=["monitoring"])
logger = get_logger(__name__)


@router.post(
    "/ground-truth",
    response_model=GroundTruthResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Attach a ground-truth label to a past prediction",
    dependencies=[Depends(require_scope(ApiScope.LABEL))],
)
async def create_ground_truth_label(payload: GroundTruthRequest) -> GroundTruthResponse:
    try:
        await asyncio.to_thread(
            monitoring.record_ground_truth,
            external_request_id=payload.request_id,
            true_class=payload.true_class,
            label_source=payload.label_source,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    return GroundTruthResponse(
        request_id=payload.request_id,
        true_class=payload.true_class,
        label_source=payload.label_source,
    )
