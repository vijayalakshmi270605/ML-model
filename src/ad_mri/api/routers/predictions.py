"""`POST /api/v1/predictions`: a thin HTTP wrapper around
`inference.predictor.InferenceEngine.predict()` — no inference logic lives
here, only upload handling and response shaping. Error mapping is centralized
(`api.exception_handlers`) — this router does not catch the engine's
exceptions itself.

The blocking `InferenceEngine.predict()` call runs via `asyncio.to_thread` so
a request in flight never blocks the event loop from serving other requests
concurrently (requirement: "async API where appropriate" — the upload I/O
above it is natively async; the CPU-bound inference itself cannot be made
`async def` without penalty, so it is offloaded to a worker thread instead of
serialized onto the loop).
"""

from __future__ import annotations

import asyncio
import shutil
from typing import TYPE_CHECKING

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, status

from ad_mri.api import audit, monitoring
from ad_mri.api.deps import get_inference_engine, require_scope
from ad_mri.api.schemas.predict import PredictionResponse
from ad_mri.api.upload_utils import save_upload_within_limit
from ad_mri.core.config import get_settings
from ad_mri.core.constants import ApiScope
from ad_mri.core.logging import get_logger
from ad_mri.inference.predictor import redact_image_reference

if TYPE_CHECKING:
    from ad_mri.inference.predictor import InferenceEngine

router = APIRouter(prefix="/predictions", tags=["predictions"])
logger = get_logger(__name__)

_ENGINE_NOT_CONFIGURED_DETAIL = (
    "Inference is not configured on this deployment (INFERENCE_MODEL_PATH unset "
    "or the configured checkpoint failed to load)."
)


def _current_request_id() -> str | None:
    """The ID `RequestIDMiddleware` already bound to structlog's contextvars
    for this request — reused so a prediction's `request_id` and its access
    log line correlate, rather than minting a second, unrelated ID."""
    return structlog.contextvars.get_contextvars().get("request_id")


@router.post(
    "",
    response_model=PredictionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Classify a brain MRI study",
    dependencies=[Depends(require_scope(ApiScope.PREDICT))],
)
async def create_prediction(
    request: Request,
    file: UploadFile,
    engine: InferenceEngine | None = Depends(get_inference_engine),
) -> PredictionResponse:
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=_ENGINE_NOT_CONFIGURED_DETAIL
        )

    settings = get_settings()
    request_id = _current_request_id()

    temp_dir, temp_path = await save_upload_within_limit(file, settings.max_upload_size_bytes)
    try:
        image_ref = redact_image_reference(temp_path)
        logger.info(
            "predictions_upload_received",
            request_id=request_id,
            image_ref=image_ref,
            content_type=file.content_type,
        )
        result = await asyncio.to_thread(engine.predict, temp_path, request_id=request_id)

        # Best-effort audit persistence (Phase 11) - never fails the
        # response; see `api.audit`'s module docstring.
        presented_key = request.headers.get(settings.api_key_header_name)
        await asyncio.to_thread(
            audit.record_prediction,
            result=result,
            model_info=engine.model_version_info(),
            image_ref=image_ref,
            content_type=file.content_type,
            api_key=presented_key,
        )
        # Best-effort data-drift monitoring persistence (Phase 13) - same
        # never-fails-the-response contract; see `api.monitoring`.
        if result.image_statistics is not None:
            await asyncio.to_thread(
                monitoring.record_input_statistics,
                external_request_id=result.request_id,
                image_statistics=result.image_statistics,
            )
        return PredictionResponse(**result.to_dict())
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
