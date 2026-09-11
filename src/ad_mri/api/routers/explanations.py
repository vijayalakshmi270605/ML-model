"""`POST /api/v1/explanations`: a thin HTTP wrapper around
`inference.predictor.InferenceEngine.generate_explanation()` — the **loud**
explanation path (unlike `?explain=true` on a prediction, a request whose
entire purpose is producing an explanation must fail, not silently report
"unavailable", if it can't). No inference/explanation logic lives here;
error mapping is centralized (`api.exception_handlers`).

Runs the blocking call via `asyncio.to_thread`, same as
`api.routers.predictions` — see that module's docstring for why.

Also attaches an optional, best-effort LLM narrative summary
(`explainability.llm_summary.generate_narrative_summary`) — `None` unless
`OPENAI_API_KEY` is configured; never fails or slows this request path
beyond its own timeout regardless.
"""

from __future__ import annotations

import asyncio
import shutil
import time
from typing import TYPE_CHECKING

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, status

from ad_mri.api import audit
from ad_mri.api.deps import get_inference_engine, require_scope
from ad_mri.api.schemas.explain import ExplanationResponse
from ad_mri.api.upload_utils import save_upload_within_limit
from ad_mri.core.config import get_settings
from ad_mri.core.constants import ApiScope
from ad_mri.core.logging import get_logger
from ad_mri.explainability.llm_summary import generate_narrative_summary
from ad_mri.inference.predictor import redact_image_reference

if TYPE_CHECKING:
    from ad_mri.inference.predictor import InferenceEngine

router = APIRouter(prefix="/explanations", tags=["explanations"])
logger = get_logger(__name__)

_ENGINE_NOT_CONFIGURED_DETAIL = (
    "Inference is not configured on this deployment (INFERENCE_MODEL_PATH unset "
    "or the configured checkpoint failed to load)."
)


def _current_request_id() -> str | None:
    return structlog.contextvars.get_contextvars().get("request_id")


@router.post(
    "",
    response_model=ExplanationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate a LIME/SHAP explanation for one MRI study",
    dependencies=[Depends(require_scope(ApiScope.PREDICT))],
)
async def create_explanation(
    file: UploadFile,
    methods: str = Query("lime", description="Comma-separated: lime, shap."),
    target_class: str | None = Query(
        None, description="Explain this class by name. Default: the predicted class."
    ),
    engine: InferenceEngine | None = Depends(get_inference_engine),
) -> ExplanationResponse:
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=_ENGINE_NOT_CONFIGURED_DETAIL
        )

    request_id = _current_request_id()
    resolved_methods = tuple(m.strip() for m in methods.split(",") if m.strip())

    settings = get_settings()
    temp_dir, temp_path = await save_upload_within_limit(file, settings.max_upload_size_bytes)
    try:
        logger.info(
            "explanations_upload_received",
            request_id=request_id,
            image_ref=redact_image_reference(temp_path),
            content_type=file.content_type,
            methods=resolved_methods,
        )
        start = time.perf_counter()
        bundle = await asyncio.to_thread(
            engine.generate_explanation, temp_path, request_id, resolved_methods, target_class
        )
        processing_time_ms = round((time.perf_counter() - start) * 1000, 3)

        # Best-effort audit persistence (Phase 11) - never fails the
        # response; see `api.audit`'s module docstring.
        await asyncio.to_thread(
            audit.record_explanation,
            bundle=bundle,
            model_info=engine.model_version_info(),
            methods=",".join(resolved_methods),
            target_class=target_class,
        )

        # Best-effort, optional LLM narrative summary (see
        # explainability.llm_summary's module docstring) - None when
        # OPENAI_API_KEY isn't configured or the call fails; never fails
        # this response either way.
        narrative_summary = await generate_narrative_summary(bundle)

        return ExplanationResponse(
            request_id=bundle.input_id,
            model_version=bundle.model_version,
            preprocessing_version=bundle.preprocessing_version,
            prediction=bundle.prediction.predicted_class,
            confidence=bundle.prediction.confidence,
            probabilities=bundle.prediction.class_probabilities,
            explanation_id=bundle.input_id,
            methods=[name for name in ("lime", "shap") if getattr(bundle, name) is not None],
            processing_time_ms=processing_time_ms,
            timestamp=bundle.generated_at,
            narrative_summary=narrative_summary,
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
