"""`GET /api/v1/models`: describes the model(s) currently loaded and serving.

Not the full MLflow Model Registry (multiple versions, promotion, staged vs.
production) — that remains a separate, not-yet-implemented phase (see
`model_registry.client`). With no real registry, the list has at most one
entry: whatever checkpoint `INFERENCE_MODEL_PATH` currently points the
process at. An empty list (not an error) means no model is configured —
"list the models" has a perfectly good real answer (zero) in that case.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends

from ad_mri.api.deps import get_inference_engine
from ad_mri.api.schemas.models import ModelInfo

if TYPE_CHECKING:
    from ad_mri.inference.predictor import InferenceEngine

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", summary="List currently-served model version(s)")
async def list_models(
    engine: InferenceEngine | None = Depends(get_inference_engine),
) -> list[ModelInfo]:
    if engine is None:
        return []
    info = engine.model_version_info()
    return [
        ModelInfo(
            name=info.architecture,
            version=info.model_version,
            stage="serving",
            model_config_hash=info.model_config_hash,
            preprocessing_version=info.preprocessing_version,
            device=info.device,
        )
    ]
