"""Response contract for `GET /api/v1/models`.

Describes the model(s) currently loaded and serving — not a full MLflow
Model Registry listing (multiple versions, promotion history, staged vs
production), which remains a separate, not-yet-implemented phase (see
`model_registry.client`). With no real registry, there is at most one entry:
the checkpoint `INFERENCE_MODEL_PATH` currently points the process at.
"""

from __future__ import annotations

from pydantic import BaseModel


class ModelInfo(BaseModel):
    name: str
    version: str
    stage: str
    model_config_hash: str
    preprocessing_version: str
    device: str
