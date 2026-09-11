"""Request/response contract for `POST /api/v1/predictions`.

`PredictionResponse` mirrors `inference.predictor.PredictionResult` field for
field, plus one the engine itself doesn't carry: `disclaimer` (Phase 14, see
`core.disclaimers`) — the HTTP boundary is where this response becomes
something a real caller reads, so it is where the "not a clinical diagnosis"
disclaimer belongs, attached to the data it qualifies rather than left in a
document the caller might never open. See docs/inference.md, docs/api.md,
docs/responsible_ai.md, docs/TECHNICAL_SPECIFICATION.md §6, §9.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ad_mri.core.disclaimers import PREDICTION_DISCLAIMER


class PredictionResponse(BaseModel):
    """Deterministic response structure for `POST /api/v1/predictions`."""

    request_id: str
    model_version: str
    preprocessing_version: str
    prediction: str
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float]
    processing_time_ms: float
    explanation_available: bool
    explanation_id: str | None = None
    timestamp: str
    disclaimer: str = PREDICTION_DISCLAIMER
