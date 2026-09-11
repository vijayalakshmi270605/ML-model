"""Response contract for `POST /api/v1/explanations`.

Mirrors `PredictionResponse` (the prediction the explanation is *for* is
useful context, not a separate round-trip) plus explanation-specific fields.
The heavy artifacts themselves (overlay PNGs, raw SHAP values, the full
`manifest.json`) are not embedded — they live on disk under
`explanation_id` (see `docs/explainability.md` §8); this response is a
pointer to them, not the payload.

`disclaimer` (Phase 14, see `core.disclaimers`) is explanation-specific, not
`PredictionResponse`'s: an explanation invites a stronger, easier
misreading ("this is *why* the model thinks the patient has the disease")
than a bare prediction does, so its wording addresses causality directly,
not just "not a diagnosis" — see docs/responsible_ai.md.

`narrative_summary` (optional LLM addendum, see `explainability.llm_summary`)
is `None` whenever that feature isn't configured or the call failed/timed
out — it is never load-bearing: `disclaimer` above is the field this
response's safety guarantee actually rests on, regardless of whether a
narrative summary was generated.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ad_mri.core.disclaimers import EXPLANATION_DISCLAIMER


class ExplanationResponse(BaseModel):
    request_id: str
    model_version: str
    preprocessing_version: str
    prediction: str
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float]
    explanation_id: str
    methods: list[str]
    processing_time_ms: float
    timestamp: str
    disclaimer: str = EXPLANATION_DISCLAIMER
    narrative_summary: str | None = None
