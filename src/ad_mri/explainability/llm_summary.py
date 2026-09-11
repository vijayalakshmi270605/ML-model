"""Optional, best-effort natural-language summary of a LIME/SHAP explanation's
*numeric* output, via the OpenAI API — off by default (`Settings.openai_api_key`
unset means this simply no-ops; `POST /explanations` works exactly as it did
before this module existed).

**Privacy boundary (this is the point of this module's design, not an
afterthought)**: the only thing ever sent to the third-party API is a small
JSON object built entirely from numbers/labels this codebase already computed
locally — predicted class, calibrated probabilities, and each requested
method's already-aggregated region-importance statistics (LIME's ranked
superpixel `segment_id`/`weight` pairs; SHAP's `attribution_sum` and
`explainer_type`). **Never sent**: the uploaded file's name or path, any
patient-identifying string, or a single raw pixel/voxel value — the per-voxel
SHAP array stays on disk (`ExplanationBundle.artifacts["shap_values"]`) and is
never read by this module. See `_build_payload` for the exact, exhaustive set
of fields — if a field isn't listed there, it doesn't leave this process.

**Responsible-AI boundary**: the system prompt explicitly forbids a
diagnostic or causal claim (mirroring `core.disclaimers`), but an LLM's
output is not a control this codebase can fully guarantee — the response
schema's own `disclaimer` field (`ExplanationResponse.disclaimer`, defined
independently of this module) remains the authoritative safety statement
regardless of what this function returns. This function's output is always
additive, labeled, and never replaces that field.

`openai` is a lazily-imported, optional dependency (the `llm` extra) — never
required by the core API/train images, same convention as `lime`/`shap`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ad_mri.core.config import Settings, get_settings
from ad_mri.core.logging import get_logger

if TYPE_CHECKING:
    from ad_mri.explainability.service import ExplanationBundle

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You summarize the numeric output of an image-explainability tool (LIME/SHAP) "
    "for a research-grade brain MRI classifier. You are given only region-importance "
    "statistics and class probabilities, never image data or patient information. "
    "Write 2-4 plain-language sentences describing which regions the model's own "
    "output was most sensitive to. You MUST NOT provide a medical diagnosis, MUST "
    "NOT state or imply a causal or clinical relationship between any region and a "
    "disease process, and MUST NOT suggest a clinical action. This is a description "
    "of model behavior only, not medical advice."
)


def _round_all(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def _build_payload(bundle: ExplanationBundle) -> dict[str, Any]:
    """The complete, exhaustive set of fields sent to the API — every value
    here traces back to a number this codebase already computed, never raw
    input. Anything not assembled here is, by construction, never sent."""
    payload: dict[str, Any] = {
        "predicted_class": bundle.prediction.predicted_class,
        "target_class": bundle.class_names[bundle.target_class_index],
        "class_probabilities": {
            name: _round_all(prob) for name, prob in bundle.prediction.class_probabilities.items()
        },
    }
    if bundle.lime is not None:
        payload["lime"] = {
            "top_regions": [
                {
                    "segment_id": int(sp["segment_id"]),
                    "weight": _round_all(sp["weight"]),
                }
                for sp in bundle.lime.top_superpixels[:5]
            ],
            "surrogate_fit_r2": _round_all(bundle.lime.surrogate_r2),
        }
    if bundle.shap is not None:
        payload["shap"] = {
            "explainer_type": bundle.shap.explainer_type,
            "attribution_sum": _round_all(bundle.shap.attribution_sum),
        }
    return payload


async def generate_narrative_summary(bundle: ExplanationBundle) -> str | None:
    """Best-effort: returns `None` (never raises) when `Settings.openai_api_key`
    is unset, the `openai` package isn't installed, or the call fails/times
    out for any reason — mirrors `api.audit`'s "loud service, best-effort
    caller" split, one layer removed: a downstream/third-party failure here
    must never fail a `POST /explanations` request whose actual job (the
    LIME/SHAP explanation itself) already succeeded."""
    settings: Settings = get_settings()
    if settings.openai_api_key is None:
        return None

    try:
        from openai import AsyncOpenAI
    except ImportError:
        logger.warning("llm_summary_unavailable", reason="openai package not installed")
        return None

    payload = _build_payload(bundle)
    try:
        client = AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            timeout=settings.openai_timeout_seconds,
        )
        response = await client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": str(payload)},
            ],
            max_tokens=200,
            temperature=0.2,
        )
    except Exception as exc:  # noqa: BLE001 - a third-party API failure must never fail the request
        logger.warning("llm_summary_generation_failed", error_type=type(exc).__name__)
        return None

    choices = response.choices
    if not choices or choices[0].message.content is None:
        logger.warning("llm_summary_generation_failed", error_type="EmptyResponse")
        return None
    return choices[0].message.content.strip()
