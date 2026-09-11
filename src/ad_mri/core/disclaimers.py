"""ML-safety disclaimers (Phase 14, docs/responsible_ai.md) — defined once,
here, and attached directly to every prediction/explanation response
(`api.schemas.predict.PredictionResponse.disclaimer`,
`api.schemas.explain.ExplanationResponse.disclaimer`) and to the CLI's
equivalent output (`cli.main.predict`), rather than living only in
documentation a caller might never read. A disclaimer that only exists in a
markdown file a human has to go find is not the same control as one that
arrives in the same JSON payload as the number it qualifies.

Two disclaimers, not one, because they guard against two different
misreadings of two different pieces of output:

- A **prediction** (`prediction`/`confidence`/`probabilities`) can be
  mistaken for a diagnosis — `PREDICTION_DISCLAIMER` says, in the response
  itself, that it is not one.
- An **explanation** (a LIME/SHAP saliency map) can be mistaken for evidence
  of *why*, medically, the model predicted what it did — `EXPLANATION_DISCLAIMER`
  says, in the response itself, that highlighting the regions a model's
  output was most sensitive to is not the same claim as identifying a causal
  driver of disease.
"""

from __future__ import annotations

PREDICTION_DISCLAIMER = (
    "This is a machine-generated research prediction, not a clinical diagnosis. "
    "It has not been clinically validated and must not be used as the sole basis "
    "for any medical decision. Always consult a qualified healthcare professional."
)

EXPLANATION_DISCLAIMER = (
    "This explanation shows which input regions most influenced the model's own "
    "output for this prediction. It does not establish a causal or clinical "
    "relationship between these regions and any disease process, and is not a "
    "clinical diagnosis. Always consult a qualified healthcare professional."
)
