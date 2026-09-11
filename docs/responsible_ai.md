# Responsible AI & ML safety (Phase 14)

Status: **Phase 14.** This document is the ML-safety half of the final security/responsible-AI
review (see [`docs/security.md`](security.md) for the application-security half). **Nothing in
this system has been clinically validated, and nothing in this document is a clinical claim** —
see [`docs/TECHNICAL_SPECIFICATION.md`](TECHNICAL_SPECIFICATION.md) §0. No accuracy/sensitivity/
specificity/AUC number appears anywhere in this document, because no real training run against a
full ADNI/OASIS/Kaggle dataset has been executed in this sandbox — every checkpoint exercised by
this codebase's own tests is a small, synthetic one used to verify pipeline correctness, not a
clinically meaningful model (see `docs/evaluation.md`/`docs/model_selection.md`'s own disclosure
of the same fact, and `docs/model_cards/` — deliberately empty: a model card is only honest to
publish once a real training run exists to describe).

## 1. Prediction is not diagnosis

Every `POST /api/v1/predictions` and `POST /api/v1/explanations` response carries a `disclaimer`
field (`core.disclaimers.PREDICTION_DISCLAIMER` / `EXPLANATION_DISCLAIMER`) **inline, in the JSON
body itself** — not only in this document or the OpenAPI description. A caller integrating against
this API programmatically, who never reads a markdown file, still receives the disclaimer on every
successful call:

> "This is a machine-generated research prediction, not a clinical diagnosis. It has not been
> clinically validated and must not be used as the sole basis for any medical decision. Always
> consult a qualified healthcare professional."

This was a deliberate design choice made this phase: a disclaimer that exists only in documentation
is a control a downstream engineer can simply never see. Putting it in the response schema
(`api.schemas.predict.PredictionResponse.disclaimer`) means it survives even if this document is
never read, and the CLI's `predict` command prints the same text to the terminal after every run
(`cli.main`).

The prediction output itself is also named and shaped to avoid implying diagnostic authority: the
field is `prediction` (a model output), not `diagnosis`; it takes one of the dataset's configured
class labels (e.g. `AD`/`CN`), never a phrase like "has Alzheimer's."

## 2. Model version is always exposed

Every prediction is traceable to the exact model that produced it — `model_version` and
`preprocessing_version` appear in every `PredictionResponse`/`ExplanationResponse`; their content
hashes (`model_config_hash` — queryable via `GET /api/v1/models`'s `ModelInfo.model_config_hash` —
and `preprocessing_config_hash`) plus `checkpoint_uri`/`device` are persisted with every audit row
(`db.audit_service.AuditService.record_prediction`/`record_model_version`), one step further back
than the response body itself. This is not a version *number* alone (which could mean different
things across a retrain) — the hash is tied to the actual content of the weights and preprocessing
config that ran, so "which model produced this prediction" is always an exact, verifiable answer
for anyone with audit-database access, never an approximation. See
[`docs/inference.md`](inference.md) and [`docs/monitoring.md`](monitoring.md) §2 for how this same
version tag drives drift/performance monitoring per-version.

## 3. Uncertainty and probability, exposed carefully

`PredictionResponse` returns the **full class-probability distribution** (`probabilities: {"AD":
0.82, "CN": 0.18}`), not only the top predicted class — deliberately, because a bare class label
with no confidence signal invites over-trusting a low-margin call exactly as much as a high-margin
one. `confidence` is the probability of the predicted class specifically, passed through
`inference.calibration.apply_calibration` — temperature scaling when a deployment has fit and
configured one (`method: "temperature"`), otherwise (`method: "none"`, the default) plain softmax,
**not** implicitly calibrated. This is disclosed, not hidden: a deployment that has not fit a
temperature is running on raw softmax confidence, a well-known source of overconfidence in deep
classifiers, and `calibration_method` is tracked on `inference.predictor.ModelVersionInfo` and
persisted with the audited model-version row (`db.audit_service`) — not yet surfaced on `GET
/api/v1/models`'s public schema, a real gap worth closing in a future pass so a caller can tell
which mode is active without reading the audit database directly.

**What is deliberately *not* exposed, and why**: raw pre-calibration logits (an internal detail
that invites misreading as a "truer" probability than the calibrated one this system actually
stands behind) and any single scalar "risk score" that would collapse the two-class distribution
into a false sense of a single, simple number. The full distribution plus one calibrated confidence
value is the considered middle ground between "too little information to judge reliability" and
"more numbers than the model can honestly support."

## 4. Explanations show association, not causality

`POST /api/v1/explanations` (LIME/SHAP, [`docs/explainability.md`](explainability.md)) returns
which input regions the model's own output was most sensitive to for *this* input — a statement
about the model's behavior, not about disease biology. `ExplanationResponse.disclaimer`
(`EXPLANATION_DISCLAIMER`) says this explicitly, in the response itself:

> "This explanation shows which input regions most influenced the model's own output for this
> prediction. It does not establish a causal or clinical relationship between these regions and any
> disease process, and is not a clinical diagnosis. Always consult a qualified healthcare
> professional."

**Optional LLM narrative summary, and why it doesn't weaken this guarantee**: `ExplanationResponse`
can also carry `narrative_summary` — a plain-language rendering of the same numeric output via the
OpenAI API, off by default (`explainability.llm_summary`, [`docs/explainability.md`](explainability.md)
§7.5). Its system prompt independently forbids a diagnostic or causal claim, but an LLM's own
output is not a control this codebase can fully guarantee the way a fixed constant string is —
`disclaimer` above, defined independently in `core.disclaimers` and never influenced by the LLM
call's outcome, remains the field this response's safety guarantee actually rests on.
`narrative_summary` is always additive and labeled, never a replacement for it.

This distinction is easy to elide by accident — a saliency map overlaid on a brain scan looks like
it is pointing at "the cause," and LIME/SHAP's own literature is explicit that both methods explain
a model's local decision boundary, not a ground-truth causal mechanism. The explanation-specific
disclaimer exists as its own field, separate from the prediction disclaimer, precisely because
conflating the two would understate this particular risk.

## 5. Documented limitations

- **No real clinical validation.** Every number this system could report (accuracy, AUC, ...)
  would come from a held-out test split of whichever training dataset was configured — not an
  independent clinical trial, not a regulatory-grade validation study, and not a claim about
  real-world diagnostic performance. See [`docs/TECHNICAL_SPECIFICATION.md`](TECHNICAL_SPECIFICATION.md)
  §0 and §10.
- **Single-modality input.** The model classifies from structural MRI alone — no clinical history,
  cognitive test scores (MMSE/CDR), genetic markers (APOE), CSF biomarkers, or PET imaging factor
  into the prediction, all of which a real diagnostic workup for Alzheimer's disease considers
  together. A prediction from this system reflects only what a structural MRI alone can show.
  Multi-modal fusion is a real limitation of the *current architecture*, not a claim that MRI alone
  is clinically sufficient.
- **Domain shift risk.** Like any imaging model, this one is only as reliable as its similarity to
  the distribution it was trained on. Its output on a scan from a materially different scanner,
  field strength, acquisition protocol, or reconstruction pipeline than its training distribution
  is unvalidated — the preprocessing pipeline normalizes intensity/spacing (`docs/inference.md`)
  but cannot fully eliminate this class of shift.
- **Binary/coarse class taxonomy** (whichever taxonomy the active dataset config defines — e.g.
  2-class CN/AD or a finer-grained staging). Real Alzheimer's progression is a continuum
  (preclinical → MCI → mild → moderate → severe); collapsing it into discrete classes is a
  simplification the model's output inherits, not something the API surface can compensate for.
- **No temporal/longitudinal signal.** Each prediction is from a single scan at a single point in
  time; the model has no memory of a patient's prior scans and cannot assess progression, rate of
  change, or trajectory — a real clinical assessment of Alzheimer's routinely does consider change
  over time.
- **Explainability is post-hoc, not a design constraint.** LIME/SHAP explain a model that was
  optimized purely for predictive accuracy, not for producing biologically-grounded explanations —
  see §4. A high-fidelity local explanation of an unreliable model is still unreliable.

## 6. Documented dataset limitations

Whichever training dataset a given deployment configures (`configs/data/dataset_{oasis,
kaggle_adni,...}.yaml` — see [`docs/TECHNICAL_SPECIFICATION.md`](TECHNICAL_SPECIFICATION.md) §1),
the following are structural limitations of the public Alzheimer's-MRI datasets this system is
designed against (ADNI, OASIS, Kaggle-hosted derivatives), not limitations this codebase can fix
through engineering alone:

- **Recruitment bias.** ADNI/OASIS participants are volunteers, generally recruited through
  academic medical centers in higher-income countries (predominantly the US) — not a random or
  representative sample of the global population that could plausibly present for Alzheimer's
  evaluation.
- **Demographic skew.** These cohorts are documented in their own literature as
  disproportionately white, English-speaking, and more educated than the general population;
  representation of Black, Hispanic, Asian, and Indigenous populations is well below their share
  of real-world dementia burden. A model trained on such data has no demonstrated basis for equal
  performance across demographic groups it was not adequately represented in.
- **Age-confound risk.** AD cohorts skew older than CN control cohorts in these datasets; a model
  can learn to separate classes partly on ordinary age-related brain change rather than
  disease-specific change (comparing an AD patient's brain to a much younger control's, rather than
  to an age-matched one) unless the training config explicitly age-matches or stratifies — this
  codebase's `datasets`/`data_pipeline` layers accept whatever splits/stratification a given
  dataset config defines; they do not enforce age-matching by themselves.
- **Diagnostic label uncertainty.** Ground-truth AD/MCI/CN labels in these cohorts come from
  clinical consensus diagnosis, which itself carries inter-rater variability and is an imperfect
  proxy for the true underlying neuropathology (only autopsy-confirmed neuropathology is
  definitive, and essentially no training dataset has that at scale). The model's ceiling is
  bounded by the reliability of the labels it was trained against.
- **Scanner/protocol homogeneity within, heterogeneity across, datasets.** ADNI standardized
  acquisition protocols across its own sites; a Kaggle-derived dataset assembled from multiple
  sources may not have. A model trained on one dataset's acquisition characteristics carries no
  guarantee of transferring to another's.

## 7. Bias & generalization risk summary

Combining §5 and §6: the most concrete, actionable risk this system carries is that **its
demonstrated performance, whatever it measures to be on a held-out split of its training
distribution, is not evidence of equivalent performance on a demographic group, scanner, or
disease-stage distribution underrepresented in that same training data.** This is a property of
the class of models and datasets involved, not a defect introduced by this codebase — but it is
this codebase's responsibility to document rather than let a downstream user discover the hard way.

Two concrete engineering mitigations already exist elsewhere in this system and are worth naming
here because they are load-bearing for this risk, not incidental:

- [`docs/monitoring.md`](monitoring.md)'s per-model-version drift/performance tracking means a
  real deployment has a mechanism to *detect* degraded performance against its actual live traffic
  distribution after the fact — it does not prevent the underlying dataset-bias risk, but it means
  the risk is monitored, not silent.
- The full probability distribution + calibrated confidence (§3) gives a downstream consumer more
  signal to weight a given prediction's reliability than a bare class label would — again, a
  mitigation of consequence, not of the underlying bias itself.

Anything beyond monitoring and calibrated uncertainty exposure — a fairness audit stratified by
demographic subgroup, an out-of-distribution detector, a formal external validation study — is
explicitly **not implemented** and would require both real (not synthetic) training data with
demographic annotations and a scope of work beyond this engineering phase.

## 8. Where these controls live in code

| Control | Implementation |
|---|---|
| Prediction disclaimer | `core.disclaimers.PREDICTION_DISCLAIMER` → `PredictionResponse.disclaimer` / CLI output |
| Explanation disclaimer | `core.disclaimers.EXPLANATION_DISCLAIMER` → `ExplanationResponse.disclaimer` |
| Model version exposure | `inference.predictor.ModelVersionInfo` → `model_version`/`preprocessing_version` + hashes, `GET /api/v1/models`, `db.audit_service` |
| Calibrated confidence + full probability distribution | `inference.calibration.apply_calibration`, `PredictionResult.confidence`/`.probabilities` |
| Drift/performance monitoring (mitigates §7's bias-transfer risk in production) | `docs/monitoring.md` |

See [`docs/security.md`](security.md) for the application-security half of this same phase's
review.
