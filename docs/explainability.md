# Explainability (LIME + SHAP)

Status: **Phase 8 — LIME and SHAP implemented.** Nothing on this page is a claim about
*medical* interpretation — see §7 "Model explanation vs. medical interpretation" and
[`docs/TECHNICAL_SPECIFICATION.md`](TECHNICAL_SPECIFICATION.md) §0.

## 1. What this is

`explainability.service.ExplainabilityService` is the single entry point: give it a trained
checkpoint and one MRI scan, and it returns the model's prediction, class probabilities, and
LIME and/or SHAP explanations for one target class (by default, the predicted class). Runnable as
a library call, `ad-mri explain run`, or `python -m ad_mri.explainability.explain`.

```bash
ad-mri explain run \
    --model artifacts/checkpoints/final-model/best.pt \
    --image data/raw/subject-042/scan.nii.gz \
    --background-manifest data/splits/<name>/train.csv \
    --methods lime,shap
```

It is deliberately independent of the MLflow-registry-backed `inference.model_loader`/
`inference.predictor` (still stubs pending the model-registry phase) — it loads a checkpoint
directly, the same way `evaluation.evaluate`/`training.cross_validation` already do. It *does*
reuse the identical deterministic preprocessing pipeline every other phase uses
(`datasets.transforms.build_transforms(..., train=False)`) — there is no second,
explainability-only preprocessing implementation (requirement: "keep preprocessing identical to
inference preprocessing").

## 2. What "features" mean here

Every voxel in a preprocessed MRI volume is one input feature to the CNN. There is no
hand-engineered feature set (no "hippocampal volume", no "cortical thickness") — the model
consumes raw, normalized voxel intensities after resampling/orientation/normalization
(`datasets.preprocessing_config.PreprocessingConfig`, unchanged since Phase 3). Consequently:

- A LIME **superpixel** is a spatially-contiguous group of voxels *on one representative 2D
  slice*, found by unsupervised image segmentation (quickshift or SLIC) — it has no anatomical
  label attached by the system. A reviewer with neuroanatomical training may recognize it as
  "temporal lobe" or "ventricle"; the system itself does not know or assert that.
- A SHAP **attribution value** is one signed number per voxel (or, for the aggregated view, per
  downsampled region) — how much that voxel's actual intensity pushed the model's output toward
  or away from the target class, relative to the background/reference distribution. It is a
  statement about the model's computation, not a statement about tissue.

## 3. Why not just flatten the volume

A preprocessed volume is typically ~10^5–10^7 voxels. Two very different scaling problems follow,
and this system solves them with two different (and deliberately different) representations —
never with a naive flatten-everything approach:

- **LIME** needs *hundreds of forward passes over perturbed input* to fit a local linear surrogate
  model, and a "perturbation" only makes interpretive sense as a 2D superpixel mask (LIME's image
  explainer is fundamentally a 2D-image tool). Perturbing an entire 3D volume's worth of
  superpixels would multiply the already-expensive sample budget by the number of slices, for a
  return an MRI reviewer can't easily inspect anyway (a 3D superpixel map has no single readable
  view). This system instead perturbs **one representative 2D slice**
  (`explainability.slicing`, config-selected axis/index — default: the middle slice) while every
  other voxel stays the real, preprocessed input, and every perturbed candidate is scored by a
  genuine forward pass through the real 3D model — not a separate, unfaithful 2D proxy model. See
  §6 for what this does and does not tell you about the rest of the volume.
- **SHAP** (`shap.GradientExplainer`) has no such problem — it computes attributions directly from
  gradients w.r.t. the *entire* input tensor in one pass per background/interpolation sample, so
  every voxel gets its own attribution with no flattening or slicing needed for the computation
  itself. Slicing only re-enters for *visualizing* the result (the overlay PNG shows one 2D slice
  of the full per-voxel attribution volume) and for the *aggregated* view (§5), where downsampling
  is a deliberate interpretability choice, not a computational necessity.

## 4. LIME

`explainability.lime_explainer.explain_with_lime` — `lime.lime_image.LimeImageExplainer` applied
to the representative slice:

1. Extract the slice (`explainability.slicing.extract_slice`), rescale it to `[0, 1]` for display
   and segmentation.
2. Segment it into superpixels (config: `quickshift` or `slic`, with their standard parameters).
3. LIME perturbs superpixels (hides/reveals them) to build a neighborhood of `num_samples`
   candidate images.
4. Each candidate is de-normalized back to the model's real input range, spliced into a **copy**
   of the real volume at the representative slice (`explainability.slicing.insert_slice` — every
   other voxel is untouched), and scored by the real model (`classifier_fn`).
5. LIME fits a local Ridge regression on (superpixel-presence, model-output) pairs; its
   coefficients are the reported superpixel importances for the target class.

**Result fields** (`LimeExplanationResult`): `top_superpixels` (ranked by `|weight|` —
"identify image regions/features contributing to the prediction"), `surrogate_r2` (the local
linear model's own fit quality — low values mean *trust this explanation less*, see §6),
`local_prediction` (the surrogate's own probability estimate, for comparing against the real
model's probability), and an overlay PNG (`explainability.overlays.render_lime_overlay`) with
superpixel boundaries drawn on the slice.

**Reproducibility**: `LimeConfig.seed` pins both `LimeImageExplainer`'s `random_state` and
`explain_instance`'s `random_seed` (which also seeds the segmentation algorithm) — identical
model + volume + config always produce identical superpixel weights (verified in
`tests/explainability/test_lime_explainer.py::test_lime_is_reproducible_given_the_same_seed`).

## 5. SHAP

`explainability.shap_explainer.explain_with_shap` — `shap.GradientExplainer` (default) or
`shap.DeepExplainer`, config-selected:

1. Draw a background/reference sample set from a manifest (`explainability.sampling.load_volume_batch`
   — deterministically sampled, preprocessed the same way as the instance being explained; "a
   config-defined background/reference sample set, versioned" — the manifest path is recorded on
   every explanation).
2. Compute expected-gradient attributions for every voxel, for the target class, in one call.
3. Save the full per-voxel attribution array (`shap_values.npy`) and render one slice of it as a
   diverging-colormap overlay (`explainability.overlays.render_shap_overlay`).

**Aggregated/global importance** (`explainability.shap_explainer.compute_global_shap_importance`,
`ad-mri explain global-shap`): runs the same explainer across several samples for one target
class, averages `|attribution|` per voxel, then **average-pools down to a coarse region grid**
(`region_pool_size`, default 4 voxels per side) before saving/rendering. This is "technically
meaningful" specifically in this multi-sample, downsampled form — see §6 for why a raw per-voxel
ranking across a handful of samples is not.

## 6. Limitations (read before trusting an explanation)

**LIME**

- **One slice, not the volume.** The explanation only covers the representative 2D slice
  (default: the middle slice along `slice_axis`). A lesion or region visible on a different slice
  is invisible to this explanation entirely — it is not "explained as unimportant", it simply was
  never shown to LIME. Changing `slice_axis`/`slice_index` and re-running produces a genuinely
  different (and equally partial) explanation.
- **Superpixel granularity is a segmentation-algorithm artifact.** Quickshift/SLIC parameters
  (kernel size, compactness, target segment count) determine what counts as "one region" — the
  same underlying anatomy can be split or merged differently under different settings, and the
  importance attached to "one region" is only meaningful relative to that specific segmentation.
- **Linear surrogate fidelity is not guaranteed.** LIME assumes the model behaves approximately
  linearly in the *local* neighborhood it samples; `surrogate_r2` on the result is the actual fit
  quality — a low value means the reported weights describe the surrogate, not necessarily the
  real model, well. This system reports `surrogate_r2` on every result rather than hiding it.
- **Sample-budget/runtime tradeoff.** Each of `num_samples` perturbations costs one real 3D
  forward pass; a realistic MRI volume's forward pass is not free, so `num_samples` (default 200)
  trades faithfulness against runtime. Too few samples under-fits the surrogate; there is no
  universal "enough" — the config default is a starting point, not a validated minimum.
- **Perturbation validity.** LIME's superpixel masking assumes "hiding" a region (replacing it
  with a mean/fudge value) produces a semantically valid input. For MRI, a fudged superpixel is
  not a realistic anatomical absence — it's an out-of-distribution patch the model was never
  trained to see, which can produce misleadingly large or small importances for regions the fudge
  color happens to resemble or clash with.

**SHAP**

- **Background-set dependence.** `GradientExplainer`'s attributions are defined *relative to* the
  background/reference distribution — a different (equally reasonable) background sample set
  produces different attribution values for the same instance. There is no single "correct"
  background for MRI; this system requires it to be explicit and versioned (manifest path +
  sample count recorded on every explanation) specifically so this dependence stays visible rather
  than hidden behind an unstated default.
- **No exact completeness guarantee for `GradientExplainer`.** Unlike `DeepExplainer`/
  `KernelExplainer`, the expected-gradients approximation `GradientExplainer` uses does not
  guarantee `sum(attributions) == prediction - base_rate`. `attribution_sum` on the result is
  reported as a diagnostic, not as a validated identity.
- **Per-voxel global ranking is not statistically meaningful.** A realistic MRI volume has
  hundreds of thousands to millions of voxels; averaging `|attribution|` for a handful of samples
  and ranking individual voxels would produce a ranking dominated by per-sample noise, not a
  stable population-level signal. This is why `compute_global_shap_importance` reports a
  **downsampled region grid**, never a flat per-voxel ranking — and even the region grid should be
  read as "roughly where attribution tends to concentrate across these particular samples", not as
  a validated feature-importance ranking with statistical guarantees. More samples narrow this
  uncertainty; this system does not claim any fixed sample count is "enough".
- **Gradient saturation.** Gradient-based methods can under-attribute importance to inputs where
  the network's activation has saturated (e.g. deep in a ReLU/BatchNorm stack) — a real limitation
  of the *class* of method `GradientExplainer` belongs to, independent of this implementation.

**Both methods, together**

- **LIME and SHAP are not guaranteed to agree**, and this system does not assert one method is
  "more correct" (see `docs/TECHNICAL_SPECIFICATION.md` §7, "Comparison surfacing") — they answer
  related but different questions (a local linear approximation vs. a gradient-based attribution)
  and disagreement between them is itself informative, not a bug in either.
- **An explanation describes the model, not the disease.** Both methods explain *why this specific
  model produced this specific output on this specific input* — they say nothing about whether the
  model's reasoning corresponds to genuine neuroanatomical or clinical significance. See §7.

## 7. Model explanation vs. medical interpretation

> **This system produces no clinical or diagnostic claims anywhere**, and explanation output is no
> exception. A LIME/SHAP explanation is a description of *this model's* computation on *this
> input* — it is not validated against, and must not be presented as, a radiological or clinical
> finding. Concretely:
>
> - A highlighted region is "a region the model's output was sensitive to" — never "the lesion" or
>   "the pathology".
> - A high-confidence prediction with a highlighted region that looks anatomically plausible is
>   *consistent with* that anatomy mattering to the model; it is not evidence the model is
>   reasoning correctly, and it is not a second opinion.
> - Explanation review by a qualified clinician is about whether the model's stated reasoning is
>   *plausible*, not about confirming a diagnosis the explanation supposedly reveals.
>
> See [`docs/TECHNICAL_SPECIFICATION.md`](TECHNICAL_SPECIFICATION.md) §0 for the project-wide
> non-goals this follows from.

## 7.5. Optional LLM narrative summary

`POST /api/v1/explanations` can attach a `narrative_summary` field — a plain-language rendering of
the numeric explanation output above, generated via the OpenAI API. **Off by default**: `None`
unless `OPENAI_API_KEY` is configured (`Settings.openai_api_key`); `POST /explanations` behaves
identically either way, and this was never load-bearing for the endpoint's own success/failure.

- **Privacy boundary**: `explainability.llm_summary._build_payload` is the complete, exhaustive
  set of what is ever sent — the predicted class, calibrated probabilities, and each requested
  method's already-aggregated region-importance numbers (LIME's ranked `segment_id`/`weight`
  pairs; SHAP's `attribution_sum`/`explainer_type`). **Never sent**: the uploaded file's name or
  path, any patient identifier, or a single raw pixel/voxel value — the per-voxel SHAP array never
  leaves disk. `test_llm_summary.py::test_payload_never_includes_a_file_path_or_raw_pixel_data`
  makes this an enforced, tested guarantee, not a claim.
- **Responsible-AI boundary**: the system prompt explicitly forbids a diagnostic or causal claim,
  mirroring §7 above — but an LLM's own output is not a control this codebase can fully guarantee.
  `ExplanationResponse.disclaimer` (defined independently in `core.disclaimers`) remains the
  authoritative safety statement regardless of what `narrative_summary` says.
- **Best-effort, like every other optional side call in this API** (`api.audit`,
  `record_model_version_seen`): a missing key, network failure, timeout, or SDK error degrades to
  `None`, never fails the request whose actual job (the LIME/SHAP explanation itself) already
  succeeded. See `explainability/llm_summary.py`'s module docstring.
- **Never exercised with a real network call in the automated test suite** — `tests/conftest.py`
  forces `OPENAI_API_KEY` blank for the whole run regardless of a real key configured locally;
  `test_llm_summary.py` and one integration test stub only the `openai` SDK's transport layer, the
  actual external boundary this feature isolates.
- Requires the optional `llm` extra (`pip install -e ".[llm]"`) — lazily imported, exactly like
  `lime`/`shap`/`torch` elsewhere in this codebase; nothing else depends on it being installed.

## 8. Storage: what's recorded and where

Every call to `ExplainabilityService.explain()` writes one `manifest.json` under
`artifacts/explanations/<input_id>-<timestamp>/` (or `--output-dir`) recording exactly what Phase
8 requires to be stored:

| Field | Where |
|---|---|
| Model version | `model_version`, `model_config_hash` |
| Input ID | `input_id` (defaults to the image filename; overridable — e.g. a subject/visit ID) |
| Preprocessing version | `preprocessing_version`, `preprocessing_config_hash` |
| Explanation method(s) | `methods.lime`, `methods.shap` (present only for the methods actually run) |
| Explanation parameters | `methods.<name>.config_hash` (and, on `LimeConfig`/`ShapConfig`, every knob — segmentation algorithm, sample counts, background size, seed) |
| Generated artifacts | `artifacts` (top-level: `prediction`, `manifest`, plus each method's own — `lime_overlay`; `shap_values`, `shap_overlay`) |

Alongside `manifest.json`: `prediction.json` (predicted class, confidence, full class-probability
distribution), `lime_overlay.png`, `shap_overlay.png`, and `shap_values.npy` (the full per-voxel
attribution array — not embedded in JSON; even a modest MRI volume is too large for that).

## 9. Reproducibility guarantees

Mirrors `evaluation.evaluate`'s guarantees (`docs/evaluation.md` §5):

- **Preprocessing match check**: `ExplainabilityService` refuses to run (`ConfigurationError`) if
  the given `--preprocessing-config`'s content hash doesn't match the one the checkpoint's
  training-provenance metadata recorded.
- **Class labels from the checkpoint, never re-derived**: the same convention as
  `evaluation.evaluate` — a checkpoint without a recorded `class_names` mapping is rejected rather
  than guessed at.
- **Deterministic given a seed**: identical model + image + configs + `--seed` reproduce identical
  LIME superpixel weights and identical SHAP background sampling.
