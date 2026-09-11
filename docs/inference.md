# Inference Engine

Status: **Phase 9 — production inference engine implemented; Phase 10 wired it behind the real REST
API** (`POST /api/v1/predictions`, `POST /api/v1/explanations`, `GET /api/v1/models` — see
[`docs/api.md`](api.md) for the full API reference). No performance number in this document is a
claim about real-world diagnostic accuracy — everything here is a methodology description. See
[`docs/TECHNICAL_SPECIFICATION.md`](TECHNICAL_SPECIFICATION.md) §0, §6.

## 1. Pipeline

```
MRI file -> validation -> preprocessing -> (cached) model loading -> inference
  (torch.inference_mode(), no gradients) -> probability calibration (if configured)
  -> prediction + confidence -> optional LIME/SHAP explanation
```

`inference.predictor.InferenceEngine.predict()` is the single implementation of this pipeline —
called identically by `ad-mri predict`, `python -m ad_mri.inference.predict`, and the FastAPI
`POST /api/v1/predictions` endpoint. There is no second, HTTP-only copy of the logic.

```bash
ad-mri predict --model artifacts/checkpoints/final-model/best.pt \
    --image data/raw/subject-042/scan.nii.gz

# equivalently:
python -m ad_mri.inference.predict --model .../best.pt --image .../scan.nii.gz
```

## 2. Requirement-by-requirement

| Requirement | Where |
|---|---|
| Load model from a versioned artifact | `inference.model_loader.ModelLoader` loads a checkpoint written by `modeling.checkpoint.save_checkpoint` — the same self-describing artifact `evaluation.evaluate`/`explainability.service` already load, bundled with its architecture config and a content hash. |
| Load the exact preprocessing config associated with that model | `ModelLoader` hash-checks the given preprocessing config against the checkpoint's own training-provenance metadata and refuses to load on a mismatch (`ConfigurationError`) — the same guarantee `evaluation.evaluate`/`explainability.service` give. |
| Validate input | `datasets.validation.validate_image_for_pipeline` — the same fail-fast checks (existence, readable NIfTI, dimension/voxel-count bounds, non-degenerate affine) every other phase uses. |
| Handle invalid files gracefully | `DataValidationError`/`PreprocessingError` are raised as well-typed exceptions, never an uncaught crash; the API maps them to HTTP 422 with a generic (non-leaking) message. |
| CPU/GPU support | `device_choice: "auto" \| "cpu" \| "cuda"`, resolved once at engine construction. |
| Inference mode / no gradients | Every forward pass runs inside `torch.inference_mode()`. |
| Configurable timeout | `timeout_seconds` — the full pipeline runs in a worker thread; `Future.result(timeout=...)` raises `InferenceError` if it doesn't finish in time. |
| Structured logging | `structlog`, consistent with every other phase — see §4 for what's deliberately excluded. |
| Request ID | Every `predict()` call is tagged with a `request_id` (caller-supplied or generated); the API reuses the same ID `RequestIDMiddleware` already bound for the HTTP request, returned in both the JSON body and the `X-Request-ID` header. |
| Deterministic response structure | `PredictionResult` — see §3. |
| Do not retrain during inference | The cached model is `.eval()`'d exactly once at load time; nothing in `inference/` ever calls `.train()`, `.backward()`, or touches an optimizer. |
| Cache model in memory | `ModelLoader` caches by `(checkpoint_path, preprocessing_config_path, device)`; the API's `InferenceEngine` is itself process-cached (`functools.lru_cache`, mirroring `core.config.get_settings`). |
| Health check | `GET /api/v1/ready` (aggregate) / `InferenceEngine.health_check()` — runs a real dummy forward pass, not just "is the object present". See `docs/api.md`. |
| Model version endpoint | `GET /api/v1/models` / `InferenceEngine.model_version_info()`. |
| Unit/integration tests | `tests/unit/test_inference_*.py` (real, tiny checkpoints, no mocks), `tests/integration/test_predictions_api.py`/`test_explanations_api.py` (real `TestClient`). |

## 3. `PredictionResult` — the clean response schema

```json
{
  "request_id": "8f14e45f-...",
  "model_version": "1.0.0",
  "preprocessing_version": "1.0.0",
  "prediction": "AD",
  "confidence": 0.87,
  "probabilities": {"AD": 0.87, "CN": 0.13},
  "processing_time_ms": 142.3,
  "explanation_available": false,
  "explanation_id": null,
  "timestamp": "2026-01-01T12:00:00+00:00"
}
```

`api.schemas.predict.PredictionResponse` mirrors this exactly — the HTTP boundary adds no fields
and drops none. `prediction` is always the model's own argmax; a caller cannot override which
class gets reported as "the prediction" (only which class an *explanation* covers, if requested,
is choosable — see §5). `preprocessing_version` and `model_version` are the human-set semantic
versions on the config/checkpoint that produced this result (their content hashes are available via
`GET /api/v1/models`, not embedded in every prediction — kept lean).

## 4. What never gets logged

An MRI's file path — and certainly a clinical PACS export's original filename — can encode a
patient identifier in a way this project's already-de-identified manifest `subject_id`s do not
(e.g. `john_doe_1957.nii.gz`). Concretely, this codebase:

- Never logs an uploaded file's original filename. The API writes an upload to a temp file named
  generically (`upload<ext>`) — only the extension (needed for format detection) survives from the
  client-supplied name.
- Never logs a raw file path. Every log line identifies an image by
  `inference.predictor.redact_image_reference()` — a one-way SHA-256 hash (truncated) of the
  resolved path, never the path itself.
- Never logs `str(exc)` for a validation/preprocessing failure, even though the underlying
  exception's own message (useful to the *caller*, who already has the path) may legitimately
  include the path — only the exception's type name is logged.
- The temp file an upload is written to is always deleted (`finally`), whether the request
  succeeded or failed.

Every log line still carries a `request_id`, so operational correlation/debugging doesn't require
any of the above.

## 5. Probability calibration

Off by default (`method: "none"`, plain softmax — bit-identical to every earlier phase's
uncalibrated path). Pass `--calibration-config` to enable temperature scaling
(`inference.calibration.CalibrationConfig(method="temperature", temperature=T)`): softmax(logits /
T), the standard minimal post-hoc calibration method (Guo et al., ICML 2017). `T` is fit offline
against held-out validation logits/labels (`inference.calibration.fit_temperature`, real
NLL-minimizing optimization via `scipy.optimize.minimize_scalar`) — never during inference itself,
and temperature scaling never changes the predicted class (softmax is monotonic in a positive
rescaling of the logits).

## 6. Optional explanation (XAI)

Two paths, both against the *same, already-loaded* checkpoint (`ExplainabilityService.from_loaded_model`
— never a second disk load):

- **Inline, best-effort** (`InferenceEngine.predict(include_explanation=True)`, `ad-mri predict --explain`):
  a failed explanation never fails the underlying prediction — it only sets
  `explanation_available: false`. Library/CLI use only; the HTTP API does not expose this — see below.
- **Dedicated, loud** (`InferenceEngine.generate_explanation()`, `POST /api/v1/explanations` — see
  `docs/api.md`): the caller's entire request *is* the explanation, so a failure is a real error
  (mapped to a proper HTTP status by `api.exception_handlers`), never a silent "unavailable".

Both default to explaining the model's own predicted class, or a caller-chosen class
(`target_class`/`explanation_target_class`), independent of what was predicted. `explanation_id`
locates the full LIME/SHAP artifact bundle under
`--explanation-output-dir/<explanation_id>/manifest.json` — see `docs/explainability.md`.

## 7. Reproducibility & versioning guarantees

Identical to `evaluation.evaluate`/`explainability.service` (same checks, same code pattern):

- **Preprocessing match check**: refuses to load if the preprocessing config's content hash
  doesn't match what the checkpoint was trained with.
- **Class labels from the checkpoint, never re-derived**: a checkpoint without a recorded
  `class_names` mapping is rejected, not guessed at.
- **Deterministic given identical inputs**: no randomness anywhere in the pipeline (no
  augmentation at inference — `train=False`; calibration is a fixed rescaling).

## 8. Deployment configuration

Which checkpoint/preprocessing/calibration config the API serves is environment configuration
(`INFERENCE_MODEL_PATH` etc. — see `.env.example`), not a versioned config file: there is nothing
about "which model this deployment serves" that needs its own content hash the way the model's
*architecture* or *preprocessing* do. `INFERENCE_MODEL_PATH` unset (the default) means "not
configured yet" — `/predictions` and `/explanations` respond `503`, `/models` returns an empty
list (not an error), and `/ready` reports the omission only if a model *was* expected (see
`docs/api.md`) — rather than crashing the process at startup; `/healthz`/`/health` and `/metrics`
stay unaffected either way.
