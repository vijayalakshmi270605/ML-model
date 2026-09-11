# REST API

Status: **Phase 10 — production-grade FastAPI backend implemented** (Phase 13 adds one endpoint,
`POST /api/v1/monitoring/ground-truth` — see [`docs/monitoring.md`](monitoring.md) §5; Phase 14 adds
scoped multi-key authorization, security headers, and the `disclaimer` field below — see
[`docs/security.md`](security.md) for the full review). No performance number in this document is a
claim about real-world diagnostic accuracy, and nothing here is a clinical claim — see
[`docs/TECHNICAL_SPECIFICATION.md`](TECHNICAL_SPECIFICATION.md) §0 and
[`docs/responsible_ai.md`](responsible_ai.md).

The API is a thin HTTP boundary over `inference.predictor.InferenceEngine` and
`explainability.service.ExplainabilityService` — **every route handler calls into one of those two,
and neither contains business/model logic itself** (requirement: "the API must call the inference
service rather than duplicating model logic" / "keep business logic separate from API routes").
Start it with `ad-mri serve` (`uvicorn ad_mri.api.main:app`); see README "Run the API".

## 1. Endpoints

| Method | Path | Auth | Calls into | Success |
|---|---|---|---|---|
| `POST` | `/api/v1/predictions` | API key, scope `predict` (if `API_KEY_REQUIRED=true`) | `InferenceEngine.predict()` | `201` |
| `POST` | `/api/v1/explanations` | API key, scope `predict` (if `API_KEY_REQUIRED=true`) | `InferenceEngine.generate_explanation()` | `201` |
| `GET` | `/api/v1/models` | none | `InferenceEngine.model_version_info()` | `200` |
| `POST` | `/api/v1/monitoring/ground-truth` | API key, scope `label` (if `API_KEY_REQUIRED=true`) | `MonitoringService.record_ground_truth()` | `201`/`404` |
| `GET` | `/api/v1/health` (+ unversioned `/healthz`) | none | liveness only | `200` |
| `GET` | `/api/v1/ready` (+ unversioned `/readyz`) | none | DB + inference engine | `200`/`503` |
| `GET` | `/api/v1/version` | none | — | `200` |
| `GET` | `/metrics` | none | — | `200` (Prometheus) |

`/api/v1/explain/{prediction_id}` and the old singular `/api/v1/predict` from earlier phases are
gone — replaced by the resource-oriented `/predictions`/`/explanations` above. `GET /api/v1/models`
is real (Phase 10) but is *not* the full MLflow Model Registry (multiple versions, promotion
history) — that remains a separate, not-yet-implemented phase (`model_registry.client`); with no
registry, `/models` reports at most the one checkpoint the process is actually serving, and an
empty list (not an error) when none is configured.

## 2. Pydantic schemas & request validation

Every request/response body is a Pydantic model (`api.schemas.*`) — FastAPI validates a request
body/query param against its declared type before the route body ever runs; a validation failure
short-circuits with FastAPI's own `422` before `api.exception_handlers` is even involved.

| Schema | Used by |
|---|---|
| `schemas.predict.PredictionResponse` | `POST /predictions` — mirrors `inference.predictor.PredictionResult` field for field, plus `disclaimer` (see §14). |
| `schemas.explain.ExplanationResponse` | `POST /explanations` — also carries its own `disclaimer` (see §14). |
| `schemas.models.ModelInfo` | `GET /models`. |
| `schemas.health.LivenessResponse` / `ReadinessResponse` / `VersionResponse` | `/health`, `/ready`, `/version`. |
| `schemas.common.ErrorResponse` | Every error response (see §5). |

## 3. File validation & maximum upload size

Both upload endpoints (`predictions`, `explanations`) share one implementation
(`api.upload_utils.save_upload_within_limit`):

1. Streamed to disk in 1 MiB chunks — the request body is never buffered whole in memory.
2. Aborted with `413` the moment the running total exceeds `Settings.max_upload_size_bytes`
   (`MAX_UPLOAD_SIZE_MB`, default 200 MB) — before the rest of the upload is even read.
3. Rejected with `422` if the upload is empty.
4. Content/format validation itself (readable NIfTI/DICOM, dimension and voxel-count bounds, a
   non-degenerate affine) happens downstream, inside `InferenceEngine`/`ExplainabilityService`
   (`datasets.validation.validate_image_for_pipeline` — the same check every phase uses), raising
   `DataValidationError`/`PreprocessingError`, mapped to `422` by `api.exception_handlers`.

## 4. Async design

Route handlers are `async def` so upload I/O (`await file.read(...)`) never blocks the event loop.
The actual inference/explanation call is CPU-bound and synchronous
(`InferenceEngine.predict`/`generate_explanation`) — it cannot be made `async def` without penalty,
so it runs via `await asyncio.to_thread(...)` rather than being called directly, which would
otherwise serialize every concurrent request onto the single event-loop thread for the full
duration of each inference call.

## 5. Centralized exception handling

Route handlers do not catch `InferenceEngine`/`ExplainabilityService` exceptions themselves — they
let them propagate, and `api.exception_handlers.register_exception_handlers` maps every
`core.exceptions.AppError` subclass to a status code and a consistent `ErrorResponse` body
(`{"detail": ..., "error_code": ...}`) **once**, application-wide:

| Exception | Status | `error_code` |
|---|---|---|
| `DataValidationError` | 422 | `VALIDATION_FAILED` |
| `PreprocessingError` | 422 | `PREPROCESSING_FAILED` |
| `ModelNotFoundError` | 404 | `MODEL_NOT_FOUND` |
| `InferenceError` (timeout) | 504 | `INFERENCE_TIMEOUT` |
| `InferenceError` (other) | 500 | `INFERENCE_FAILED` |
| `ExplainabilityError` | 500 | `EXPLANATION_FAILED` |
| `ConfigurationError` | 500 | `CONFIGURATION_ERROR` |
| any other `AppError` | 500 | `INTERNAL_ERROR` |
| "engine not configured" (a router-level precondition, not a raised domain exception) | 503 | — |
| rate limit exceeded | 429 | `RATE_LIMIT_EXCEEDED` |
| upload too large | 413 | — |

Every handler logs the exception's **type** and the current `request_id` — never `str(exc)` (see
§7 and `inference.predictor`'s module docstring: a domain exception's own message can legitimately
include a filesystem path, which must never reach a log line or a response body here).

## 6. API versioning

`Settings.api_v1_prefix` (default `/api/v1`) prefixes every versioned router — never hardcoded at
each call site. `GET /api/v1/version` reports the running application version
(`ad_mri.__version__`) and the active API version segment. A future `v2` would add a parallel
router set under a new prefix rather than mutating `v1`'s contracts in place.

## 7. Structured logging, request IDs, and patient-information handling

Every request is tagged with a `request_id` (`api.middleware.request_id.RequestIDMiddleware` —
reuses an incoming `X-Request-ID` header if the caller supplied one, else generates one; bound to
`structlog.contextvars` for the duration of the request and echoed back in the `X-Request-ID`
response header, `Access-Control-Expose-Headers`-listed so a browser client can read it). A
prediction's own `request_id` field always matches this header exactly.

**No patient-sensitive information ever reaches a log line or a response body**, an invariant that
spans this whole layer, not just the inference engine:

- An uploaded file is written to a temp file under a **generic** name (`upload<ext>`) —
  `api.upload_utils`/the routers never write, log, or echo the client's original filename.
- Every log line identifies an image by `inference.predictor.redact_image_reference()` — a one-way
  hash of the resolved path, never the path itself.
- `api.exception_handlers` never logs `str(exc)` for any exception — only its type name.
- The temp upload directory is always removed (`finally: shutil.rmtree(...)`) whether the request
  succeeded or failed — see §9.

See `docs/inference.md` §4 for the full rationale.

## 8. OpenAPI documentation

Auto-generated at `/openapi.json`, browsable at `/docs`/`/redoc`. `create_app()` sets
title/version/description/contact/license plus per-tag descriptions
(`predictions`/`explanations`/`models`/`meta`/`health`); every schema field has an inferred or
explicit description. The API-key requirement is a real OpenAPI `securityScheme`
(`api.security.api_key_scheme`, an `APIKeyHeader`) — `/docs`' "Authorize" button targets it, so a
developer sees exactly which endpoints require a key without reading the source.

## 9. Authentication & authorization architecture

`API_KEY_REQUIRED=false` (default, local dev) is a no-op — every endpoint is open. Set it `true`
(required in `APP_ENV=prod`, enforced by `Settings._validate_production_secrets`) and requests to
`POST /predictions`/`POST /explanations`/`POST /monitoring/ground-truth` must present the header
named by `API_KEY_HEADER_NAME` (default `X-API-Key`), checked by `api.deps.require_scope`, or get
`401`. Read-only endpoints (`/models`, `/health`, `/ready`, `/version`) stay open regardless —
metadata/health surfaces, not compute.

**Two authorization modes, chosen per-deployment, not per-request:**

- **Legacy single key** (`API_KEY`, Phase 10-13 behavior, still fully supported): one shared
  secret. A matching key is a superuser holding every scope — this mode has no concept of "this
  key can predict but not label."
- **Scoped multi-key** (`API_KEYS`, Phase 14): a JSON object mapping each key to the list of
  `core.constants.ApiScope` values it holds (`predict` | `label`) —
  `API_KEYS={"ingestion-pipeline-key": ["predict"], "clinician-key": ["predict", "label"]}`. Lets a
  deployment hand an automated ingestion pipeline a `predict`-only key while reserving `label`
  (submitting ground truth, which shapes `docs/monitoring.md`'s drift detection) for a clinician's
  own key — least privilege, not "everyone with any key can do everything."

`API_KEYS` set means every route uses it exclusively for that deployment (it does not merge with
`API_KEY`); leave it unset (the default) to keep the legacy single-key behavior. Each router
declares the scope it needs directly (`Depends(require_scope(ApiScope.PREDICT))` /
`ApiScope.LABEL)`) — see `api.routers.predictions`/`explanations`/`monitoring`.

**401 vs. 403 is a real, meaningful distinction** a caller's own error handling can act on: an
unrecognized key (or scoped mode with no key at all) is `401` (not authenticated); a *recognized*
key that lacks the scope a route requires is `403` (authenticated, not authorized for this action).

This remains a shared-secret scheme (per-key, not per-user identity) — no user accounts, sessions,
or OAuth. The seam for a stronger scheme (per-user keys persisted in Postgres, OAuth2/JWT via
`fastapi.security.OAuth2PasswordBearer`) is narrow and already isolated: `api.security` (the
OpenAPI declaration) and `api.deps.require_scope`/`require_api_key` (the enforcement) are the only
two places that would change — every route already depends on one of those by name, not on how it
authenticates.

## 10. Rate limiting architecture

Enforced by default (`Settings.rate_limit_per_minute`, default 60/minute; `<= 0` disables it) via
`api.middleware.rate_limit.RateLimitMiddleware` — an in-memory, per-process sliding-window counter
keyed by API key (if presented) else source IP, exempting `/health(z)`, `/ready(z)`, and `/metrics`
so infra polling can never be throttled or throttle a real client. Exceeding the limit returns
`429` with a `Retry-After` header.

This implementation is correct for **one process**. A multi-instance deployment shares no state
between processes (each instance enforces the limit independently). The seam to close that gap:
swap the in-memory counter for a Redis-backed one (`INCR`+`EXPIRE`, or a sorted-set sliding window)
— `REDIS_URL` is already configured (used by Celery today) — without touching
`RateLimitMiddleware`'s call sites.

## 11. CORS configuration

`CORSMiddleware` is the **outermost** middleware (see `api.main`'s module docstring for the full
ordering rationale) so every response — including a `429` from the rate limiter or a `5xx` from a
centralized exception handler — still carries correct CORS headers; without this, a browser client
would see those as an opaque, unreadable network error rather than a real HTTP status it can act
on. `ALLOWED_ORIGINS` (comma-separated, default empty — **no** origin is allowed until configured)
is the only origin allow-list; methods/headers are the explicit set the API actually uses (`GET`,
`POST`; `Content-Type`, `X-API-Key`, `Authorization`), not a wildcard, and `X-Request-ID` is
explicitly exposed so client-side JS can read it.

## 12. Secure file handling & temporary file cleanup

See §3 for size/streaming and `docs/inference.md` §4 for the filename-redaction discipline. Every
upload's temp directory is created fresh per request (`tempfile.mkdtemp`) and removed in a
`finally` block in both `predictions` and `explanations` routers — on success, on a `4xx`/`5xx`,
or on an unexpected exception, the temp file never outlives the request. LIME/SHAP artifacts an
`/explanations` call *keeps* (overlay PNGs, raw SHAP values, `manifest.json`) are a deliberate,
separate, longer-lived store (`Settings.inference_explanation_output_dir`) — see
`docs/explainability.md` §8 — not the request-scoped upload copy this section covers.

## 13. Testing

- `tests/unit/test_api_upload_utils.py` — real streaming/size-limit/cleanup behavior, no mocks,
  plus Phase 14's path-traversal-filename tests (§12 below).
- `tests/unit/test_api_rate_limit_middleware.py` — a real sliding-window limiter under a real ASGI
  app.
- `tests/unit/test_api_security_headers.py`, `test_api_exception_handlers.py` — Phase 14: every
  response carries the headers in §14, including one raised by a genuinely unhandled exception.
- `tests/integration/test_authorization_scopes.py` — Phase 14: the 401-vs-403 scope distinction,
  legacy single-key backward compatibility, real SQLite-backed `/monitoring/ground-truth` scope
  checks.
- `tests/integration/test_predictions_api.py`, `test_explanations_api.py`, `test_models_api.py`,
  `test_meta_api.py` — a real `TestClient` against a real, tiny checkpoint (no mocks): success
  paths, every error-mapping row in §5, auth enforcement, CORS/OpenAPI/security-scheme presence.

## 14. Security headers (Phase 14)

`api.middleware.security_headers.SecurityHeadersMiddleware` is the **outermost** middleware (added
even before `CORSMiddleware` — see `api.main`'s module docstring) and sets, on every response:
`X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`,
`Strict-Transport-Security` (a no-op over plain HTTP local dev, effective once a real deployment
terminates TLS in front of this process — see `docs/deployment.md`), `Permissions-Policy` (denying
geolocation/microphone/camera/payment/usb), and a route-appropriate `Content-Security-Policy` — a
strict `default-src 'none'` for every JSON API route, a looser policy scoped to `/docs`/`/redoc`
that allows only the `cdn.jsdelivr.net` assets those pages themselves load.

**One real architectural gap, by design, not oversight**: a genuinely unhandled exception is caught
by Starlette's `ServerErrorMiddleware`, which sits *outside every user middleware including this
one* (see `inspect.getsource(starlette.applications.Starlette.build_middleware_stack)`) — a bare
ASGI app with no registered exception handler would return a 500 with none of these headers. This
project closes that gap not by fighting the framework's own last-resort layer, but by registering a
real `Exception` handler (`api.exception_handlers._handle_unexpected_error`, dispatched by the
*inner* `ExceptionMiddleware` instead, which does sit inside user middleware) that calls
`apply_security_headers()` on its own response directly — so in practice every response this
application actually produces carries these headers, verified by
`test_api_exception_handlers.py::test_an_unexpected_exception_still_gets_security_headers`.

## 15. ML-safety response fields (Phase 14)

Both `PredictionResponse` and `ExplanationResponse` carry a `disclaimer` string field
(`core.disclaimers.PREDICTION_DISCLAIMER` / `EXPLANATION_DISCLAIMER`) inline in every successful
response body — not left only in this documentation, so a caller integrating against the API sees
it on every call, not only the ones who read the docs first. The prediction disclaimer states this
is not a clinical diagnosis; the explanation disclaimer additionally states that LIME/SHAP output
identifies association, not causal, evidence. See
[`docs/responsible_ai.md`](responsible_ai.md) for the full rationale and every other ML-safety
control (uncertainty exposure, model versioning, documented limitations/bias/dataset risks).

`ExplanationResponse` also carries `narrative_summary: str | None` — an optional, off-by-default
plain-language rendering of the numeric LIME/SHAP output via the OpenAI API
(`explainability.llm_summary`), `None` unless `OPENAI_API_KEY` is configured. Only aggregate
region-importance numbers this codebase already computed are ever sent to that API — never image
data, a file path, or a patient identifier. See [`docs/explainability.md`](explainability.md) §7.5.
