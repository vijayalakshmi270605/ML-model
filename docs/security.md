# Security, privacy & responsible-AI review (Phase 14)

Status: **Phase 14 — the final phase of this build.** This document is the security review itself:
a requirement-by-requirement account of what was found, what was fixed, what was already safe, and
what remains a documented residual risk. Nothing here is a clinical claim — see
[`docs/TECHNICAL_SPECIFICATION.md`](TECHNICAL_SPECIFICATION.md) §0. ML-safety-specific controls
(disclaimers, uncertainty exposure, dataset/bias limitations) are covered in full in
[`docs/responsible_ai.md`](responsible_ai.md); this document indexes them but does not repeat them.

> **Post-review addendum**: an optional LLM narrative-summary feature
> (`explainability.llm_summary`, [`docs/explainability.md`](explainability.md) §7.5) was added
> after this review's initial sign-off, introducing this system's first outbound call to a
> third-party API. It was designed against, and stays within, every principle below: off by
> default, a strictly minimal/aggregate payload (never raw pixel data, a file path, or a patient
> identifier — enforced by a real test, not just documented), best-effort (never fails a request),
> and never exercised with a real network call in the automated test suite. The secret it depends
> on (`OPENAI_API_KEY`) follows the same blank-string-means-`None` convention and the same
> git-ignored-`.env`-only storage as every other credential in this project (§3 below). No
> section's original findings needed revision as a result.

## 0. Method

Every finding below was verified against running code — a real exploit-shaped input constructed and
rejected, a real header inspected on a real response, a real tool run against this actual dependency
tree — not asserted from reading source alone. Where a check concluded "already safe," that
conclusion itself is backed by an adversarial test added under `tests/`, not by assumption. The full
test suite (536 tests), `ruff`, `black --check`, `mypy --strict`-equivalent config, `bandit`, and
`pip-audit` were run clean as part of this review — see §8.

## 1. Threat model

This system accepts an uploaded brain MRI volume from an authenticated (optionally) HTTP client,
runs it through a CNN classifier and optional LIME/SHAP explainability pipeline, and returns a
prediction. The realistic threat surface is therefore:

- **A malicious or malformed upload** — the only untrusted *content* this system ever parses.
- **A malicious or careless API caller** — abusive request volume, an unauthorized attempt to read
  another deployment's data, a client depending on more than this API promises (e.g. treating a
  probability as a diagnosis).
- **Misconfiguration at deploy time** — a leaked secret, a database exposed to the network, a
  production instance still running with dev defaults.

Out of scope for this system as built: host/OS-level compromise, physical access, and anything
requiring an already-compromised CI/CD credential (`docs/deployment.md` §3/§4/§6 cover that
boundary's own hygiene). There is no browser-rendered UI of this system's own to attack (XSS/CSRF
against a page this project serves) — the only HTML this process ever returns is FastAPI's own
`/docs`/`/redoc`, addressed in §5.

## 2. Requirement-by-requirement findings

| Requirement | Finding | Fix / control |
|---|---|---|
| Unauthorized file uploads | Upload endpoints were open when `API_KEY_REQUIRED=false` (dev default) — expected, documented behavior, not a bug. In `true` mode, real coverage gap: every key held every permission. | Phase 14: scoped multi-key authorization (§4) — a key can be limited to `predict` without `label`. |
| Malicious files | A crafted `.nii.gz` can declare an enormous voxel count in its cheaply-readable ~348-byte header while the file itself is tiny — a real decompression-bomb vector that would OOM the process on the later full-volume load, regardless of upload size limits. | `PreprocessingConfig.max_voxel_count` (default 100M voxels) — `validate_image_for_pipeline` rejects it before `LoadImaged` ever materializes the array. Verified with a real ~74-byte crafted header claiming a 5000³ volume (`test_validate_image_for_pipeline_rejects_a_decompression_bomb`). |
| Oversized uploads | Already real (Phase 10) — `api.upload_utils.save_upload_within_limit` streams in 1 MiB chunks and aborts with `413` the instant the running total exceeds `MAX_UPLOAD_SIZE_MB`, never buffering the full body in memory. Re-verified, no change needed. | No change — `test_predictions_rejects_an_oversized_upload_with_413`. |
| Path traversal | Reviewed adversarially: does an uploaded filename like `../../../etc/passwd` or a 5000-character pathological name ever influence where a temp file is written? | Confirmed already safe *by construction*, not by luck: `save_upload_within_limit` always names the temp file a fixed `upload<ext>` inside a freshly-`mkdtemp`'d directory — the client-supplied filename is read only for its suffix, never concatenated into a path. New adversarial tests added to convert this from an assumption into evidence (`test_upload_suffix_never_contains_a_path_separator`, `test_save_upload_with_a_path_traversal_filename_stays_inside_its_temp_dir`, 7+3 parametrized cases). |
| Sensitive information leakage | Already real (Phase 9/10): filenames are never logged/echoed (generic `upload<ext>` temp name; `redact_image_reference` hashes the path for logging); exception handlers log exception **type**, never `str(exc)` (a domain exception's message can legitimately contain a filesystem path). Re-verified: `test_predictions_response_never_contains_the_uploaded_filename`, and the new `test_an_unexpected_exception_returns_a_clean_500_not_a_raw_traceback` proves an *unmapped* exception's message/type never leaks either. | No change to the existing discipline; new test coverage for the unmapped-exception case specifically. |
| Insecure logging | Reviewed `core.logging`, `api.audit`, `db.audit_service`, `api.deps`, `api.middleware.rate_limit` for anything that could log a credential, a raw pixel array, or PHI. Every audit-write failure logs `error_type` only; API keys are never logged in the clear (`db.audit_service` hashes them via `db.privacy.hash_value` before persistence); no code path logs `settings.database_url`, `settings.secret_key`, or `settings.api_key`/`api_keys`. | No leak found — see grep evidence in this review's own working notes; §8 confirms no regression. |
| Exposed model endpoints | `GET /api/v1/models` is intentionally open (read-only metadata, not compute) — reviewed for anything beyond version/hash exposure (no checkpoint path, no training data reference). `POST /predictions`/`/explanations` (the actual compute/cost surface) already required a key when configured. | Scope tightened (§4): `predict` scope now gates both, distinct from `label`. |
| Exposed database credentials | `Settings.postgres_password`/`secret_key` are `SecretStr` (never rendered in `repr()`/logs by accident); `.env` is git-ignored; `.env.example` ships every secret field blank; `.dockerignore` excludes `.env` from every build context. Confirmed every `.get_secret_value()` call site in the codebase (4 total: building the DB connection string, the insecure-default-`SECRET_KEY` check, and the two API-key comparisons in `api.deps`) exists only to compare or assemble a value at the point of use — never to log, return, or persist it in the clear. | No change — already correct; see `docs/deployment.md` §6 for the full secrets-management layering. |
| API abuse | `RateLimitMiddleware` was already real (Phase 10/13) but keyed off a **hardcoded** `"X-API-Key"` header name — a deployment that reconfigured `API_KEY_HEADER_NAME` would have every caller's rate-limit bucket silently collapse onto the unauthenticated (source-IP) bucket, since the header the middleware read never matched a real request. | Fixed: `RateLimitMiddleware` now takes `api_key_header_name` from `Settings` (`test_a_reconfigured_api_key_header_name_is_honored`). |
| Dependency vulnerabilities | Ran `pip-audit --skip-editable` and `bandit -c pyproject.toml -r src` for real, against this project's actual installed dependency tree, as part of this review (not only in CI). | **0 known vulnerabilities** (`pip-audit`; `torch` skipped only because pip-audit can't resolve a local `+cpu` build tag against PyPI — CI's own vanilla install is audited normally). **0 findings** (`bandit`, 11,365 lines scanned). See `docs/deployment.md` §3 for the CI-enforced version of both. |

## 3. Controls implemented this phase

- **Input validation / file type validation**: already real from Phase 9/10 (`datasets.validation`,
  content-sniffed NIfTI/DICOM parsing, not extension-trusted) — extended this phase with the
  voxel-count *ceiling* (`max_voxel_count`), closing the decompression-bomb gap next to the
  pre-existing floor (`min_voxel_count`).
- **Upload size limits**: already real (`MAX_UPLOAD_SIZE_MB`, streamed enforcement) — unchanged,
  re-verified.
- **Secure temporary storage**: already real (`tempfile.mkdtemp`, generic filename, `finally:
  shutil.rmtree`) — re-verified against path-traversal-shaped filenames specifically this phase.
- **Secret management**: already real (Phase 6/12) — this phase closed one inconsistency: a blank
  `API_KEY=`/`API_KEYS=` env var (the `.env.example` default) now reliably normalizes to `None`
  rather than an empty-but-truthy value, matching the existing convention for every other optional
  secret/path field (`Settings._blank_api_key`/`_blank_api_keys`).
- **Authentication architecture**: already real (single shared `API_KEY`) — unchanged, still fully
  supported for backward compatibility.
- **Authorization architecture** (new, Phase 14): `ApiScope` (`predict` | `label`) +
  `Settings.api_keys: dict[str, list[ApiScope]] | None` + `api.deps.require_scope(scope)`. A
  deployment can now issue an ingestion pipeline a `predict`-only key and reserve `label`
  (submitting ground truth — see [`docs/monitoring.md`](monitoring.md) §5) for a clinician's own
  key. A recognized key lacking the required scope gets `403` (authorized-but-forbidden), distinct
  from `401` (unauthenticated) — a real, testable signal a caller's own error handling can act on.
  Fully backward compatible: the legacy single `API_KEY` still works unchanged as a superuser
  holding every scope when `API_KEYS` is unset. See [`docs/api.md`](api.md) §9 and
  `tests/integration/test_authorization_scopes.py` (6 tests, including the 401-vs-403 distinction).
- **Audit logging**: already real (Phase 13, `api.audit`/`db.audit_service`) — every prediction,
  explanation, model-version-seen event, and system event (auth/rate-limit rejections, unexpected
  errors) is persisted, best-effort, without ever failing the request it's logging. Unchanged this
  phase; re-verified alongside the new catch-all exception handler (§ below).
- **Secure headers** (new, Phase 14): `SecurityHeadersMiddleware` — see §5.
- **Rate limiting**: already real; the header-name bug (§2) fixed.

## 4. Authorization architecture in detail

See [`docs/api.md`](api.md) §9 for the full mechanism and examples. Summary of the design
decision: authorization is **scope-based, not role-based** — there are exactly two scopes
(`predict`, `label`) because those are the two distinct capabilities this API actually exposes
(run inference / submit ground truth for monitoring). A full RBAC system (named roles, a
permissions matrix, role assignment persisted in a database) was deliberately not built — it would
solve a problem this deployment shape doesn't have yet (no per-user identity exists at all; every
"identity" is a shared API key). The `ApiScope` enum and `require_scope` dependency factory are the
seam a real RBAC system would extend, not replace.

## 5. Secure headers in detail

`api.middleware.security_headers.SecurityHeadersMiddleware` is the outermost middleware
(`api.main`), setting on every response: `X-Content-Type-Options: nosniff`, `X-Frame-Options:
DENY`, `Referrer-Policy: no-referrer`, `Strict-Transport-Security`, a `Permissions-Policy` denying
geolocation/microphone/camera/payment/usb, and a `Content-Security-Policy` — strict
(`default-src 'none'`) for every JSON API route, a looser one scoped only to `/docs`/`/redoc`
(the one place this process serves real, third-party-script-loading HTML).

**A genuinely non-obvious architectural gap was found and fixed, not assumed away**: Starlette
dispatches a handler registered for bare `Exception` via `ServerErrorMiddleware`, which sits
*outside every user-registered middleware, including this one* — confirmed by reading
`Starlette.build_middleware_stack`'s actual source, then proven with two comparative test apps (one
with an `Exception` handler, one with a specific-subclass handler) before writing the fix. A naive
implementation that only added headers inside `SecurityHeadersMiddleware.dispatch` would silently
omit them on any truly unhandled exception — exactly the response most likely to reach an
attacker probing for weaknesses. The fix: `api.exception_handlers._handle_unexpected_error` (the
catch-all handler) calls `apply_security_headers()` on its own response directly, so the header
guarantee holds for every response this application actually produces. Verified by
`test_api_exception_handlers.py::test_an_unexpected_exception_still_gets_security_headers`, with a
second test (`test_api_security_headers.py`) documenting the underlying framework gap itself so a
future change to exception-handler registration can't silently regress this.

## 6. Verification summary

Run for real as part of this review (not simulated, not skipped):

| Check | Result |
|---|---|
| `pytest` (full suite) | **536 passed**, 0 failed |
| `ruff check src tests` | All checks passed |
| `black --check src tests` | 203 files unchanged |
| `mypy --no-incremental src` | Success, 0 issues, 123 source files |
| `bandit -c pyproject.toml -r src` | 0 issues, 11,365 lines scanned |
| `pip-audit --skip-editable` | 0 known vulnerabilities |

New tests added this phase: `test_datasets_validation.py` (decompression bomb, 2 tests),
`test_api_security_headers.py` (4), `test_api_exception_handlers.py` (3),
`test_authorization_scopes.py` (6), `test_api_rate_limit_middleware.py` (+1),
`test_api_upload_utils.py` (+8 path-traversal cases), `test_config.py` (+9 for
`api_keys`/`api_key` blank-value handling).

## 7. Residual risks (explicitly not closed this phase)

Documented honestly rather than implied fixed:

- **No TLS termination in this process** — expected; TLS is terminated at the ingress/load
  balancer in front of it (`docs/deployment.md`). `Strict-Transport-Security` is sent
  unconditionally so it takes effect the instant a real deployment adds TLS in front.
- **No encryption at rest** for PostgreSQL/object storage volumes — a deployment/infrastructure
  concern, not application code; not implemented.
- **No DICOM header sanitization** — moot for what's actually built: every phase consumes NIfTI,
  never raw DICOM with its identifying header tags. If a future phase adds DICOM ingestion, this
  becomes a real, not hypothetical, requirement.
- **No per-user identity / OAuth2/JWT / full RBAC** — by design (§4); the scoped multi-key
  architecture is proportionate to this system's actual deployment shape (a handful of
  service-level keys), not a limitation discovered late.
- **In-memory rate limiting is per-process** — a multi-instance deployment enforces the limit
  independently per instance, not globally; the seam to a shared (Redis-backed) limiter is already
  identified (`docs/api.md` §10) and not built, since this system has never actually run
  multi-instance in this sandbox.
- **`torch` could not be pip-audited directly** — pip-audit can't resolve the locally-installed
  `+cpu` build tag against PyPI's index; this is a resolution artifact of this sandbox's install,
  not a suppressed finding — CI's own from-scratch install audits the real published wheel.

## 8. ML safety

See [`docs/responsible_ai.md`](responsible_ai.md) for the complete account: the
prediction-vs-diagnosis distinction (now enforced inline via the `disclaimer` response field, not
only in documentation), model version exposure, careful uncertainty/probability exposure, the
explanation-without-causality disclaimer, and documented dataset/bias/generalization limitations.
