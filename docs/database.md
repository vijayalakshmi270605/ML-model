# Database & audit architecture

Status: **Phase 11 — PostgreSQL-backed audit trail implemented.** Nothing here is a clinical claim
or a performance claim — see [`docs/TECHNICAL_SPECIFICATION.md`](TECHNICAL_SPECIFICATION.md) §0.

The audit trail records **what happened**, not **what was uploaded**: every table exists to answer
"which model version served this prediction, when, how fast, with what result" — never to hold a
patient's MRI, name, or any other identifier. See §6 for exactly what is enforced, not just
intended.

## 1. Tables

Five tables, `ad_mri.db.models`, one SQLAlchemy `Base` (`db.base.Base`):

| Table | One row per | Key columns |
|---|---|---|
| `model_versions` | distinct (model config hash, preprocessing config hash) pair ever served | `architecture`, `model_version`, `model_config_hash`, `preprocessing_config_hash`, `checkpoint_uri`, `stage`, `device` |
| `prediction_requests` | inbound `POST /predictions` (or `/explanations`) call | `external_request_id`, `model_version_id` (FK), `image_ref`, `content_type`, `api_key_hash`, `received_at` |
| `prediction_results` | completed prediction, 1:1 with its request | `request_id` (FK, unique), `predicted_class`, `confidence`, `probabilities`, `processing_time_ms`, `explanation_available` |
| `explanation_records` | generated LIME/SHAP explanation | `external_explanation_id`, `request_id` (FK, nullable), `model_version_id` (FK), `methods`, `target_class`, `artifact_uri`, `params_hash` |
| `system_events` | any other audit-worthy occurrence | `event_type`, `severity`, `request_id` (free text, not a FK), `detail` (JSON) |

`prediction_requests`/`prediction_results` are split deliberately: a request that fails before a
result exists still gets an audit trail (its request row, plus a `system_events` row — see §5),
rather than either not being recorded at all or forcing nullable result columns onto every row.

## 2. Why `id` is not the caller-facing request/explanation id

`X-Request-ID` is caller-suppliable (`api.middleware.request_id.RequestIDMiddleware` honors an
inbound header verbatim) and therefore not guaranteed to be UUID-shaped — the same is true of
`ExplanationBundle.input_id`, which falls back to `image_path.stem` when no request id was ever
set. Requirement 5 ("UUID-based request IDs") is met by making each table's own primary key a
server-generated `uuid.uuid4()` (`db.types.GUID`, see §3) — never by coercing arbitrary caller text
into a UUID column, which would either raise on a real request or silently need a permissive type
that a real PostgreSQL foreign key could never safely reference. The caller-facing value is kept
too, in `external_request_id`/`external_explanation_id` (indexed, free text) — for correlating a
row back to a client response or a log line — but it is not this schema's identity.

## 3. `GUID`: one model layer, two databases

`db.types.GUID` is a native PostgreSQL `UUID` column when bound to a `postgresql` engine, and a
`CHAR(32)` hex-string column everywhere else. This is not a hypothetical portability nicety: it is
exactly what lets `tests/unit/test_db_*.py` and `tests/integration/test_audit_integration.py`
exercise the *real* SQLAlchemy models, repositories, and service layer against a real SQLite file
instead of requiring a locally running PostgreSQL server — the schema PostgreSQL actually gets is
still the one in `db/migrations/versions/0001_create_audit_tables.py`, hand-verified in this phase
by running it for real (`alembic upgrade head` / `downgrade base`) against a SQLite database and
inspecting the resulting DDL.

## 4. Object storage: no raw imaging data in PostgreSQL

Requirement: *"Do NOT store raw MRI images in PostgreSQL. Use object storage architecture for
large medical imaging files if persistence is required."* Two columns exist specifically to make
that the only option: `model_versions.checkpoint_uri` and `explanation_records.artifact_uri`,
each `VARCHAR(512)` — wide enough for a URI, nowhere near wide enough for a checkpoint or a
saliency array. `storage.object_storage` provides:

- `ObjectStorageClient` — the `Protocol` every caller depends on.
- `LocalFilesystemObjectStorage` — real today: `file://` URIs over local/mounted disk, which is
  exactly where `InferenceEngine`/`ExplainabilityService` already write checkpoints and explanation
  manifests (Phases 9-10) — this class only frames those existing paths as URIs, it does not
  relocate anything.
- `S3ObjectStorageClient` — a documented, unimplemented seam for a multi-node deployment, in
  exactly the same style as `model_registry.client.MLflowModelRegistryClient`: the interface a real
  S3/MinIO client will satisfy is fixed now, so adopting one later touches no caller.

Raw MRI uploads themselves are never persisted at all, in any form — `api.upload_utils` writes each
one to a per-request temp directory that `finally: shutil.rmtree(...)` deletes before the response
is returned (Phase 10); there is currently no requirement to retain them, so no object-storage
write of upload bytes happens today. If that requirement appears, it goes through this same
`ObjectStorageClient` seam, never into a database column.

## 5. Wiring into the API — best-effort, never load-bearing

`api.audit` is the only place the API layer touches the database. Every function in it:

1. Opens its own short-lived `Session` (`db.session.get_sessionmaker()`), independent of any
   per-request session.
2. Calls into `db.audit_service.AuditService` (see §7).
3. Catches `Exception`, logs `error_type` only, and returns — **never raises**.

This mirrors `monitoring.health_checks` (database reachability is tracked, never load-bearing) and
is the same independence `GET /ready` already models between the database and the inference engine
(Phase 10): an unreachable audit database degrades observability, it never fails a client's
`/predictions` or `/explanations` call. `tests/integration/test_audit_integration.py` proves both
halves for real — one test with a real, reachable SQLite database asserting the rows exist, one
against the default (unreachable-in-this-environment) PostgreSQL asserting the response is
unaffected.

Call sites:

| Where | What | When |
|---|---|---|
| `api.deps._build_inference_engine` | `record_model_version_seen` | once, right after a checkpoint loads successfully |
| `api.routers.predictions.create_prediction` | `record_prediction` | after a successful `201` |
| `api.routers.explanations.create_explanation` | `record_explanation` | after a successful `201` |
| `api.exception_handlers` (every mapped handler) | `record_event` | on every mapped domain-exception response |

`Settings.audit_enabled` (`AUDIT_ENABLED`, default `true`) turns all of it off with a single flag,
for a deployment that wants zero database dependency.

## 6. Privacy enforcement — not just convention

Everything below is a real, tested invariant (`tests/unit/test_db_privacy.py`), not a comment:

- **No patient names / raw filenames.** `image_ref` is validated
  (`db.privacy.ensure_redacted_image_reference`, called from
  `PredictionRequestRepository.create`) to already be a `redact_image_reference()` hash — 16
  lowercase hex characters. A raw path or filename raises `ValueError` before it can reach the
  database, not just before it reaches a log line.
- **No secrets.** `api_key_hash` stores `sha256(presented_key)` (`db.privacy.hash_value`), never
  the key. `checkpoint_uri`/`database_url`/`secret_key` never appear in any row.
- **No raw MRI data.** See §4 — there is no column wide enough, by design.
- **No unnecessary identifiers.** `system_events.detail` and every log call this module makes
  carry only already-sanitized, small, structured values (status codes, error-type names, counts) —
  never `str(exc)`, matching the convention `api.exception_handlers`'s module docstring already
  established in Phase 10.

## 7. SQLAlchemy, repositories, and transaction management

- **Repository layer** (`db.repositories`) — one class per table, typed `create`/`get`/`list_*`
  methods. No business logic, no `commit()` — every write `flush()`es (so a caller can read back a
  generated id) but leaves the transaction boundary to its caller.
- **Service layer** (`db.audit_service.AuditService`) — one instance per `Session`; each public
  method is one whole transaction, via a private `_transaction()` context manager that commits on
  success and rolls back **the entire method**, not just the final `commit()` call, on any failure
  (including one raised by an early repository call, e.g. the `image_ref` guard above) — the
  original version of this service only wrapped the final `commit()` in try/except, which could
  leave a flushed-but-uncommitted row dangling on an early failure;
  `tests/unit/test_audit_service.py::TestRecordPrediction::test_a_failed_write_rolls_back_the_entire_transaction`
  reads the database back with a second, independent session to prove the fix.
- **Loud, not best-effort.** `AuditService` raises on failure — the best-effort boundary is
  `api.audit`, one layer up (§5), the same "loud service / best-effort caller" split
  `inference.predictor.generate_explanation` / `_try_generate_explanation` already uses. Keeping
  this layer loud is what makes `assert no exception raised` in its unit tests actually mean "the
  row is durably there."
- **Connection pooling** (`db.session.build_engine`) — `pool_size`/`max_overflow`/`pool_timeout`/
  `pool_recycle` (`DB_POOL_SIZE`, `DB_MAX_OVERFLOW`, `DB_POOL_TIMEOUT`, `DB_POOL_RECYCLE`), applied
  only for a non-SQLite URL — SQLite's own pool class rejects those kwargs outright, which is
  exactly the failure `tests/unit/test_db_session.py` would catch if the branch were removed.
  `pool_pre_ping=True` always, so a connection PostgreSQL silently dropped is detected and replaced
  rather than surfacing as an opaque error on whichever request happened to draw it.
- **Migrations** — Alembic, `alembic.ini` + `db/migrations/env.py` (wired to `Settings`, never a
  hardcoded connection string). `db/migrations/versions/0001_create_audit_tables.py` is hand-written
  (this environment has no reachable PostgreSQL for `alembic revision --autogenerate` to inspect),
  but was run for real — `alembic upgrade head` then `alembic downgrade base` against a real SQLite
  database, with the resulting schema inspected column-by-column against `db.models` (§3).

## 8. Running migrations

```bash
# against the configured DATABASE_URL / POSTGRES_* settings
alembic upgrade head

# roll back everything (used in this phase's own verification, see §7)
alembic downgrade base
```

## 9. What is explicitly out of scope for this phase

Documented seams, not implemented:

- **A live PostgreSQL server in this development environment.** Confirmed unreachable
  (`psycopg2.OperationalError: connection refused`) since Phase 10's `/ready` work; every table,
  migration, repository, and service method in this phase is verified against a real SQLite
  database instead (§3), plus the migration file itself run for real against SQLite (§7) — the
  PostgreSQL-specific paths (`GUID`'s native-`UUID` branch, `build_engine`'s pooling branch) are
  unit-tested directly against a `postgresql` dialect object without a live connection
  (`tests/unit/test_db_types.py`, `tests/unit/test_db_session.py`).
- **S3/MinIO object storage** — `storage.object_storage.S3ObjectStorageClient` is a fixed
  interface, `NotImplementedError` bodies.
- **A `prediction_requests`/`prediction_results` read API.** This phase adds the write path (audit
  persistence) and the repository/service read methods used by its own tests; no `GET` endpoint
  exposes them yet.
- **Retention/archival policy** for `system_events`/`prediction_requests` growth over time — not
  addressed by this phase.
