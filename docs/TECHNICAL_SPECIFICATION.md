# Technical Specification — Alzheimer's Disease MRI Classification System

Status: **Phase 0 — Architecture & Design (no implementation code yet)**
Owner: vijayalakshmivadivel103@gmail.com
Last updated: 2026-09-08

---

## 0. Scope, Intent, and Non-Goals

This document specifies a **production-grade ML system** for classifying Alzheimer's Disease stage from
structural brain MRI, built around a CNN/deep-learning classifier, with explainability (LIME + SHAP),
rigorous evaluation, and full MLOps tooling (MLflow, DVC, Docker, CI/CD).

**Explicitly out of scope / non-goals:**
- This is a **research/decision-support and portfolio-grade engineering system**, not a certified medical
  device. It makes no clinical diagnostic claims.
- No performance numbers (accuracy, sensitivity, specificity, AUC, etc.) are asserted anywhere in this
  spec. All such numbers will only ever come from actual measured runs, logged in MLflow, and reported
  in a model card — never assumed or fabricated.
- No claim is made about which published paper this system "beats." The "comparison against existing
  research" module produces a **methodology-normalized comparison table** populated from the literature's
  *reported* numbers (with citations) next to this system's *measured* numbers — the system does not
  guarantee superiority.
- Real patient data (e.g., ADNI, OASIS) requires **Data Use Agreements** and ethics/IRB compliance that
  are outside engineering scope; the architecture assumes the data owner has already secured usage rights.
  The system is designed to be compliant-by-construction (de-identified inputs, audit trails, PHI-aware
  logging) but does not itself grant regulatory clearance.

---

## 1. System Architecture Overview

### 1.1 Architectural style

A **modular, layered, service-oriented architecture** with clear separation between:

1. **Data layer** — acquisition, validation, preprocessing, versioning (DVC).
2. **Modeling layer** — training, evaluation, explainability (PyTorch + MONAI, MLflow-tracked).
3. **Serving layer** — inference API (FastAPI), explainability API, async workers.
4. **Persistence layer** — PostgreSQL (metadata, predictions, audit trail, users), object storage
   (model/data artifacts).
5. **MLOps/platform layer** — MLflow tracking + registry, DVC pipelines, CI/CD, containerization,
   monitoring/observability.

The system is designed **offline-training / online-inference**: training is a batch, reproducible,
config-driven pipeline run on demand (locally, CI, or a training host/GPU box); inference is a
low-latency synchronous API, with explainability treated as a **heavier, optionally async** operation.

### 1.2 High-level component diagram (textual)

```
                              ┌─────────────────────────────┐
                              │        Data Sources         │
                              │ (ADNI / OASIS / Kaggle /     │
                              │  institutional DICOM/NIfTI)  │
                              └──────────────┬───────────────┘
                                             │ ingest
                                             ▼
                     ┌───────────────────────────────────────────┐
                     │              Data Pipeline (DVC)           │
                     │  raw → validate → preprocess → QC → split  │
                     └──────────────┬──────────────────────────────┘
                                    │ versioned datasets (DVC + remote storage)
                                    ▼
                     ┌───────────────────────────────────────────┐
                     │            Training Pipeline                │
                     │  MONAI transforms → PyTorch CNN → eval      │
                     │  → MLflow tracking → Model Registry          │
                     └──────────────┬──────────────────────────────┘
                                    │ registered model artifact (versioned)
                                    ▼
        ┌───────────────────────────────────────────────────────────────┐
        │                        Serving Layer                           │
        │  ┌───────────────┐   ┌────────────────┐   ┌──────────────────┐ │
        │  │  FastAPI       │   │  Explainability │   │  Async Worker     │ │
        │  │  Inference API │◄─►│  Service (LIME/ │◄─►│  (Celery/RQ +     │ │
        │  │                │   │  SHAP)          │   │  Redis)           │ │
        │  └───────┬────────┘   └────────┬────────┘   └─────────┬─────────┘ │
        └──────────┼─────────────────────┼──────────────────────┼───────────┘
                   │                     │                      │
                   ▼                     ▼                      ▼
        ┌────────────────────────────────────────────────────────────────┐
        │   PostgreSQL (predictions, audit log, users, model metadata)    │
        │   Object storage (MLflow artifacts, DVC remote, explanation      │
        │   heatmaps)                                                      │
        └────────────────────────────────────────────────────────────────┘

        Cross-cutting: MLflow (tracking/registry) · Prometheus/Grafana (monitoring)
        · structured logging · GitHub Actions CI/CD · Docker/Compose/K8s
```

### 1.3 Design principles

- **Config-driven, not hardcoded**: dataset paths, class taxonomy, model architecture, hyperparameters,
  explainability settings all come from versioned config files, never literals in code.
- **Reproducibility as a first-class requirement**: every trained model is traceable to an exact
  git commit, DVC data hash, config hash, and random seed.
- **Separation of concerns**: preprocessing logic is shared (same code path) between training and
  inference to eliminate train/serve skew — implemented as an installable internal package, not
  duplicated code.
- **Fail closed on data/model validation**: malformed input, schema mismatches, or missing model
  metadata reject the request rather than silently degrading.
- **Explainability is a peer citizen, not an afterthought**: it has its own module, its own tests,
  its own API contract, and its own artifacts store.

---

## 2. Major Modules

| Module | Responsibility |
|---|---|
| `data_pipeline` | Ingestion, DICOM→NIfTI conversion, validation, preprocessing (skull-stripping, registration, normalization), quality control, subject-level train/val/test split generation. |
| `datasets` | PyTorch/MONAI `Dataset`/`transforms` wrappers, caching, augmentation pipelines — training-time and inference-time variants sharing a common base. |
| `modeling` | CNN architecture definitions (2D slice-based and 3D volumetric options), model factory, loss functions, metrics. |
| `training` | Training loop orchestration, experiment configuration, checkpointing, early stopping, hyperparameter search (Optuna), MLflow run management. |
| `evaluation` | Confusion matrix, per-class precision/recall/F1, ROC/AUC, calibration, error analysis, literature-comparison table generation. |
| `explainability` | LIME (image segmentation-based local explanations) and SHAP (gradient/deep explainer) integrations, heatmap/overlay generation, explanation artifact storage. |
| `model_registry_client` | Thin wrapper around MLflow Model Registry: register, promote (Staging→Production), fetch by stage/version. |
| `inference` | Model loading/caching, preprocessing invocation, prediction, confidence scoring, response assembly. |
| `api` | FastAPI application: routers, request/response schemas (Pydantic), dependency injection, auth, rate limiting. |
| `worker` | Celery/RQ async task runner for explainability and batch jobs. |
| `db` | SQLAlchemy models + Alembic migrations for PostgreSQL (predictions, audit log, datasets metadata, users/API keys). |
| `common` / `core` | Config loading (pydantic-settings), logging setup, exceptions, constants, seeding utilities. |
| `monitoring` | Metrics exporters (Prometheus client), drift-detection jobs, health checks. |
| `cli` | Typer/Click CLI entry points for `train`, `evaluate`, `explain`, `preprocess`, `serve`, `register-model`. |

---

## 3. Repository Structure

> **As implemented (Phase 12)**: `.github/workflows/` is real, but as two workflows
> (`ci.yml` — lint/type-check/security/unit/integration/build, and `cd.yml` —
> dev/staging/production deployment), not the five originally sketched below; there is no
> `data-pipeline-ci.yml`/`train-smoke-test.yml` yet. `docker/` is real with a fourth `mlflow.Dockerfile`
> and both `docker-compose.yml` (dev) and `docker-compose.prod.yml` (production) — no MinIO
> service (object storage remains the `storage.object_storage` seam, see
> [`docs/database.md`](database.md) §4). `deploy/environments/` (non-secret per-environment config
> templates) is real; `deploy/k8s/`/`deploy/terraform/` remain placeholders. `configs/env/*.yaml`
> was not built — environment configuration is env-var-driven throughout (`core.config.Settings`),
> matching every other phase. See [`docs/deployment.md`](deployment.md) for the real structure.
> Phase 13 adds `deploy/prometheus/` (a real, `promtool`-validated scrape config + alert rules)
> and `deploy/grafana/` (a real dashboard + datasource/dashboard provisioning), not present in
> the original sketch — see [`docs/monitoring.md`](monitoring.md). The rest of this tree (much of
> it already covered by earlier phases' own "As implemented" callouts) is left as originally
> sketched below, as a historical design record.

```
project-1/
├── .github/
│   └── workflows/
│       ├── ci.yml                  # lint, type-check, unit+integration tests
│       ├── data-pipeline-ci.yml    # dvc repro validation on PR (light/smoke data)
│       ├── train-smoke-test.yml    # tiny-subset training run sanity check
│       ├── docker-build-push.yml   # build & push images on tag/release
│       └── security-scan.yml       # bandit, pip-audit/safety, trivy image scan
│
├── configs/
│   ├── base.yaml                   # shared defaults
│   ├── data/
│   │   ├── dataset_oasis.yaml
│   │   └── dataset_kaggle_adni.yaml
│   ├── model/
│   │   ├── resnet3d.yaml
│   │   ├── densenet2d.yaml
│   │   └── efficientnet2d.yaml
│   ├── training/
│   │   ├── default.yaml
│   │   └── hpo_optuna.yaml
│   ├── explainability/
│   │   ├── lime.yaml
│   │   └── shap.yaml
│   └── env/
│       ├── dev.yaml
│       ├── staging.yaml
│       └── prod.yaml
│
├── data/                           # DVC-tracked, not committed to git content
│   ├── raw/                        # .dvc pointer files only
│   ├── interim/
│   ├── processed/
│   └── splits/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── src/
│   └── ad_mri/                     # installable internal package (pip install -e .)
│       ├── __init__.py
│       ├── core/                   # config, logging, seeding, exceptions
│       ├── data_pipeline/          # ingestion, conversion, preprocessing, QC
│       ├── datasets/               # MONAI/PyTorch Dataset + transforms
│       ├── modeling/               # architectures, model factory, losses
│       ├── training/               # trainer, callbacks, HPO
│       ├── evaluation/             # metrics, confusion matrix, lit-comparison
│       ├── explainability/         # lime_explainer.py, shap_explainer.py, overlays
│       ├── inference/              # predictor, model loader/cache
│       ├── model_registry/         # MLflow registry client
│       ├── api/                    # FastAPI app
│       │   ├── main.py
│       │   ├── routers/
│       │   │   ├── predict.py
│       │   │   ├── explain.py
│       │   │   ├── models.py
│       │   │   └── health.py
│       │   ├── schemas/
│       │   ├── deps.py             # auth, db session, model cache DI
│       │   └── middleware/
│       ├── worker/                 # Celery/RQ tasks
│       ├── db/                     # SQLAlchemy models, session, migrations (alembic/)
│       ├── monitoring/             # prometheus exporters, drift checks
│       └── cli/                    # Typer CLI (train/evaluate/explain/serve)
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── data_validation/
│   ├── api/
│   ├── explainability/
│   └── fixtures/
│
├── notebooks/                      # exploratory only — never on the prod path
│   └── exploration/
│
├── scripts/                        # one-off/dev scripts (not production code)
│
├── docker/
│   ├── api.Dockerfile
│   ├── worker.Dockerfile
│   ├── training.Dockerfile
│   └── docker-compose.yml          # api + worker + db + mlflow + minio + redis
│
├── deploy/
│   ├── k8s/                        # manifests/Helm chart (phase 2+)
│   └── terraform/                  # infra as code (future)
│
├── docs/
│   ├── TECHNICAL_SPECIFICATION.md  # this document
│   ├── model_cards/
│   ├── adr/                        # architecture decision records
│   └── api/                        # OpenAPI export, usage docs
│
├── dvc.yaml                        # DVC pipeline stages
├── dvc.lock
├── .dvcignore
├── params.yaml                     # DVC-tracked hyperparameters (mirrors configs/)
├── mlflow/                         # local mlflow server config (dev only)
├── pyproject.toml
├── requirements/
│   ├── base.txt
│   ├── dev.txt
│   └── train.txt                   # heavier GPU deps isolated from API image
├── alembic.ini
├── .env.example
├── .pre-commit-config.yaml
└── README.md
```

---

## 4. Data Flow

1. **Acquisition**: Raw scans (DICOM series or NIfTI) land in `data/raw/`, tracked by DVC pointer
   files (`.dvc`), physical bytes in a DVC remote (S3/GCS/Azure Blob/on-prem).
2. **Conversion**: DICOM → NIfTI (`dcm2niix`-style conversion) where needed; standardize orientation
   (RAS), file naming keyed by subject/session ID.
3. **Validation (schema + integrity)**: verify modality, voxel spacing sanity, non-corrupt files,
   presence of required metadata (subject ID, label, acquisition site) — reject/flag failures rather
   than silently dropping.
4. **Preprocessing** (deterministic, versioned, shared with inference):
   - Skull stripping / brain extraction.
   - Registration to a common template space (e.g., MNI152) via MONAI/ANTsPy.
   - Bias field correction (N4).
   - Intensity normalization (z-score or percentile clipping).
   - Resampling to a fixed voxel spacing/shape.
5. **Quality control**: automated checks (e.g., brain-mask coverage ratio, intensity range) plus a
   manual QC review queue for borderline cases; QC status stored as metadata.
6. **Labeling/taxonomy**: class taxonomy is **config-defined**, not hardcoded, to match whichever
   dataset is used (e.g., a 3-class CN/MCI/AD taxonomy or a 4-class Non/Very-Mild/Mild/Moderate
   taxonomy commonly used in public Kaggle Alzheimer's MRI sets). The chosen taxonomy is documented
   per dataset config.
7. **Splitting**: **subject-level** stratified train/val/test split (never slice-level or session-level
   for the same subject split across sets) to prevent data leakage; split assignment persisted as
   versioned CSVs under `data/splits/` and tracked by DVC.
8. **Dataset versioning**: each pipeline stage is a DVC stage (`dvc.yaml`); `dvc repro` regenerates
   processed data deterministically from raw + code + params hash.
9. **Consumption**: training and inference both consume data through the same `datasets/` transform
   pipeline definitions, parameterized by the same preprocessing config, guaranteeing train/serve
   parity.

---

## 5. Training Flow

1. **Entry point**: `ad-mri train --config configs/training/default.yaml` (CLI wraps the trainer).
2. **Config resolution**: base config + dataset config + model config + environment overrides merged
   and validated via pydantic; resolved config is hashed and logged.
3. **Reproducibility setup**: global seed set (Python/NumPy/PyTorch/CUDA), deterministic algorithms
   flag where feasible, DVC data hash and git commit SHA captured.
4. **Data loading**: MONAI `Dataset`/`CacheDataset` + transform pipeline (train-time augmentation:
   random flips/rotations/intensity jitter within clinically plausible bounds; val/test: deterministic
   preprocessing only).
5. **Model construction**: model factory instantiates architecture from config (e.g., a 3D CNN such as
   MONAI's `DenseNet121`/`resnet` variants, or a 2D slice-based CNN/transfer-learning backbone) —
   architecture choice is a config value, not a code fork.
6. **Training loop**:
   - Mixed-precision (AMP) where GPU available.
   - Class-imbalance handling (weighted loss or sampler) — enabled/configured explicitly, not implicit.
   - Checkpointing every N epochs + best-checkpoint-by-metric.
   - Early stopping on validation metric plateau.
   - Optional hyperparameter search via Optuna (separate CLI subcommand), each trial logged as its own
     MLflow run, nested under a parent study run.
7. **Experiment tracking (MLflow)**: params (config snapshot), metrics per epoch, learning curves,
   confusion matrix image, model artifact, environment/requirements snapshot, git commit, DVC data
   hash — all logged per run.
8. **Post-training evaluation**: full evaluation module run on held-out test set (see §Evaluation) —
   metrics only asserted from this actual run output.
9. **Model registration**: if the run passes a **quality gate** (config-defined minimum thresholds
   used as a *regression gate*, not a clinical claim), the model artifact is registered in the MLflow
   Model Registry as a new version in `Staging`.
10. **Promotion**: a separate, explicit human/CI step promotes `Staging → Production` after review of
    the evaluation report and (for CI) automated smoke tests against the API.

---

## 6. Inference Flow

> **As implemented (Phase 9/10/11)**: the core pipeline below (intake → preprocess → predict →
> respond) is real — see [`docs/inference.md`](inference.md) and [`docs/api.md`](api.md). The
> endpoint is `POST /api/v1/predictions` (this section's original design used the singular
> `/predict`). Model resolution loads a checkpoint directly (no MLflow Registry — that integration
> remains unimplemented, see `model_registry.client`), and there is no async worker queue or batch
> endpoint. PostgreSQL audit persistence (step 5 below) **is** real as of Phase 11 — see
> [`docs/database.md`](database.md) — but best-effort: a request's response is never blocked or
> failed by an unreachable audit database, only degraded observability results.

1. **Request intake**: client calls `POST /api/v1/predict` with an image (multipart upload) or a
   reference to a stored study; request validated (file type, size limits, modality sanity) before
   any heavy processing.
2. **Preprocessing**: identical transform pipeline used at training time (same package, same config
   version pinned to the served model's training config) applied to the input.
3. **Model resolution**: inference service loads the model version currently tagged `Production` in
   the MLflow Registry (or a pinned version, per environment config); model is cached in-process with
   a TTL/refresh check so repeated requests don't reload from the registry each time.
4. **Prediction**: forward pass → class probabilities → predicted class + confidence; **out-of-range/
   low-confidence** predictions flagged (config-defined threshold) rather than silently returned as
   high-confidence.
5. **Persistence**: prediction request/response metadata (redacted image reference, model version,
   predicted class, confidence, timestamp, hashed requester API key) written to PostgreSQL
   (`prediction_requests`/`prediction_results`/`model_versions` — see
   [`docs/database.md`](database.md)) — raw pixel data is never persisted, in any table, by design.
6. **Response assembly**: JSON response with class probabilities, model version, preprocessing
   version, and an `explanation_id`/link if explainability was requested inline or is available async.
7. **Explainability trigger** (optional, see §7): synchronous for lightweight cases or asynchronous
   (queued job, polled/streamed result) for LIME/SHAP given their extra compute cost.

---

## 7. Explainability Flow (LIME + SHAP)

> **As implemented (Phase 8/10/11)**: LIME/SHAP themselves are real and match §7.2/§7.3 below — see
> [`docs/explainability.md`](explainability.md). The HTTP trigger is a dedicated, synchronous
> `POST /api/v1/explanations` (this section's original design used `/explain/{prediction_id}`,
> keyed off a persisted prior prediction). A prediction audit trail now exists (Phase 11, see
> [`docs/database.md`](database.md)), but `/explanations` still takes its own image upload directly
> rather than referencing a prior prediction by id — the two calls remain independent uploads, not
> a "predict, then explain the same request" flow; there is no async worker/queue — see
> [`docs/api.md`](api.md).

**Goal**: every prediction can be paired with a human-interpretable explanation; explanations are
versioned artifacts tied to (model version, input, preprocessing version).

1. **Trigger**: either inline (`?explain=true` on `/predict`) or via a dedicated
   `POST /api/v1/explain/{prediction_id}` — the latter is preferred in production since LIME/SHAP are
   noticeably slower than the base forward pass, and is executed by the **async worker** so the API
   stays responsive.
2. **LIME**:
   - Operates on 2D representative slice(s) (or a 2D projection strategy for 3D volumes, config-
     defined) using image-segmentation-based perturbation (e.g., quickshift/SLIC superpixels).
   - Produces a superpixel-importance map for the predicted class.
   - Number of perturbation samples, segmentation algorithm, and kernel width are config-driven for
     reproducibility.
3. **SHAP**:
   - Uses a gradient-based explainer (e.g., `GradientExplainer`/`DeepExplainer`) suited to CNNs,
     with a config-defined background/reference sample set (drawn from the training set, versioned).
   - Produces per-voxel/per-pixel SHAP value maps, aggregated into an overlay heatmap.
4. **Overlay generation**: both explanation types are rendered as heatmap overlays on the original
   (or preprocessed) image slice(s), stored as image artifacts (PNG) plus raw numeric explanation
   arrays (for downstream analysis/tests).
5. **Consistency checks (explainability tests)**: sanity checks such as — explanation region overlaps
   plausible brain tissue (not background), explanation is stable under tiny input perturbations
   within tolerance, runtime stays within a configured SLA budget.
6. **Storage & versioning**: explanation artifacts stored in object storage alongside a PostgreSQL
   record (`explanation_id`, `prediction_id`, `method`, `model_version`, `artifact_uri`, `params_hash`,
   `created_at`).
7. **Comparison surfacing**: API/response layer can return LIME and SHAP side-by-side for the same
   prediction so a reviewer can visually cross-check agreement between the two explanation methods —
   the system does not assert one method is "more correct."

---

## 8. MLOps Architecture

- **Experiment tracking & model registry**: MLflow Tracking Server (backed by PostgreSQL for
  metadata, object storage — S3/MinIO — for artifacts). All training/eval/HPO runs log here.
  Model lifecycle stages: `None → Staging → Production → Archived`.
- **Data/pipeline versioning**: DVC pipelines (`dvc.yaml`) define reproducible stages
  (`convert → validate → preprocess → qc → split`); DVC remote is the same or a separate object
  store bucket from MLflow artifacts, kept logically distinct.
- **Reproducibility contract**: a trained model version is only considered valid if it can be traced
  to: git commit SHA, resolved config hash, DVC data/pipeline hash, random seed, and dependency lock
  file — all captured automatically as MLflow run tags/params.
- **CI/CD (GitHub Actions)**:
  - `ci.yml`: lint (ruff/black/mypy), unit + integration tests on every PR.
  - `data-pipeline-ci.yml`: validates `dvc repro` runs cleanly against a small smoke-test data fixture
    (not full dataset) to catch pipeline breakage early.
  - `train-smoke-test.yml`: runs a tiny (few-epoch, tiny-subset) training job to catch training-code
    regressions without needing a GPU or the full dataset.
  - `docker-build-push.yml`: builds/pushes versioned images on tag/release.
  - `security-scan.yml`: dependency and container vulnerability scanning.
- **Model promotion gate**: promotion from `Staging` to `Production` requires (a) passing the
  automated quality-gate thresholds defined in config, and (b) an explicit approval step (human
  reviewer or a protected CI environment) — never fully automatic for a system in this domain.
- **Retraining**: manually triggered in the initial phases; an automated retraining trigger
  (schedule- or drift-based) is noted as a **future** enhancement, not part of the initial build.

---

## 9. API Architecture

> **As implemented (Phase 10)**: see [`docs/api.md`](api.md) for the real endpoint list, schemas,
> auth/rate-limiting/CORS architecture, and error contract. Real: FastAPI, `/api/v1` versioning,
> Pydantic schemas, structured logging + request IDs, centralized exception handling, an API-key
> auth scheme (with a documented seam for OAuth2/JWT later), an in-memory rate limiter (with a
> documented seam for a Redis-backed one), CORS, and OpenAPI docs. Not implemented: OAuth2/JWT
> itself, the batch/idempotency-key endpoints below, and Celery-based async offload for
> explainability (`/explanations` is synchronous, offloaded per-request to a worker thread instead
> — see `docs/api.md` §4).

- **Framework**: FastAPI, versioned routes under `/api/v1/...`, OpenAPI schema auto-generated and
  exported to `docs/api/`.
- **Key endpoints** (contract-level, no implementation yet):
  - `POST /api/v1/predict` — submit an image, get classification + confidence.
  - `POST /api/v1/predict/batch` — submit multiple studies (async job, polled or webhook).
  - `POST /api/v1/explain/{prediction_id}` — request LIME/SHAP explanation for an existing prediction.
  - `GET /api/v1/predictions/{id}` — retrieve a prior prediction + explanation links.
  - `GET /api/v1/models` — list registered models/versions and their current stage.
  - `GET /healthz` / `GET /readyz` — liveness/readiness probes.
  - `GET /metrics` — Prometheus scrape endpoint.
- **Request/response contracts**: strict Pydantic schemas; input validation includes file type,
  size limits, and (for DICOM) a basic header sanity check before any processing.
- **Auth**: OAuth2/JWT for interactive/service users, API keys for machine-to-machine, scopes/roles
  (e.g., `predict:read`, `predict:write`, `admin`) enforced via FastAPI dependencies.
- **Async heavy work**: explainability and batch prediction are offloaded to Celery/RQ workers via a
  Redis broker; the API returns a job/prediction ID immediately and the client polls or subscribes.
- **Model serving**: model loaded once per worker process and cached (with a background refresh check
  against the registry's current `Production` pointer) rather than reloaded per request.
- **Rate limiting & quotas**: per-API-key rate limits (e.g., via `slowapi`/reverse-proxy level) to
  protect the inference service.
- **Idempotency**: predict requests accept an optional idempotency key to avoid duplicate processing
  on client retries.

---

## 10. Testing Strategy

> **As implemented (Phase 12)**: the "Security tests" row is real — `bandit`, `pip-audit`, and a
> Trivy container-image scan all run in CI (`.github/workflows/ci.yml`), see
> [`docs/deployment.md`](deployment.md) §3. Unit/data-validation/integration/explainability rows
> are real and already ran in CI before this phase. Regression ("golden fixture"), contract,
> load/performance, and the post-training model-quality-gate rows remain unimplemented.

| Layer | Tooling | What's covered |
|---|---|---|
| Unit tests | `pytest` | Preprocessing transforms (shape/range invariants), model forward-pass shapes, loss/metric math, config parsing/validation. |
| Data validation tests | `pytest` + schema checks (e.g., `pandera`/`great_expectations`-style rules) | Split files have no subject leakage, label taxonomy matches config, no null/corrupt entries. |
| Integration tests | `pytest` + FastAPI `TestClient` | End-to-end `/predict` and `/explain` flows against a test DB and a tiny dummy model artifact. |
| Explainability tests | `pytest` | LIME/SHAP output shapes, runtime SLA, overlay-on-brain-mask sanity checks, determinism given a fixed seed. |
| Regression tests | `pytest` + stored "golden" fixtures | A pinned model + fixed input reproduces the same prediction/metrics within tolerance across code changes. |
| Contract tests | schemathesis or Pydantic schema tests | OpenAPI schema stays backward compatible across versions. |
| Load/performance tests | Locust (or k6) | Latency/throughput under concurrent load for `/predict`; separate profile for `/explain`. |
| Security tests | `bandit`, `pip-audit`/`safety`, Trivy (containers) | Static analysis and dependency/image vulnerability scanning in CI. |
| Model quality gate | custom pytest marker run post-training | Holdout-set metrics must clear config-defined minimum thresholds before registry promotion — a regression gate, not a clinical claim. |
| Coverage target | `pytest-cov` | Enforced minimum coverage threshold in CI (value to be set per team standard, e.g., 80%+ for `src/`, excluding notebooks/scripts). |

CI runs unit/integration/data-validation/explainability/contract/security tests on every PR; load
tests and full-dataset training run outside the PR gate (scheduled/manual).

---

## 11. Deployment Strategy

> **As implemented (Phase 12)**: see [`docs/deployment.md`](deployment.md) for the real
> containerization, CI/CD, and deployment strategy. Differences from the original design below:
> a fourth `mlflow` image replaces "MinIO (S3-compatible)" (object storage stays a documented,
> unimplemented seam — `storage.object_storage.S3ObjectStorageClient`, see
> [`docs/database.md`](database.md) §4); environment config is env-var-driven
> (`deploy/environments/*.env.example`), not `configs/env/*.yaml`; CD promotion is GitHub
> Environments + a `workflow_dispatch`-driven pipeline, not a Helm/Kubernetes rollout (`deploy/k8s/`
> remains a placeholder); model rollback is `INFERENCE_MODEL_PATH` re-pointing plus a redeploy
> (there is no MLflow Registry stage-pointer integration yet, see §16); Terraform remains
> unimplemented.

- **Containerization**: separate Dockerfiles for `api`, `worker`, and `training` images (multi-stage
  builds; the API/worker images stay lean — no heavyweight training-only dependencies).
- **Local/dev**: `docker-compose.yml` spins up API + worker + PostgreSQL + MLflow + MinIO (S3-compatible)
  + Redis for a fully self-contained local environment.
- **Environments**: `dev → staging → prod`, each with its own config (`configs/env/*.yaml`), its own
  DB, and its own pinned `Production` model version (staging can run ahead of prod for validation).
- **Production orchestration**: Kubernetes (Helm chart) as the target for horizontal scaling of the
  API and worker pools independently (inference is latency-sensitive/CPU-or-GPU-bound; explainability
  workers are throughput/CPU-bound and scale separately). This is scoped as a later phase; Docker
  Compose is sufficient for early phases and demos.
- **Release strategy**: tagged releases build versioned images; rollout uses a rolling/blue-green
  strategy so a bad model or code promotion can be rolled back without downtime.
- **Model rollback**: because the model artifact is decoupled from the code deploy (fetched from the
  MLflow Registry by stage/version pointer), a bad model can be rolled back by re-pointing the
  `Production` stage without a full redeploy.
- **Infra as code**: Terraform manifests for cloud resources (object storage, managed Postgres, etc.)
  noted as a future addition once a target cloud is chosen.

---

## 12. Monitoring Strategy

> **As implemented (Phase 11/13)**: "Audit trail" is real (Phase 11, see
> [`docs/database.md`](database.md)). As of Phase 13, "System metrics" and the input-drift/
> prediction-drift halves of "Model/data monitoring" are real — see
> [`docs/monitoring.md`](monitoring.md) — using this project's own PSI/KS implementation
> (`monitoring.drift`) rather than Evidently (a declared, unused dependency — see
> [`docs/monitoring.md`](monitoring.md) §9). "Alerting" is real as Prometheus alert *rules*
> (`deploy/prometheus/alerts.yml`) reflected into two live gauges
> (`monitoring.alert_state`) and a Grafana dashboard (`deploy/grafana/`); Alertmanager routing to
> email/Slack/PagerDuty is not wired up. "Explainability SLA monitoring" (a distinct metric for
> the `/explanations` path) remains unimplemented — see
> [`docs/monitoring.md`](monitoring.md) §9 for exactly what that gap is. "Health/readiness" was
> already real since Phase 1/10 (`/healthz`, `/readyz`).

- **System metrics**: request rate, latency (p50/p95/p99), error rate, queue depth (Celery/RQ) —
  exposed via `prometheus-client` and scraped by Prometheus, visualized in Grafana.
- **Model/data monitoring**:
  - **Input drift**: distribution checks on preprocessed image statistics (e.g., intensity histograms)
    comparing live traffic to the training distribution, using a tool such as Evidently AI.
  - **Prediction drift**: monitor the distribution of predicted classes/confidence over time for
    sudden shifts.
  - **Explainability SLA monitoring**: track LIME/SHAP job latency and failure rate separately from
    the fast prediction path.
- **Audit trail**: every prediction and explanation is a row in PostgreSQL (what/when/model
  version/preprocessing version — "who" is a hashed API key, never an identity), plus a generic
  `system_events` table for other audit-worthy occurrences (engine startup, mapped error responses)
  — this doubles as the dataset for retrospective monitoring queries. See
  [`docs/database.md`](database.md).
- **Alerting**: Alertmanager (or equivalent) rules for error-rate spikes, latency SLA breaches, queue
  backlog, and drift-threshold breaches; routed to email/Slack/PagerDuty per environment.
- **Health/readiness**: `/healthz` (process alive), `/readyz` (DB reachable + model loaded), used by
  the orchestrator for restarts/traffic gating.
- **Dashboards**: a Grafana dashboard set covering system health, model-serving metrics, and a
  data/prediction-drift panel.

---

## 13. Security Considerations

> **As implemented (Phase 14 supersedes the Phase 12 callout below)**: "Secrets management" and
> "Dependency & container hygiene" are real (Phase 12 — [`docs/deployment.md`](deployment.md) §3,
> §6). **AuthN/AuthZ is now real, scoped, multi-key authorization** (Phase 14 —
> `api.deps.require_scope`, `Settings.api_keys`/`ApiScope`), not only the single-key seam; still
> not OAuth2/JWT or per-user identity — see [`docs/api.md`](api.md) §9 and
> [`docs/security.md`](security.md). **Input hardening, secure headers, and audit logging are now
> real and tested** (decompression-bomb/oversized-volume rejection, path-traversal-safe temp
> filenames, secure HTTP headers on every response, structured audit events) — see
> [`docs/security.md`](security.md) for the full requirement-by-requirement review. Transport/
> at-rest encryption and DICOM header sanitization remain as originally designed, not yet
> implemented (this system consumes NIfTI, not raw DICOM, in every phase built so far — see
> [`docs/security.md`](security.md) §7 for the residual-risk accounting).

- **Data sensitivity**: brain MRIs are potentially re-identifiable and are treated as sensitive
  health data end-to-end, regardless of whether the initial dataset is a public de-identified corpus.
- **AuthN/AuthZ**: real (Phase 14) — API-key authentication (`api.deps.require_api_key`) plus a
  scoped, multi-key authorization layer (`require_scope`, `Settings.api_keys: dict[str,
  list[ApiScope]]`) so a deployment can hand different keys different permissions (`predict` vs.
  `label`) rather than one shared superuser secret. Still not per-user identity, sessions, or
  OAuth2/JWT/RBAC — see [`docs/api.md`](api.md) §9 for the seam a stronger scheme would use.
- **Transport & at-rest encryption**: TLS everywhere in front of the API (terminated at
  ingress/load balancer); encrypted object storage and encrypted PostgreSQL storage/volumes. Not
  yet implemented — `Strict-Transport-Security` is sent unconditionally (harmless over the plain
  HTTP this sandbox runs today) so it takes effect the moment a real deployment terminates TLS.
- **Secrets management**: real — no secrets in source control; `.env` files git-ignored with
  `.env.example` as the template (every secret-shaped field left blank); production secrets via a
  secrets manager / GitHub Environment secrets, injected at deploy time; `Settings
  ._validate_production_secrets` refuses to start in `APP_ENV=prod` with a default `SECRET_KEY` or
  no configured API key. See [`docs/deployment.md`](deployment.md) §6.
- **Input hardening**: real and tested (Phase 14) — file-type/size validation on uploads (§3 of
  [`docs/api.md`](api.md)), a decompression-bomb guard (`PreprocessingConfig.max_voxel_count`: a
  compressed NIfTI's cheaply-readable header can declare an enormous voxel count before its body
  is ever fully decompressed — rejected before `LoadImaged` would materialize it), and
  path-traversal-safe temp filenames (an uploaded file is always renamed to a generic
  `upload<ext>`, never round-tripping the client's filename into a path). DICOM header
  sanitization does not apply yet — every phase built so far consumes NIfTI, not raw DICOM.
- **PHI-aware logging**: application logs never contain raw pixel data, full file paths that embed
  patient identifiers, or PHI fields; only opaque hashes/IDs are logged (see §15).
- **Dependency & container hygiene**: real — automated dependency vulnerability scanning
  (`pip-audit`, zero known vulnerabilities as of Phase 14's review), static analysis (`bandit`,
  zero findings), container image scanning (Trivy), minimal non-root base images.
- **Rate limiting & abuse protection**: real — per-key/per-IP sliding-window rate limits
  (`RateLimitMiddleware`), request size caps (`MAX_UPLOAD_SIZE_MB`), and audit logging of
  auth/rate-limit rejections and unexpected errors (`api.audit.record_event`).
- **Secure response headers**: real (Phase 14) — `SecurityHeadersMiddleware` sets
  `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, `Permissions-Policy`,
  `Strict-Transport-Security`, and a route-appropriate `Content-Security-Policy` on every response,
  including one produced by a genuinely unhandled exception — see [`docs/security.md`](security.md).
- **Regulatory posture (explicit disclaimer)**: the system is built with compliant-by-construction
  practices (audit trail, access control, de-identification hooks) but **is not represented as HIPAA-
  certified or FDA-cleared**; any real clinical deployment requires a separate compliance/regulatory
  review outside this engineering scope.

---

## 14. Configuration Management

- **Mechanism**: layered YAML configs (`configs/base.yaml` + `configs/{data,model,training,
  explainability}/*.yaml` + `configs/env/{dev,staging,prod}.yaml`), merged and **validated through
  pydantic-settings models** — invalid/unknown config keys fail fast at startup, not silently ignored.
- **Precedence**: defaults → dataset/model/training-specific file → environment override → explicit
  environment variables (for secrets/deployment-specific values only, never for scientific parameters).
- **No hardcoded literals**: class taxonomy, preprocessing parameters, model architecture choice,
  training hyperparameters, explainability parameters, and quality-gate thresholds are all
  config-driven and version-controlled.
- **Config provenance**: the fully resolved config for a given run is hashed and logged as an MLflow
  artifact/tag, so any run's exact configuration is retrievable later.
- **Secrets separation**: `configs/` never contains credentials; those come exclusively from
  environment variables / secrets manager, referenced by name only.

---

## 15. Logging Strategy

- **Format**: structured JSON logging (`structlog` or stdlib `logging` + JSON formatter) suitable for
  ingestion by a log aggregator (ELK/Loki/CloudWatch, per deployment target).
- **Correlation**: every request gets a `request_id`; async jobs propagate the originating
  `request_id`/`prediction_id` so a full trace (predict → queue → explain → response) can be
  reconstructed.
- **Log streams kept separate**:
  - **Application/system logs** (service lifecycle, errors, performance).
  - **Audit logs** (who requested what prediction/explanation, when, against which model version) —
    written to PostgreSQL as the durable source of truth, optionally mirrored to the log aggregator.
  - **Training logs** (per-run, captured by MLflow alongside metrics, not mixed with serving logs).
- **Sensitivity rule**: no raw image data, no free-text patient identifiers, no full DICOM headers in
  logs — only opaque IDs/hashes, consistent with §13.
- **Levels**: `DEBUG` (dev only) / `INFO` (default) / `WARNING` / `ERROR` / `CRITICAL`, level
  configurable per environment via config, not hardcoded.

---

## 16. Model Versioning Strategy

> **As implemented (Phase 11/13)**: the MLflow Model Registry described below remains unimplemented
> (`model_registry.client`). A separate, much narrower `model_versions` PostgreSQL table exists
> (Phase 11, see [`docs/database.md`](database.md)) — it is not this registry: it only records
> which (model config hash, preprocessing config hash) pairs this deployment has actually served,
> for the audit trail, with no lifecycle-stage tracking or promotion workflow. Phase 13 adds one
> monitoring-specific column to it, `baseline_metrics` — the held-out evaluation metrics
> `ad-mri monitor register-baseline` records for a version, compared against delayed,
> ground-truth-backed production performance by `ad-mri monitor model-drift-report` — see
> [`docs/monitoring.md`](monitoring.md) §5.

- **Registry**: MLflow Model Registry is the single source of truth for model versions and their
  lifecycle stage (`None/Staging/Production/Archived`).
- **Traceability tuple**: every registered model version carries, as tags/params: git commit SHA,
  resolved config hash, DVC data-pipeline hash (i.e., which exact processed dataset it trained on),
  dependency lock hash, and random seed — enabling exact reproduction.
- **Semantic pairing with code**: application/serving code is tagged/released with standard semver;
  a compatibility matrix (which code versions can serve which model-schema versions) is documented in
  `docs/adr/` as it evolves, since the model's expected input/output contract can change across
  versions.
- **Model card**: each registered version gets a generated model card (`docs/model_cards/`) recording
  training data description, class taxonomy, **measured** evaluation metrics (never assumed), known
  limitations, and intended use/non-use statement.
- **Promotion workflow**: `Staging` = passed automated quality gate; `Production` = additionally
  human-approved; `Archived` = superseded, retained for audit/rollback.

---

## 17. Dataset Versioning Strategy

- **DVC as source of truth** for raw and processed data, plus split assignment files — all tracked
  via `.dvc` pointer files in git, physical data in a DVC remote.
- **Pipeline-as-code**: `dvc.yaml` defines each stage (`convert → validate → preprocess → qc → split`)
  with explicit dependencies/outputs, so `dvc repro` deterministically regenerates any stage when its
  inputs or params change, and `dvc dag` documents the lineage visually.
- **Params tracking**: `params.yaml` (DVC-tracked) mirrors the relevant subset of `configs/` so data-
  pipeline parameter changes are diffable and tied to specific DVC pipeline runs.
- **Lineage linkage**: a training run's MLflow record stores the DVC commit/hash of the exact
  processed dataset + split files it consumed, closing the loop from raw data → processed data →
  split → trained model.
- **Multiple dataset sources**: each candidate public dataset (e.g., OASIS, ADNI-derived, a Kaggle
  Alzheimer's MRI collection) gets its own `configs/data/dataset_*.yaml` and is versioned
  independently, so results are never silently mixed across incompatible label taxonomies or
  preprocessing assumptions without an explicit, documented harmonization step.

---

## Summary

### Architecture
Modular, layered, service-oriented system separating data pipeline (DVC-versioned), training
(PyTorch/MONAI + MLflow-tracked), serving (FastAPI + async workers for explainability), and
persistence (PostgreSQL + object storage), with MLOps (MLflow, DVC, CI/CD, monitoring) as a
cross-cutting platform layer. See §1 diagram.

### Folder structure
See §3 — `src/ad_mri/` as an installable package with `data_pipeline`, `datasets`, `modeling`,
`training`, `evaluation`, `explainability`, `inference`, `api`, `worker`, `db`, `monitoring`, `cli`
submodules; `configs/`, `tests/`, `docker/`, `deploy/`, `docs/` as siblings; `data/` and DVC files at
the root.

### Technology decisions
Python 3.11+, PyTorch + MONAI (modeling/medical transforms), FastAPI (serving), PostgreSQL
(metadata/audit), MLflow (tracking + registry), DVC (data/pipeline versioning), Docker/Compose/
(K8s later) for deployment, Celery/RQ + Redis for async explainability jobs, LIME + SHAP for
explainability, Prometheus/Grafana for monitoring, GitHub Actions for CI/CD, pytest for testing.

### Data flow
Raw DICOM/NIfTI → validate → preprocess (skull-strip, register, normalize, resample) → QC →
subject-level stratified split → DVC-versioned processed dataset → consumed identically by training
and inference through a shared transform pipeline (§4).

### Component responsibilities
See module table in §2 — each module owns one concern (data, modeling, training, evaluation,
explainability, serving, persistence, monitoring), communicating through well-defined interfaces
(config objects, DVC artifacts, MLflow registry, DB models, Pydantic schemas).

### Development phases
0. Repo scaffolding, config system, CI skeleton.
1. Data pipeline + DVC (ingestion → preprocessing → QC → split).
2. Baseline model training + MLflow tracking + evaluation module (confusion matrix, metrics).
3. Literature-comparison framework (structured, citation-backed, no invented numbers).
4. Explainability module (LIME + SHAP) with its own tests.
5. Inference API (FastAPI) + async worker for explainability.
6. MLOps hardening: model registry promotion workflow, full CI/CD, quality gates.
7. Monitoring/observability (Prometheus/Grafana, drift checks, audit trail).
8. Security hardening (auth, secrets, input hardening, scanning) + deployment (Compose → K8s).
9. Documentation: model cards, ADRs, API docs, production-readiness review.

### Risks
- **Subject-level data leakage** if splitting is done at slice/session level instead of subject level.
- **Class imbalance** in typical Alzheimer's MRI datasets skewing metrics/training.
- **Small dataset sizes** relative to 3D CNN capacity → overfitting risk; mitigated via transfer
  learning, augmentation, and regularization choices made explicit in config.
- **Domain shift** across scanners/sites/protocols hurting generalization; requires explicit
  documentation of which data sources a given model was trained/validated on.
- **Explainability compute cost/latency**, especially SHAP on 3D volumes — mitigated by async
  processing and 2D-slice strategies, documented as a deliberate trade-off.
- **Dataset licensing/PHI/ethics constraints** (ADNI/OASIS usage terms) — must be respected by
  whoever supplies the data; out of engineering control but the system is designed to support it
  (audit trail, de-identification hooks).
- **Reproducibility drift** if any of {config, data hash, code commit, seed} isn't captured — mitigated
  by the traceability tuple in §16.
- **Over-claiming risk**: any performance number in reports/dashboards must always be sourced from an
  actual MLflow run, never asserted from memory or assumption — this is a process discipline as much
  as an engineering one.

### Recommended implementation order
1. Repo scaffolding + config/logging/core utilities + CI skeleton (Phase 0).
2. Data pipeline + DVC integration (Phase 1) — this unblocks everything downstream.
3. Baseline CNN training + MLflow tracking + evaluation/confusion-matrix module (Phase 2).
4. Literature-comparison module (Phase 3) — depends only on evaluation module, low risk, high value
   early to keep expectations calibrated.
5. Explainability (LIME/SHAP) module (Phase 4) — depends on a trained model existing.
6. Inference API + async worker (Phase 5) — depends on model registry + explainability module.
7. MLOps hardening / promotion workflow / full CI-CD (Phase 6).
8. Monitoring & observability (Phase 7).
9. Security hardening + deployment packaging (Phase 8).
10. Documentation (model cards/ADRs) + production-readiness review (Phase 9) — continuous, but
    finalized last.

---

**Next phase**: awaiting confirmation to begin **Phase 0 implementation** (repo scaffolding, config
system, CI skeleton) per the order above.
