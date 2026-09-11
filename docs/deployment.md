# Production deployment

Status: **Phase 12 — containerization, CI/CD, and a deployment strategy implemented.** Nothing
here is a clinical claim: this system is research / decision-support tooling, it is **not a
certified medical device**, and no deployment step in this document turns it into one — see
[`docs/TECHNICAL_SPECIFICATION.md`](TECHNICAL_SPECIFICATION.md) §0. **The model this system
serves has not been clinically validated.** Deploying it to any environment — development,
staging, or production — does not change that; "production" here means "the environment real
non-development traffic reaches," not "clinically approved."

## 1. Containerization

Four images, one per logically-separate concern (`docker/*.Dockerfile`), all multi-stage
(a `builder` stage installs into a venv; the `runtime` stage copies only that venv — no
compiler toolchain ships in the final image), all non-root (`USER 1000:1000`), all carrying
OCI labels (`org.opencontainers.image.version`/`.revision`/`.created`, populated by
`--build-arg` from CI — see §3):

| Image | Built from | Runs |
|---|---|---|
| `ad-mri-api` | `docker/api.Dockerfile` | The FastAPI service (`docs/api.md`) — also doubles as the one-off Alembic migration job (`docker compose run --rm migrate`), since it already carries `alembic.ini` + `db/migrations/`. Installs `.[train]`, not just the base package (**corrected in Phase 13**: since Phase 9's synchronous, in-process `InferenceEngine`, `/predictions` needs torch/monai in the api process itself; the original Phase 12 image only installed base dependencies, which would have `ModuleNotFoundError`ed the first time a real checkpoint was configured — never caught earlier because no request against a fully-built api image was ever exercised in this sandbox, see §9). |
| `ad-mri-worker` | `docker/worker.Dockerfile` | The Celery worker — async explainability/batch jobs, off the request path. |
| `ad-mri-training` | `docker/training.Dockerfile` | Offline `ad-mri train`/`evaluate`/... batch jobs — never long-running, no `HEALTHCHECK`. |
| `ad-mri-mlflow` | `docker/mlflow.Dockerfile` | An MLflow Tracking Server, backed by PostgreSQL. |

`.dockerignore` keeps secrets, local state, and dev-only files (`.env`, `data/`, `tests/`,
`.git/`, ...) out of every build context — nothing in any image is a credential.

### "Model inference" is a module, not a fifth container — deliberately

The requirement lists API / model inference / PostgreSQL / MLflow as logically separate
services. PostgreSQL and MLflow are indeed separate containers (`db`, `mlflow`). Model
inference is separated **logically but not as its own network service**: it is
`inference.predictor.InferenceEngine` (Phase 9), a module with no HTTP/gRPC boundary of its
own, called in-process by `api` on the synchronous request path and by `worker` for
async/batch work — two independently deployable, independently scalable containers, neither of
which duplicates the inference logic itself (`docs/api.md`'s "the API must call the inference
service rather than duplicating model logic" requirement, satisfied at the module boundary,
not a network one). Splitting it into a fifth container (the API calling an inference
container over the network) would fork or wrap that already-tested contract for no
requirement this system currently has — there is no independent inference-only scaling need
yet (e.g. GPU-specific autoscaling distinct from the API's own scaling). If that need appears,
`InferenceEngine` is already the seam a dedicated inference service would wrap; nothing about
today's design blocks it.

### Health checks vs. readiness checks

`api`/`mlflow` carry a Docker `HEALTHCHECK` against `/healthz` (liveness) only, deliberately —
`/readyz` reports 503 when a *downstream* dependency (the database) is unreachable, which must
not make the container runtime kill and restart an otherwise-healthy process. `worker`'s
`HEALTHCHECK` runs `celery inspect ping`, round-tripping through the real broker. An
orchestrator (Kubernetes, ECS, ...) should instead configure **both** probes explicitly and
distinctly:

- **liveness**: `GET /healthz` — restart the container if this fails.
- **readiness**: `GET /readyz` — remove the container from the load-balancing pool (do **not**
  restart it) if this fails; add it back once it passes again.

This project does not yet ship Kubernetes manifests (`deploy/k8s/` is a placeholder, like
`model_registry.client`'s documented-but-unimplemented seam) — the two probe paths above are
the only orchestrator-specific piece a real manifest would need.

## 2. docker-compose: local development vs. production

Two files, deliberately not one with profiles/overrides for everything — the differences are
large enough (build-from-source vs. pull-a-pinned-tag; permissive dev defaults vs. hardened
prod settings) that one shared file would need more conditionals than either file's own logic:

- **`docker/docker-compose.yml`** (local development): builds every image from source, exposes
  every service's port to `localhost` for convenience, reads `../.env` (git-ignored — copy from
  `.env.example`). Includes a `migrate` one-off job (`profiles: [tools]`, so it never starts
  with a plain `up`) and an `mlflow` service backed by a second `mlflow` database on the same
  `db` container (created by `docker/init-mlflow-db.sh` on first init).
- **`docker/docker-compose.prod.yml`**: never builds — pulls
  `${IMAGE_REGISTRY}/${IMAGE_REPO}-<service>:${IMAGE_TAG}` (both `IMAGE_REGISTRY` and
  `IMAGE_TAG` are *required* — `${IMAGE_TAG:?set IMAGE_TAG}` fails the compose invocation
  outright rather than silently deploying `:latest` if either is unset). Only `api`'s port is
  published; `db`/`redis`/`mlflow` stay on the internal `ad_mri_net` network only. Every service
  runs with `security_opt: [no-new-privileges:true]`, `cap_drop: [ALL]`, and (`api`/`worker`)
  `read_only: true` root filesystem with a `tmpfs` mount at `/tmp` for per-request upload temp
  dirs. Reads secrets from `${SECRETS_ENV_FILE:-/etc/ad-mri/production.env}` — a path outside
  this repository entirely (§6).

Both files run every service under **resource limits** (`deploy.resources.limits`/
`.reservations` — applied by plain `docker compose up`, Compose V2, no Swarm required) sized
for local dev vs. production scale respectively, and bound container log growth
(`logging: json-file, max-size/max-file`).

### `db`/`redis`/`mlflow` in production: self-hosted fallback, not a mandate

`docker-compose.prod.yml`'s `db`/`redis`/`mlflow` services are a single-host fallback. A larger
deployment typically points `DATABASE_URL`/`REDIS_URL`/`MLFLOW_TRACKING_URI` (in the external
secrets env file) at managed equivalents (RDS/ElastiCache/a managed MLflow) instead and omits
these three services from the compose invocation entirely — nothing in `api`/`worker` cares
which; both only ever see a URL.

## 3. CI pipeline (`.github/workflows/ci.yml`)

Runs on every push to `main`/`master` and every pull request. Six jobs implementing the
required 7-step pipeline (install is an explicit, named step inside each job — not a separate
top-level job, since every job needs its own runner and dependency set anyway):

| Job | Steps |
|---|---|
| **Lint** | install `.[dev]` → `ruff check` → `black --check` |
| **Type check** | install `.[dev]` → `mypy src` (needs no `train` extra — every heavy dependency this codebase has is imported lazily inside functions, exactly so type-checking never needs torch/monai installed) |
| **Security checks** | install `.[dev]` → `bandit -c pyproject.toml -r src` (SAST) → `pip-audit --skip-editable` (dependency vulnerabilities) |
| **Unit tests** | install `.[dev,train]` → `pytest tests/unit tests/api tests/data_validation tests/explainability` with coverage, uploaded as an artifact |
| **Integration tests** | install `.[dev,train]` → `pytest tests/integration` |
| **Build image** *(matrix: api/worker/training/mlflow)* | `docker buildx build` (with GitHub Actions layer caching) → Trivy scan (fails on CRITICAL/HIGH, unfixed CVEs ignored) → push to GHCR **only** on a push to the default branch, using the workflow's own auto-issued `GITHUB_TOKEN` — never a stored secret |

Unit/integration tests and the build job both `needs:` lint + type-check, so a trivial
formatting mistake fails in under a minute rather than after a multi-minute `torch` install.
Build additionally `needs:` unit tests, integration tests, and security — nothing is ever
pushed to the registry from a state that hasn't passed every other check.

Verified for real in this environment (no live GitHub Actions runner was available, so this is
schema/lint validation, not an executed run — see §9): `actionlint` (with `shellcheck` enabled
for every embedded `run:` block) reports zero errors against both workflow files and the
composite action below; `yamllint` reports zero errors/warnings.

## 4. CD architecture: development → staging → production (`.github/workflows/cd.yml`)

- **`development`**: automatic. A `workflow_run` trigger fires once CI succeeds on the default
  branch and deploys that exact commit's already-built image — the fastest, least-gated path.
- **`staging`** / **`production`**: always a deliberate `workflow_dispatch`, naming one
  already-CI-passed `image_tag` (a short git SHA) — **never a fresh build**. Promoting code
  means re-deploying the identical artifact that was validated in the previous stage, not
  rebuilding from source a second time (which could silently pull a newer dependency version
  between builds).
- **Promotion gating**: the `environment:` key on each deploy job — GitHub's own native
  Environments feature. Configure `staging` and `production` as protected environments (repo
  Settings → Environments) with required reviewers before this repository's first real
  deployment; that protection rule, not anything in this YAML, is what actually blocks an
  unreviewed promotion. This also gives each environment its **own** scoped secrets (§6) — a
  `staging` secret is never visible to a `production` job and vice versa.

### Rollback capability

Implemented, not just documented as a manual runbook step: the composite action
`.github/actions/deploy` (used by every CD job) records the currently-deployed tag in a marker
file on the target host, deploys the new tag, then polls the deployed environment's `/readyz`
endpoint for up to 100 seconds. If it never reports ready, the action **automatically
redeploys the previous tag** and fails the job loudly (so the failure is visible, not silently
swallowed) — a bad deploy self-heals without waiting for a human to notice and manually
intervene. A manual rollback is the same mechanism used for promotion: re-dispatch this
workflow with `image_tag` set to any previous known-good short SHA.

Verified for real (§9): the embedded deploy script was extracted and run through `shellcheck`
at its strictest (`-S style`) severity level — zero findings, including for the intentionally
unquoted `$COMPOSE` variable (a deliberate word-split to build a multi-word command from a
single string, not an oversight).

## 5. Versioned model artifacts

Two independent version axes, never conflated:

- **Serving code version** — the Docker image tag (a git SHA, per §3/§4). `GET /api/v1/version`
  reports this (`ad_mri.__version__`, see `docs/api.md`).
- **Model version** — `INFERENCE_MODEL_PATH` pins one immutable checkpoint artifact (never a
  mutable "latest" pointer); `model_config_hash`/`preprocessing_config_hash`
  (`modeling.model_config.ModelConfig.compute_hash()`) content-address it, and
  `db.models.ModelVersionRecord` (Phase 11, see [`docs/database.md`](database.md)) durably
  records which version has actually been served, when. `GET /api/v1/models` reports the
  version currently loaded.

Promoting a *model* (not code) from staging to production means changing `INFERENCE_MODEL_PATH`
in the target environment's secrets file (§6) to a checkpoint already validated in staging, then
redeploying — the same `workflow_dispatch` path as any other promotion, since it changes what
`docker-compose.prod.yml` pulls in as configuration, not the image itself.

## 6. Secrets management

**No secret is ever committed to this repository, in any form — not a real value, not a
filled-in template, not a `.env` file.** Concretely:

| Layer | Where secrets live | Never here |
|---|---|---|
| CI (`ci.yml`) | Nothing beyond the auto-issued, per-run `GITHUB_TOKEN` (GHCR push) — no secret this repository controls the value of. | Any `${{ secrets.* }}` reference in `ci.yml`. |
| CD (`cd.yml`) | GitHub Environment secrets (`DEPLOY_HOST`, `DEPLOY_USER`, `DEPLOY_SSH_KEY`), scoped per-environment (repo Settings → Environments → `<environment>` → Secrets). | This repository's regular (non-Environment) secrets — those are visible to every workflow run, defeating the staging/production isolation §4 relies on. |
| The running application (`api`/`worker`/`mlflow` containers) | `docker-compose.prod.yml`'s `${SECRETS_ENV_FILE:-/etc/ad-mri/production.env}` — a file placed on the target host by the deployment process (provisioning/config-management tooling, or a cloud secrets manager rendering it at boot), outside this repository's working tree entirely. | `docker-compose.yml`/`docker-compose.prod.yml` themselves, `.env.example`, or any file under `deploy/environments/`. |
| Local development | `.env` (git-ignored — `.gitignore` already excludes it; copy from `.env.example`). | Committed, ever. |

`deploy/environments/{development,staging,production}.env.example` document every
environment-specific **non-secret** setting (log format, rate limits, DB pool sizing, ...);
every secret-shaped field in them is deliberately left blank with a comment pointing back here
— see that directory's own `README.md`.

`core.config.Settings._validate_production_secrets` is a second, independent enforcement layer:
with `APP_ENV=prod`, the process refuses to start at all if `SECRET_KEY` is still its insecure
default, or if `API_KEY_REQUIRED=true` and neither `API_KEY` nor `API_KEYS` (Phase 14's scoped
multi-key architecture — see [`docs/security.md`](security.md) §4 and [`docs/api.md`](api.md) §9)
is set — a misconfigured production deployment fails loudly at startup, not silently at the first
request that needed the secret that was never set.

## 7. Structured logs & resource limits

- **Structured logs**: `LOG_FORMAT=json` in every deployed environment template (§6) —
  `core.logging.configure_logging` (Phase 1) already renders every log line as one JSON object
  via `structlog`; nothing new was needed here beyond setting the flag consistently. Every
  compose service also bounds the Docker log driver itself (`json-file`, `max-size`/`max-file`)
  so structured logs never fill a host's disk.
- **Resource limits**: `deploy.resources.limits`/`.reservations` per service, in both compose
  files (§2) — CPU and memory ceilings sized for local-dev vs. production scale.

## 8. Environment-specific configuration

`core.config.Settings` (env-var-driven, `pydantic-settings`) already made this possible since
Phase 1; this phase adds the concrete per-environment templates
(`deploy/environments/*.env.example`, §6) that make the *differences* explicit rather than
implicit:

| Setting | dev | staging | production |
|---|---|---|---|
| `DEBUG` | `true` | `false` | `false` |
| `LOG_FORMAT` | `json` | `json` | `json` |
| `API_KEY_REQUIRED` | `false` | `true` | `true` |
| `DB_POOL_SIZE` | 5 | 10 | 20 |
| `RATE_LIMIT_PER_MINUTE` | 120 | 60 | 60 |

## 9. What was, and was not, executed for real

Consistent with this project's established practice of never claiming an unverifiable result:

- **Executed for real in this environment**: `hadolint` against all four Dockerfiles (zero
  findings after fixes — see each Dockerfile's `hadolint ignore=` comments for the two that
  remain, each justified inline); `docker compose config` against both compose files (resolves
  correctly, including the `migrate`/`tools`-profile service and the `IMAGE_TAG`/
  `MLFLOW_BACKEND_STORE_URI` required-variable guards genuinely failing when unset);
  `actionlint` (with `shellcheck` enabled) against both workflow files and the composite deploy
  action (zero errors); `yamllint` against every new/changed YAML file (zero errors); `bandit`
  and `pip-audit` against this codebase's actual dependency tree (see §3's table — this is what
  CI's "Security checks" job runs, run here first to establish a clean baseline, catching and
  fixing a real `torch.load(weights_only=False)` finding along the way, see
  `modeling.checkpoint.load_checkpoint`); the full existing test suite, to confirm that fix
  didn't break anything.
- **Not executed for real, and cannot be from this sandbox**: an actual GitHub Actions run (no
  GitHub remote/runner is reachable here), an actual `docker build`/`docker compose up` (the
  local Docker daemon is not running in this environment — the same limitation noted in
  `docs/database.md` §9 for PostgreSQL), and an actual SSH deploy to a real host (no target host
  exists). The YAML/Dockerfile/compose content itself was validated as thoroughly as tooling
  allows without those; running it live is the next real step once a GitHub remote, a running
  Docker daemon, and a deploy target all exist.

## 10. Step-by-step: first production deployment

Everything above is architecture; this is the concrete runbook. One-time setup, then the
repeatable path.

### 10.1 One-time setup

1. Push this repository to GitHub (`git remote add origin ...`, `git push`).
2. In repo Settings → Environments, create `development`, `staging`, and `production`. Add
   required reviewers to `staging`/`production` (this is the actual promotion gate — §4).
3. In each Environment, add secrets `DEPLOY_HOST`, `DEPLOY_USER`, `DEPLOY_SSH_KEY` (a private key
   whose public half is authorized on that environment's target host) — never as repository-level
   secrets, which would be visible to every workflow run regardless of environment (§6).
4. In each Environment, add variables (not secrets — these aren't sensitive) `IMAGE_REPO` (e.g.
   `your-org/ad-mri`), `DEPLOY_PATH` (e.g. `/opt/ad-mri`), and `HEALTH_URL` (e.g.
   `https://staging.example.com/api/v1/ready`).
5. On each target host: `mkdir -p /opt/ad-mri && cd /opt/ad-mri`, copy `docker/docker-compose.prod.yml`
   there, and create `/etc/ad-mri/production.env` (mode `600`, owned by the deploy user) by
   starting from `deploy/environments/production.env.example` and filling in every blank field
   with that environment's real secret — this file is created directly on the host, **never**
   committed, transmitted through a workflow log, or round-tripped through this repository.
6. Ensure the target host has Docker + the Compose plugin installed, and that the deploy user can
   run `docker` (e.g. is in the `docker` group).

### 10.2 Every deployment after that

1. Open a PR; CI (§3) runs lint/type-check/security/unit/integration tests and builds + scans
   every image. Merge once it's green.
2. The merge to the default branch triggers CD's `development` job automatically (§4) — no
   action needed.
3. Once satisfied with `development`, promote the same image tag: **Actions → CD → Run
   workflow**, `target_environment: staging`, `image_tag: <the short SHA from the CI run>`.
   Approve the environment protection prompt if you configured required reviewers.
4. Validate in staging (this is what staging is *for* — a bad release candidate is caught here,
   not in production).
5. Promote the identical tag to production the same way: **Run workflow**,
   `target_environment: production`, the same `image_tag`.
6. The deploy job pulls that tag, runs pending Alembic migrations, brings the stack up, and polls
   `HEALTH_URL`. A failing health check triggers the automatic rollback described in §4 — check
   the workflow run's logs for what happened either way.

### 10.3 Rolling back

Re-run **Actions → CD → Run workflow** against the target environment with `image_tag` set to
any previous known-good short SHA (visible in the CI run history, or in
`/opt/ad-mri/.current_image_tag`'s history via the target host's own deploy logs). This is the
same mechanism as a forward promotion — there is no separate "rollback workflow" to keep in
sync with the real one.
