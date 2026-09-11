# syntax=docker/dockerfile:1
# Build context is the repo root — see docker/docker-compose.yml.
FROM python:3.11-slim AS builder

WORKDIR /build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# `.[train]`, not the bare package: since Phase 9, InferenceEngine runs
# synchronously in-process inside the API (inference.predictor's module
# docstring), so this image needs torch/monai too, exactly like worker's -
# see pyproject.toml's `dependencies` comment for the full explanation of
# why "the lean API image" stopped being literally true once that design
# was chosen. Its own dependency versions are pinned in pyproject.toml, the
# single source of truth.
# hadolint ignore=DL3013
RUN pip install --upgrade pip && pip install ".[train]"

FROM python:3.11-slim AS runtime

# Populated by `docker build --build-arg ...` (see .github/workflows/ci.yml /
# docs/deployment.md "Versioned model artifacts" — this labels the *serving
# code's* version; the *model's* version is a separate, independently
# tracked value, see db.models.ModelVersionRecord / GET /api/v1/models).
ARG APP_VERSION=0.0.0
ARG GIT_COMMIT=unknown
ARG BUILD_DATE=unknown
LABEL org.opencontainers.image.title="ad-mri-api" \
      org.opencontainers.image.description="Alzheimer's Disease MRI classification — REST API (research / decision-support; not a certified medical device)." \
      org.opencontainers.image.version="${APP_VERSION}" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.source="https://github.com/ad-mri/ad-mri"

# curl only, for the HEALTHCHECK below — no compiler/dev toolchain needed at
# runtime, keeping the final image lean. Its version tracks whatever the
# python:3.11-slim base image's Debian snapshot currently ships; pinning it
# here would just mean this build breaks whenever that snapshot moves.
# hadolint ignore=DL3008
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser \
    && apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY configs ./configs
# alembic.ini + db/migrations (already under src/, copied below) — this image
# also serves as the migration job (`docker compose run --rm migrate`, see
# docker-compose.yml); it is not a second image to build/scan/patch.
COPY alembic.ini ./alembic.ini
COPY --chown=appuser:appuser src ./src

# Numeric UID:GID (not the name) so bind-mounted volume ownership resolves
# identically on any host, container runtime, or orchestrator (e.g.
# Kubernetes' runAsUser) without needing /etc/passwd lookups.
USER 1000:1000
EXPOSE 8000

# Liveness only (matches Settings/README): a 503 from /readyz (a downstream
# dependency, e.g. the database, unreachable) must not make the orchestrator
# kill and restart an otherwise-healthy process — that is exactly what a
# *readiness* probe (configured at the orchestrator level, see
# docs/deployment.md "Health checks & readiness checks") is for instead.
# `curl --fail` alone already exits non-zero on an HTTP error or a refused
# connection, so this needs no extra `|| exit 1`.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["curl", "--fail", "http://localhost:8000/healthz"]

# All configuration is supplied via environment variables at runtime (see .env.example) —
# nothing in this image is a credential or an environment-specific path.
CMD ["uvicorn", "ad_mri.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
