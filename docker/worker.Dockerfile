# syntax=docker/dockerfile:1
# Build context is the repo root — see docker/docker-compose.yml.
FROM python:3.14-slim AS builder

WORKDIR /build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# Installs this project itself; its own dependency versions are pinned in
# pyproject.toml, the single source of truth.
# hadolint ignore=DL3013
RUN pip install --upgrade pip && pip install ".[train]"

FROM python:3.14-slim AS runtime

ARG APP_VERSION=0.0.0
ARG GIT_COMMIT=unknown
ARG BUILD_DATE=unknown
LABEL org.opencontainers.image.title="ad-mri-worker" \
      org.opencontainers.image.description="Alzheimer's Disease MRI classification — async explainability/batch worker." \
      org.opencontainers.image.version="${APP_VERSION}" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.source="https://github.com/ad-mri/ad-mri"

RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY configs ./configs
COPY --chown=appuser:appuser src ./src

USER 1000:1000

# `celery inspect ping` round-trips through the real broker (REDIS_URL) —
# an unreachable broker fails this exactly as it would fail an actual task.
# Shell form is required here: $HOSTNAME must be expanded by a shell, which
# JSON exec form never runs.
# hadolint ignore=DL3025
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD celery -A ad_mri.worker.celery_app inspect ping -d celery@$HOSTNAME || exit 1

# Runs LIME/SHAP explainability jobs and batch predictions off the request path
# (see docs/TECHNICAL_SPECIFICATION.md §7, §9). Implemented in the explainability phase.
CMD ["celery", "-A", "ad_mri.worker.celery_app", "worker", "--loglevel=info"]
