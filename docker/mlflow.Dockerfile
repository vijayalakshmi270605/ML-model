# syntax=docker/dockerfile:1
# A dedicated MLflow Tracking Server image — separate from the api/worker/
# training images (none of which need to *run* a tracking server, only talk
# to one via MLFLOW_TRACKING_URI). Backed by PostgreSQL for the backend
# store (see docker-compose.yml's `mlflow` service) so run/experiment
# metadata survives a container restart; artifacts go to a mounted volume
# (local dev) or object storage (production — see docs/deployment.md).
FROM python:3.15-slim AS runtime

ARG APP_VERSION=0.0.0
ARG GIT_COMMIT=unknown
ARG BUILD_DATE=unknown
LABEL org.opencontainers.image.title="ad-mri-mlflow" \
      org.opencontainers.image.description="MLflow Tracking Server for the ad-mri training pipeline." \
      org.opencontainers.image.version="${APP_VERSION}" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.source="https://github.com/ad-mri/ad-mri"

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# curl's version tracks whatever the python:3.11-slim base image's Debian
# snapshot currently ships.
# hadolint ignore=DL3008
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser \
    && apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Pinned to the same MLflow major.minor the `train` extra installs
# (pyproject.toml) so a run logged by the training image is always read by a
# server that understands its schema — intentionally a floor, not an exact
# pin, so it tracks MLflow 2.x patch releases.
# hadolint ignore=DL3013
RUN pip install --upgrade pip && pip install "mlflow>=2.12" "psycopg2-binary>=2.9"

RUN mkdir -p /mlartifacts && chown appuser:appuser /mlartifacts
# Numeric UID:GID — see api.Dockerfile's USER comment.
USER 1000:1000
WORKDIR /home/appuser

EXPOSE 5000

# `curl --fail` alone already exits non-zero on failure.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["curl", "--fail", "http://localhost:5000/health"]

# MLFLOW_BACKEND_STORE_URI / MLFLOW_ARTIFACT_ROOT are supplied at runtime
# (docker-compose.yml) — never hardcoded here, same convention as every
# other image in this repository.
ENTRYPOINT ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
