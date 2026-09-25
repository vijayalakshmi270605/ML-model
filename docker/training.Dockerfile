# syntax=docker/dockerfile:1
FROM python:3.12-slim AS builder

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

FROM python:3.12-slim AS runtime

ARG APP_VERSION=0.0.0
ARG GIT_COMMIT=unknown
ARG BUILD_DATE=unknown
LABEL org.opencontainers.image.title="ad-mri-training" \
      org.opencontainers.image.description="Alzheimer's Disease MRI classification — offline training/evaluation batch jobs." \
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

# Entry point is the CLI; the offline pipeline subcommand (train/evaluate/preprocess/...)
# is supplied at `docker run` time, so one image serves every batch job.
ENTRYPOINT ["ad-mri"]
CMD ["train"]
