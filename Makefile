.PHONY: help install install-dev install-train pre-commit-install lint format type-check \
	security test test-unit test-integration test-cov run \
	docker-build docker-up docker-down docker-migrate docker-logs docker-observability-up \
	docker-prod-up docker-prod-down clean

PYTHON ?= python

help:
	@echo "Available targets:"
	@echo "  install            Install runtime dependencies (editable)"
	@echo "  install-dev        Install dev dependencies (test/lint/format/type-check/security)"
	@echo "  install-train      Install training/MLOps dependencies"
	@echo "  pre-commit-install Install git pre-commit hooks"
	@echo "  lint               Run ruff"
	@echo "  format             Run black"
	@echo "  type-check         Run mypy"
	@echo "  security           Run bandit (SAST) + pip-audit (dependency vulnerabilities)"
	@echo "  test               Run the full test suite"
	@echo "  test-unit          Run unit/api/data_validation/explainability tests only"
	@echo "  test-integration   Run integration tests only"
	@echo "  test-cov           Run the test suite with a coverage report"
	@echo "  run                Run the API with auto-reload (dev)"
	@echo "  docker-build       Build all local-dev Docker images (api, worker, mlflow)"
	@echo "  docker-up          Start the local dev stack (api, worker, mlflow, db, redis)"
	@echo "  docker-migrate     Run Alembic migrations against the local dev stack"
	@echo "  docker-logs        Tail logs from the local dev stack"
	@echo "  docker-down        Stop the local dev stack"
	@echo "  docker-observability-up  Start Prometheus + Grafana against the local dev stack"
	@echo "  docker-prod-up     Start the production stack (requires IMAGE_REGISTRY/IMAGE_TAG/SECRETS_ENV_FILE)"
	@echo "  docker-prod-down   Stop the production stack"
	@echo "  clean              Remove caches and build artifacts"

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

install-train:
	$(PYTHON) -m pip install -e ".[train]"

pre-commit-install:
	pre-commit install

lint:
	ruff check src tests
	black --check src tests

format:
	black src tests

type-check:
	mypy src

security:
	bandit -c pyproject.toml -r src
	pip-audit --skip-editable

test:
	pytest

test-unit:
	pytest tests/unit tests/api tests/data_validation tests/explainability

test-integration:
	pytest tests/integration

test-cov:
	pytest --cov=ad_mri --cov-report=term-missing

run:
	ad-mri serve

docker-build:
	docker compose -f docker/docker-compose.yml build

docker-up:
	docker compose -f docker/docker-compose.yml up -d --build

docker-migrate:
	docker compose -f docker/docker-compose.yml --profile tools run --rm migrate

docker-logs:
	docker compose -f docker/docker-compose.yml logs -f

docker-down:
	docker compose -f docker/docker-compose.yml down

# Grafana at http://localhost:3000 (anonymous viewer access in local dev),
# Prometheus at http://localhost:9090 — see docs/monitoring.md.
docker-observability-up:
	docker compose -f docker/docker-compose.yml --profile observability up -d

# Requires IMAGE_REGISTRY, IMAGE_TAG, and (usually) SECRETS_ENV_FILE — see
# docs/deployment.md. Never builds; only ever pulls and runs a versioned image.
docker-prod-up:
	docker compose -f docker/docker-compose.prod.yml pull
	docker compose -f docker/docker-compose.prod.yml --profile tools run --rm migrate
	docker compose -f docker/docker-compose.prod.yml up -d

docker-prod-down:
	docker compose -f docker/docker-compose.prod.yml down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage build dist
