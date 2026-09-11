"""Centralized application configuration.

All configuration is sourced from environment variables (optionally loaded from a
local `.env` file — see `.env.example`) and validated at process start via
pydantic-settings. Nothing here contains a hardcoded credential, and every
filesystem path is an overridable default, never a literal used elsewhere in the
codebase. See docs/TECHNICAL_SPECIFICATION.md §14.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from ad_mri.core.constants import API_KEY_HEADER, ApiScope, Environment
from ad_mri.core.exceptions import ConfigurationError

# A sentinel checked against below (`_validate_production_secrets` raises if
# APP_ENV=prod still has this value) - never a real credential.
_INSECURE_DEFAULT_SECRET = "dev-secret-key-change-me"  # nosec B105


class Settings(BaseSettings):
    """Application-wide settings, resolved once per process."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Application ---
    app_name: str = "ad-mri"
    app_env: Environment = Environment.DEV
    debug: bool = False
    api_v1_prefix: str = "/api/v1"
    # NoDecode: pydantic-settings would otherwise try to JSON-decode env values for
    # complex (list) fields before validation ever runs, rejecting a plain
    # comma-separated string. NoDecode passes the raw string through to
    # `_split_csv` below instead.
    allowed_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)

    # --- Server ---
    # Binding to all interfaces is deliberate: this process almost always
    # runs inside a container (docker/api.Dockerfile), where binding to a
    # single interface would make it unreachable from outside the
    # container's network namespace entirely; the actual exposure boundary
    # is the container/orchestrator port mapping and network policy,
    # documented in docs/deployment.md, not this bind address.
    api_host: str = "0.0.0.0"  # nosec B104
    api_port: int = 8000
    api_reload: bool = True

    # --- Logging ---
    log_level: str = "INFO"
    log_format: str = "console"  # "console" | "json"

    # --- Security ---
    secret_key: SecretStr = SecretStr(_INSECURE_DEFAULT_SECRET)
    api_key_required: bool = False
    #: The legacy single key. Honored by `api.deps.require_scope` as a
    #: superuser (every `ApiScope`) when `api_keys` (below) isn't set -
    #: kept for backward compatibility with every deployment/test from
    #: before Phase 14's multi-key authorization architecture existed.
    api_key: SecretStr | None = None
    #: Enforcement (`api.deps.require_api_key`/`require_scope`) honors this
    #: if changed; the OpenAPI security scheme (`api.security`) always
    #: advertises `core.constants.API_KEY_HEADER` (FastAPI's `APIKeyHeader`
    #: needs a static name at declare time) — keep this at the default
    #: unless you also accept the OpenAPI docs describing a different
    #: header name.
    api_key_header_name: str = API_KEY_HEADER
    #: Authorization architecture (Phase 14, docs/security.md): multiple API
    #: keys, each with its own set of `core.constants.ApiScope` values -
    #: `{"key-for-the-ingestion-pipeline": ["predict"], "key-for-clinicians":
    #: ["predict", "label"]}`. A plain JSON object env var
    #: (`API_KEYS='{"...": ["predict"]}'`), same convention as every other
    #: secret in this project (env-var based, never committed - see
    #: docs/deployment.md §6). `None` (the default) means "not configured" -
    #: falls back entirely to the legacy single `api_key` above, which then
    #: acts as a superuser holding every scope.
    api_keys: dict[str, list[ApiScope]] | None = None

    # --- Database (PostgreSQL) ---
    database_url: str | None = None
    postgres_user: str = "postgres"
    postgres_password: SecretStr = SecretStr("postgres")
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "ad_mri"
    #: Connection-pool sizing (`db.session.build_engine`) - QueuePool-specific,
    #: applied only for non-SQLite URLs; see that function's docstring.
    db_pool_size: int = 5
    db_max_overflow: int = 10
    db_pool_timeout: float = 30.0
    #: Recycle a pooled connection after this many seconds, guarding against a
    #: database or load balancer silently dropping long-idle connections.
    db_pool_recycle: int = 1800
    db_connect_timeout_seconds: int = 5
    #: Best-effort audit-trail persistence (`api.audit`) - an unreachable
    #: audit database never fails a `/predictions`/`/explanations` request
    #: (mirrors how `GET /ready` treats the database independently of the
    #: inference engine); this only lets an operator turn writes off entirely.
    audit_enabled: bool = True

    # --- MLflow (wired in the training/MLOps phase) ---
    mlflow_tracking_uri: str | None = None
    mlflow_model_stage: str = "Production"

    # --- Inference engine (Phase 9) ---
    # Which checkpoint the standalone inference service / API serves. `None`
    # (the default) means "not configured yet" — the /predict endpoint then
    # responds 503 rather than crashing at import time; there is no real
    # model-registry integration yet (see model_registry/client.py), so this
    # points directly at a checkpoint path, the same versioned artifact
    # every other phase (evaluate/explain) already loads directly.
    inference_model_path: Path | None = None
    inference_preprocessing_config_path: Path = Path("configs/preprocessing/default.yaml")
    inference_calibration_config_path: Path | None = None
    inference_device: str = "auto"  # "auto" | "cpu" | "cuda"
    inference_timeout_seconds: float = 30.0
    #: Manifest to draw the SHAP reference/background set from, if a caller
    #: requests a SHAP explanation via ?explain=true&methods=shap. Optional —
    #: LIME explanations need no background set.
    inference_explanation_background_manifest: Path | None = None
    inference_explanation_output_dir: Path = Path("artifacts/explanations")

    # --- Redis / Celery (async explainability jobs) ---
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str | None = None
    celery_result_backend: str | None = None

    # --- Filesystem (overridable defaults, never hardcoded elsewhere) ---
    config_dir: Path = Path("configs")
    data_dir: Path = Path("data")

    # --- Rate limiting --- (api.middleware.rate_limit.RateLimitMiddleware; <= 0 disables it)
    rate_limit_per_minute: int = 60

    # --- Upload limits ---
    max_upload_size_mb: int = Field(default=200, gt=0)

    # --- LLM narrative summaries (optional, off by default) ---
    #: `explainability.llm_summary.generate_narrative_summary` turns a LIME/SHAP
    #: explanation's numeric output into a plain-English summary via the
    #: OpenAI API - entirely optional (`None`, the default, means "not
    #: configured": the feature no-ops, `POST /explanations` behaves exactly as
    #: it did before this existed). Only aggregate, already-computed numeric
    #: values ever reach this API - never raw pixel data, a file path, or a
    #: patient identifier; see `llm_summary`'s module docstring.
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o-mini"
    openai_timeout_seconds: float = 15.0

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @field_validator(
        "inference_model_path",
        "inference_calibration_config_path",
        "inference_explanation_background_manifest",
        mode="before",
    )
    @classmethod
    def _blank_optional_path(cls, value: object) -> object:
        """A blank env var (`INFERENCE_MODEL_PATH=`) must mean "not set" (`None`),
        not `Path("")` — which coerces to `Path(".")`, a *truthy* path that
        would silently pass every `if settings.inference_model_path:` check
        this project relies on to detect "not configured"."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("api_key", "openai_api_key", mode="before")
    @classmethod
    def _blank_api_key(cls, value: object) -> object:
        """A blank `API_KEY=`/`OPENAI_API_KEY=` env var (the `.env.example`
        default, so this is the common case, not an edge case) would
        otherwise coerce to `SecretStr("")` rather than `None`. For `api_key`,
        every real check in this codebase (`api.deps.require_api_key`/
        `require_scope`, `_validate_production_secrets` below) already treats
        an empty `SecretStr` as falsy, so this was never an authentication
        bypass - but it left `settings.api_key is None` false for a
        deployment that never configured a key, which is surprising and
        inconsistent with `api_keys`/`inference_model_path`'s identical blank-
        string convention above. For `openai_api_key`,
        `explainability.llm_summary` checks `is None` directly to decide
        whether the feature is configured at all, so this normalization is
        load-bearing there, not just cosmetic. Normalizing to `None` here
        removes the inconsistency rather than relying on every caller
        happening to use truthiness instead of an identity check."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        """Allow ALLOWED_ORIGINS as a simple comma-separated env var."""
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator("api_keys", mode="before")
    @classmethod
    def _blank_api_keys(cls, value: object) -> object:
        """A blank `API_KEYS=` env var must mean "not configured" (`None`) -
        the same blank-string footgun `_blank_optional_path` guards against
        for `Path` fields: pydantic-settings tries to JSON-decode a non-empty
        string for this complex-typed field, and an empty string is not
        valid JSON for a dict, which would otherwise turn a deployment that
        simply never set this into a hard startup `ValidationError` instead
        of the intended "fall back to the legacy single API_KEY" behavior."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _validate_production_secrets(self) -> Settings:
        if self.app_env == Environment.PROD:
            if self.secret_key.get_secret_value() == _INSECURE_DEFAULT_SECRET:
                raise ConfigurationError(
                    "SECRET_KEY must be set to a non-default value when APP_ENV=prod."
                )
            if self.api_key_required and not self.api_key and not self.api_keys:
                raise ConfigurationError(
                    "API_KEY or API_KEYS must be set when API_KEY_REQUIRED=true in prod."
                )
        if self.api_keys is not None:
            empty = [key for key, scopes in self.api_keys.items() if not scopes]
            if empty:
                raise ConfigurationError(
                    f"API_KEYS entries must each grant at least one scope; "
                    f"{len(empty)} entry(ies) grant none - a key with no scopes can never "
                    "authorize anything, which almost certainly means a typo, not intent."
                )
        return self

    def get_database_url(self) -> str:
        """Return the configured DATABASE_URL, or assemble one from components."""
        if self.database_url:
            return self.database_url
        password = self.postgres_password.get_secret_value()
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    def get_celery_broker_url(self) -> str:
        return self.celery_broker_url or self.redis_url

    def get_celery_result_backend(self) -> str:
        return self.celery_result_backend or self.redis_url


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide cached Settings instance."""
    return Settings()
