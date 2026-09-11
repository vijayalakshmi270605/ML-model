"""Shared enums and constants.

Note: the Alzheimer's disease class taxonomy is intentionally NOT defined here.
It is dataset-dependent (e.g. CN/MCI/AD vs. Non-/Very-Mild/Mild/Moderate-Demented)
and will be defined per-dataset under `configs/data/*.yaml` in the data-pipeline
phase — see docs/TECHNICAL_SPECIFICATION.md §17.
"""

from enum import StrEnum


class Environment(StrEnum):
    """Deployment environment — drives config layering, never business logic directly."""

    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"
    TEST = "test"


class ModelStage(StrEnum):
    """Mirrors the MLflow Model Registry lifecycle stages."""

    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


class ExplainabilityMethod(StrEnum):
    """Supported explanation methods — see docs/TECHNICAL_SPECIFICATION.md §7."""

    LIME = "lime"
    SHAP = "shap"


class ApiScope(StrEnum):
    """Authorization scopes (Phase 14, docs/security.md "Authorization
    architecture") — what an authenticated caller is allowed to *do*, as
    opposed to API-key auth (`api.deps.require_api_key`), which only answers
    *who* they are. See `api.deps.require_scope`.

    Deliberately small and coarse: two scopes map cleanly onto this API's
    two kinds of write — running the model (`PREDICT`, both `/predictions`
    and `/explanations`, since both trigger real inference) vs. attaching a
    clinical ground-truth label to a past prediction (`LABEL`) — a distinct,
    arguably more sensitive capability (it accepts a real diagnosis value,
    however constrained) that a purely automated ingestion/inference client
    has no reason to hold.
    """

    PREDICT = "predict"
    LABEL = "label"


REQUEST_ID_HEADER = "X-Request-ID"
#: Default API-key header name — also `Settings.api_key_header_name`'s default.
#: FastAPI's `APIKeyHeader` security scheme (used for OpenAPI documentation,
#: see `api.security`) must be declared with a static name, so it always uses
#: this constant; actual per-request *enforcement* still honors a
#: reconfigured `API_KEY_HEADER_NAME` (see `api.deps.require_api_key`).
API_KEY_HEADER = "X-API-Key"
