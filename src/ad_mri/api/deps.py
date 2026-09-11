"""FastAPI dependency-injection providers: DB sessions, API-key auth, and the
process-wide cached inference engine.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from functools import lru_cache
from typing import TYPE_CHECKING

from fastapi import Depends, HTTPException, Request, Security, status
from sqlalchemy.orm import Session

from ad_mri.api.security import api_key_scheme
from ad_mri.core.config import Settings, get_settings
from ad_mri.core.constants import ApiScope
from ad_mri.core.logging import get_logger
from ad_mri.db.session import get_db_session as _get_db_session

if TYPE_CHECKING:
    from ad_mri.inference.predictor import InferenceEngine

logger = get_logger(__name__)


def get_db() -> Generator[Session, None, None]:
    yield from _get_db_session()


def require_api_key(
    request: Request,
    _declared_key: str | None = Security(api_key_scheme),
    settings: Settings = Depends(get_settings),
) -> None:
    """No-op when API_KEY_REQUIRED=false (the local-dev default); enforced
    otherwise. `_declared_key` (via `Security(api_key_scheme)`) exists only so
    this dependency's OpenAPI entry gets a proper security requirement — the
    actual header read below honors a reconfigured `API_KEY_HEADER_NAME`,
    which `APIKeyHeader`'s static declaration cannot (see `api.security`).
    """
    if not settings.api_key_required:
        return
    presented = request.headers.get(settings.api_key_header_name)
    expected = settings.api_key.get_secret_value() if settings.api_key else None
    if not expected or presented != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key."
        )


def require_scope(scope: ApiScope) -> Callable[..., None]:
    """Authorization (Phase 14, docs/security.md), layered on top of
    authentication (`require_api_key`'s underlying check, re-done here
    rather than composed, since answering "is this key allowed to do X"
    needs to know *which* key was presented, not just that one was):

    - `API_KEY_REQUIRED=false` (dev default): no-op, same as
      `require_api_key` - there is no identity to check a scope against.
    - `Settings.api_keys` configured (multi-key mode): the presented key
      must be one of its entries, and `scope` must be in that entry's list -
      an unknown key is `401` (not authenticated at all); a known key
      missing this scope is `403` (authenticated, not authorized - a real,
      meaningful distinction a caller's own error-handling can act on).
    - Otherwise (legacy single `api_key`, Phase 10-13 behavior): the
      presented key matching `settings.api_key` is sufficient - the legacy
      key is a superuser holding every scope, preserving every prior
      phase's test/deployment behavior unchanged.

    Returns a dependency *function*, not a value - `require_scope` itself is
    called at router-definition time (`dependencies=[Depends(require_scope(
    ApiScope.PREDICT))]`), the function it returns is what FastAPI actually
    invokes per-request.
    """

    def _dependency(
        request: Request,
        _declared_key: str | None = Security(api_key_scheme),
        settings: Settings = Depends(get_settings),
    ) -> None:
        if not settings.api_key_required:
            return
        presented = request.headers.get(settings.api_key_header_name)

        if settings.api_keys is not None:
            granted = settings.api_keys.get(presented) if presented else None
            if granted is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or missing API key.",
                )
            if scope not in granted:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"This API key does not have the required scope: {scope.value!r}.",
                )
            return

        expected = settings.api_key.get_secret_value() if settings.api_key else None
        if not expected or presented != expected:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key."
            )

    return _dependency


@lru_cache
def _build_inference_engine() -> InferenceEngine | None:
    """Constructed at most once per process (`lru_cache`, mirroring
    `core.config.get_settings`) — "cache model in memory" at the API-process
    level. Returns `None` (never raises) when `INFERENCE_MODEL_PATH` is unset
    or the checkpoint/preprocessing pair fails to load — callers (the
    `/predictions`/`/explanations`/`/models` routers, `/ready`) treat `None`
    as "inference not available" (HTTP 503), not as a startup crash; a lean
    deployment with no model configured yet must still be able to serve
    `/health`/`/healthz`.
    """
    settings = get_settings()
    if settings.inference_model_path is None:
        logger.warning("inference_engine_not_configured", reason="INFERENCE_MODEL_PATH unset")
        return None

    try:
        from ad_mri.inference.calibration import load_calibration_config
        from ad_mri.inference.predictor import InferenceEngine

        calibration_config = (
            load_calibration_config(settings.inference_calibration_config_path)
            if settings.inference_calibration_config_path
            else None
        )
        engine = InferenceEngine(
            checkpoint_path=settings.inference_model_path,
            preprocessing_config_path=settings.inference_preprocessing_config_path,
            calibration_config=calibration_config,
            device_choice=settings.inference_device,
            timeout_seconds=settings.inference_timeout_seconds,
            explanation_background_manifest_path=settings.inference_explanation_background_manifest,
            explanation_output_dir=settings.inference_explanation_output_dir,
        )
        # Best-effort (requirement 6, "model version tracking"): register
        # this version in the audit database as soon as it is known to be
        # loadable, not only after its first prediction. Never fails engine
        # construction - see `api.audit`'s module docstring.
        from ad_mri.api.audit import record_model_version_seen

        record_model_version_seen(engine.model_version_info())
        return engine
    except Exception as exc:  # noqa: BLE001 - a bad model config must not crash the API process
        logger.error("inference_engine_construction_failed", error_type=type(exc).__name__)
        return None


def get_inference_engine() -> InferenceEngine | None:
    return _build_inference_engine()


def inference_engine_was_built() -> bool:
    """True if `get_inference_engine()` has already run at least once this
    process (whether or not it produced an engine) — lets shutdown code avoid
    constructing one for the first time just to immediately release it."""
    return _build_inference_engine.cache_info().currsize > 0


def clear_inference_engine_cache() -> None:
    """Test-only: forces the next `get_inference_engine()` call to rebuild.
    Mirrors `get_settings.cache_clear()` in `tests/conftest.py`."""
    _build_inference_engine.cache_clear()
