"""Best-effort persistence of the prediction/explanation audit trail.

Every function here is safe to call from a request handler: on any failure
(an unreachable database, a constraint violation) it logs `error_type` only -
never the exception's message, which could echo back request-derived text -
and returns without raising. An unreachable audit database must never fail a
client's `/predictions` or `/explanations` call, only degrade observability;
this mirrors `monitoring.health_checks` (database reachability is tracked,
never load-bearing for those two endpoints) and is exactly the same
independence `GET /ready` already models between the database and the
inference engine.

`db.audit_service.AuditService` itself stays loud (raises on failure) so it
remains honestly unit-testable; the swallow-and-log boundary lives here, one
layer up, deliberately - the same "loud service, best-effort caller" split
`inference.predictor.generate_explanation` / `_try_generate_explanation`
already uses.

Each `record_*` function opens and closes its own short-lived `Session`
rather than reusing the request's `api.deps.get_db` session, so a slow or
failing audit write can never hold open (or roll back) the session a request
handler's *other* work depends on - and so it is safe to run inside
`asyncio.to_thread` after the HTTP response body has already been built.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ad_mri.core.config import get_settings
from ad_mri.core.logging import get_logger
from ad_mri.db.audit_service import AuditService
from ad_mri.db.session import get_sessionmaker
from ad_mri.storage.object_storage import local_file_uri

if TYPE_CHECKING:
    from ad_mri.explainability.service import ExplanationBundle
    from ad_mri.inference.predictor import ModelVersionInfo, PredictionResult

logger = get_logger(__name__)


def _model_version_kwargs(info: ModelVersionInfo) -> dict[str, str]:
    return {
        "architecture": info.architecture,
        "model_version": info.model_version,
        "model_config_hash": info.model_config_hash,
        "preprocessing_version": info.preprocessing_version,
        "preprocessing_config_hash": info.preprocessing_config_hash,
        "checkpoint_uri": local_file_uri(Path(info.checkpoint_path)),
        "device": info.device,
    }


def record_prediction(
    *,
    result: PredictionResult,
    model_info: ModelVersionInfo,
    image_ref: str,
    content_type: str | None,
    api_key: str | None,
) -> None:
    """Fire-and-forget: persists the request/result/model-version rows for one
    completed `POST /api/v1/predictions` call. No-ops when
    `Settings.audit_enabled` is False."""
    settings = get_settings()
    if not settings.audit_enabled:
        return
    session = get_sessionmaker()()
    try:
        AuditService(session).record_prediction(
            request_id=result.request_id,
            image_ref=image_ref,
            content_type=content_type,
            api_key=api_key,
            predicted_class=result.prediction,
            confidence=result.confidence,
            probabilities=result.probabilities,
            processing_time_ms=result.processing_time_ms,
            explanation_available=result.explanation_available,
            **_model_version_kwargs(model_info),
        )
    except Exception as exc:  # noqa: BLE001 - an audit-write failure must never fail the request
        logger.warning("audit_record_prediction_failed", error_type=type(exc).__name__)
    finally:
        session.close()


def record_explanation(
    *,
    bundle: ExplanationBundle,
    model_info: ModelVersionInfo,
    methods: str,
    target_class: str | None,
) -> None:
    """Fire-and-forget: persists the explanation-metadata row for one
    completed `POST /api/v1/explanations` call."""
    settings = get_settings()
    if not settings.audit_enabled:
        return
    session = get_sessionmaker()()
    try:
        manifest_path = bundle.artifacts.get("manifest")
        artifact_uri = (
            local_file_uri(manifest_path) if manifest_path else local_file_uri(bundle.output_dir)
        )
        # `bundle.input_id` is a standalone id for this `/explanations` call -
        # no `PredictionRequestRecord` exists for it (only `/predictions`
        # creates one), so `request_id` must stay unset rather than pointing
        # at a row that was never written (a real PostgreSQL foreign key
        # would reject that insert outright).
        AuditService(session).record_explanation(
            explanation_id=bundle.input_id,
            request_id=None,
            methods=methods,
            target_class=target_class,
            artifact_uri=artifact_uri,
            **_model_version_kwargs(model_info),
        )
    except Exception as exc:  # noqa: BLE001 - an audit-write failure must never fail the request
        logger.warning("audit_record_explanation_failed", error_type=type(exc).__name__)
    finally:
        session.close()


def record_model_version_seen(model_info: ModelVersionInfo) -> None:
    """Fire-and-forget: registers the currently-served model version at
    engine startup (`api.deps._build_inference_engine`), independent of
    whether any prediction has run yet - so `model_versions` reflects what a
    deployment *can* serve, not only what it *has* served."""
    settings = get_settings()
    if not settings.audit_enabled:
        return
    session = get_sessionmaker()()
    try:
        AuditService(session).record_model_version(
            stage="Production", **_model_version_kwargs(model_info)
        )
    except Exception as exc:  # noqa: BLE001 - must never block/crash engine startup
        logger.warning("audit_record_model_version_failed", error_type=type(exc).__name__)
    finally:
        session.close()


def record_event(
    *,
    event_type: str,
    severity: str = "info",
    request_id: str | None = None,
    detail: dict[str, object] | None = None,
) -> None:
    """Fire-and-forget: a generic audit event (requirement 12) - engine
    failures, mapped error responses, auth/rate-limit rejections. `detail`
    must already be sanitized by the caller (see
    `db.models.SystemEventRecord`)."""
    settings = get_settings()
    if not settings.audit_enabled:
        return
    session = get_sessionmaker()()
    try:
        AuditService(session).record_event(
            event_type=event_type,
            severity=severity,
            request_id=request_id,
            detail=detail,
        )
    except Exception as exc:  # noqa: BLE001 - an audit-write failure must never fail the request
        logger.warning("audit_record_event_failed", error_type=type(exc).__name__)
    finally:
        session.close()
