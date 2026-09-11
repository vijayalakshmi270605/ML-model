"""Centralized exception handling: every domain exception this project
defines (`core.exceptions`) is mapped to an HTTP status code and a consistent
`ErrorResponse` body **once**, here — route handlers never write their own
try/except around a call into the inference/explainability layer; they let
the exception propagate and FastAPI dispatches it to the matching handler
below (registered via `register_exception_handlers`).

**Never logs `str(exc)`.** A domain exception's own message can legitimately
include a filesystem path or other detail useful to whoever raised it
(a CLI user with their own file, say) — exactly the kind of detail that must
never reach a server log line or an HTTP response body here (see
`inference.predictor`'s module docstring). Every handler below logs only the
exception's *type* and the current `request_id`, and returns only a generic,
category-appropriate message — never the exception's own text.
"""

from __future__ import annotations

import asyncio

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from ad_mri.api import audit
from ad_mri.api.middleware.security_headers import apply_security_headers
from ad_mri.api.schemas.common import ErrorResponse
from ad_mri.core.exceptions import (
    AppError,
    ConfigurationError,
    DataValidationError,
    ExplainabilityError,
    InferenceError,
    ModelNotFoundError,
    PreprocessingError,
    TrainingError,
)
from ad_mri.core.logging import get_logger

logger = get_logger(__name__)


def _current_request_id() -> str | None:
    import structlog

    return structlog.contextvars.get_contextvars().get("request_id")


async def _record_event(
    request: Request, exc: Exception, *, event_type: str, severity: str
) -> None:
    """Requirement 12 (audit events): every mapped domain-exception response
    also becomes a `system_events` row - best-effort (`api.audit.record_event`
    never raises), so an unreachable audit database never turns an already-
    correct 4xx/5xx response into a 500. `detail` only ever carries the
    exception's *type* and the request path (a fixed, non-identifying set of
    endpoint routes in this API) - never `str(exc)`, for the same reason
    every log call in this module already avoids it."""
    await asyncio.to_thread(
        audit.record_event,
        event_type=event_type,
        severity=severity,
        request_id=_current_request_id(),
        detail={"path": request.url.path, "error_type": type(exc).__name__},
    )


def _respond(status_code: int, detail: str, error_code: str) -> JSONResponse:
    body = ErrorResponse(detail=detail, error_code=error_code)
    return JSONResponse(status_code=status_code, content=body.model_dump())


async def _handle_data_validation_error(request: Request, exc: Exception) -> JSONResponse:
    logger.warning(
        "request_rejected_invalid_input",
        request_id=_current_request_id(),
        path=request.url.path,
        error_type=type(exc).__name__,
    )
    await _record_event(request, exc, event_type="prediction_validation_failed", severity="warning")
    return _respond(
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        "The request failed validation and cannot be processed.",
        "VALIDATION_FAILED",
    )


async def _handle_preprocessing_error(request: Request, exc: Exception) -> JSONResponse:
    logger.warning(
        "request_rejected_preprocessing_failed",
        request_id=_current_request_id(),
        path=request.url.path,
        error_type=type(exc).__name__,
    )
    await _record_event(request, exc, event_type="preprocessing_failed", severity="warning")
    return _respond(
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        "The uploaded file could not be preprocessed and cannot be classified.",
        "PREPROCESSING_FAILED",
    )


async def _handle_model_not_found_error(request: Request, exc: Exception) -> JSONResponse:
    logger.warning(
        "request_rejected_model_not_found",
        request_id=_current_request_id(),
        path=request.url.path,
        error_type=type(exc).__name__,
    )
    await _record_event(request, exc, event_type="model_not_found", severity="warning")
    return _respond(
        status.HTTP_404_NOT_FOUND, "The requested model/version was not found.", "MODEL_NOT_FOUND"
    )


async def _handle_inference_error(request: Request, exc: Exception) -> JSONResponse:
    timed_out = "timed out" in str(exc).lower()  # our own message shape, not the raw text returned
    logger.warning(
        "request_rejected_inference_failed",
        request_id=_current_request_id(),
        path=request.url.path,
        error_type=type(exc).__name__,
        timed_out=timed_out,
    )
    await _record_event(
        request,
        exc,
        event_type="inference_timed_out" if timed_out else "inference_failed",
        severity="warning" if timed_out else "error",
    )
    if timed_out:
        return _respond(
            status.HTTP_504_GATEWAY_TIMEOUT, "Inference timed out.", "INFERENCE_TIMEOUT"
        )
    return _respond(status.HTTP_500_INTERNAL_SERVER_ERROR, "Inference failed.", "INFERENCE_FAILED")


async def _handle_explainability_error(request: Request, exc: Exception) -> JSONResponse:
    logger.warning(
        "request_rejected_explanation_failed",
        request_id=_current_request_id(),
        path=request.url.path,
        error_type=type(exc).__name__,
    )
    await _record_event(request, exc, event_type="explanation_failed", severity="error")
    return _respond(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "Explanation generation failed.",
        "EXPLANATION_FAILED",
    )


async def _handle_configuration_error(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "request_rejected_configuration_error",
        request_id=_current_request_id(),
        path=request.url.path,
        error_type=type(exc).__name__,
    )
    await _record_event(request, exc, event_type="configuration_error", severity="error")
    return _respond(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "Server-side model/configuration error.",
        "CONFIGURATION_ERROR",
    )


async def _handle_training_error(
    request: Request, exc: Exception
) -> JSONResponse:  # pragma: no cover
    # Training is never triggered from the API — present only for exhaustive
    # AppError-subclass coverage, should this hierarchy ever grow a caller.
    logger.error(
        "request_rejected_training_error",
        request_id=_current_request_id(),
        path=request.url.path,
        error_type=type(exc).__name__,
    )
    await _record_event(request, exc, event_type="training_error", severity="error")
    return _respond(status.HTTP_500_INTERNAL_SERVER_ERROR, "Internal error.", "TRAINING_ERROR")


async def _handle_generic_app_error(request: Request, exc: Exception) -> JSONResponse:
    """Fallback for any `AppError` subclass without a specific handler above —
    keeps a future new exception type from ever leaking a raw traceback."""
    logger.error(
        "request_rejected_unhandled_app_error",
        request_id=_current_request_id(),
        path=request.url.path,
        error_type=type(exc).__name__,
    )
    await _record_event(request, exc, event_type="unhandled_app_error", severity="error")
    return _respond(status.HTTP_500_INTERNAL_SERVER_ERROR, "Internal error.", "INTERNAL_ERROR")


async def _handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for anything that is *not* one of this project's own
    `AppError` subclasses (a bug, a third-party library's own exception type,
    ...) — Phase 14's security review: before this handler existed, such an
    exception reached Starlette's `ServerErrorMiddleware` with nothing
    registered for it, producing an unstructured, unlogged, un-audited
    response outside this project's own control.

    Registering a handler for bare `Exception` (as this one is, in
    `register_exception_handlers` below) does **not** put it on the same
    footing as every handler above, though: per `Starlette
    .build_middleware_stack`, a handler for `Exception` (or the status code
    `500`) becomes `ServerErrorMiddleware`'s own `handler` — and
    `ServerErrorMiddleware` is the true outermost layer, added *before*
    every `app.add_middleware(...)` call in `api.main`, specifically so a
    broken middleware can never suppress it. That means this function's
    response never passes back through `SecurityHeadersMiddleware`
    (or `CORSMiddleware`, or `AccessLogMiddleware`) the way every other
    handler's does — confirmed by a real test
    (`tests/unit/test_api_security_headers.py`), not assumed. This function
    compensates for the one consequence that matters most (missing security
    headers) directly, via `apply_security_headers` — see
    `api.middleware.security_headers`'s module docstring for the full
    explanation, including the one gap even this doesn't close (CORS headers
    still won't be present on this specific response shape).
    """
    logger.error(
        "request_rejected_unexpected_error",
        request_id=_current_request_id(),
        path=request.url.path,
        error_type=type(exc).__name__,
    )
    await _record_event(request, exc, event_type="unexpected_error", severity="critical")
    response = _respond(status.HTTP_500_INTERNAL_SERVER_ERROR, "Internal error.", "INTERNAL_ERROR")
    return apply_security_headers(response, path=request.url.path)


def register_exception_handlers(app: FastAPI) -> None:
    """Register every mapping. Order matters only in that FastAPI picks the
    most specific *registered* handler for an exception's MRO — each of our
    concrete `AppError` subclasses is registered explicitly, and the base
    `AppError` handler only ever catches a subclass that has no dedicated one.
    `Exception` is registered last and separately - see
    `_handle_unexpected_error`'s docstring for why it is not "just another
    entry" in this list.
    """
    app.add_exception_handler(DataValidationError, _handle_data_validation_error)
    app.add_exception_handler(PreprocessingError, _handle_preprocessing_error)
    app.add_exception_handler(ModelNotFoundError, _handle_model_not_found_error)
    app.add_exception_handler(InferenceError, _handle_inference_error)
    app.add_exception_handler(ExplainabilityError, _handle_explainability_error)
    app.add_exception_handler(ConfigurationError, _handle_configuration_error)
    app.add_exception_handler(TrainingError, _handle_training_error)
    app.add_exception_handler(AppError, _handle_generic_app_error)
    app.add_exception_handler(Exception, _handle_unexpected_error)
