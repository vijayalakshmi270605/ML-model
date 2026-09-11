"""FastAPI application factory.

Middleware order (outermost — sees a request first, a response last — to
innermost): **SecurityHeaders -> CORS -> RequestID -> AccessLog -> RateLimit
-> routing**. Deliberate: `SecurityHeadersMiddleware` (Phase 14) wraps
literally everything so even a raw, un-mapped 500 or a CORS preflight
response still carries `X-Content-Type-Options`/CSP/etc.; CORS wraps
everything below it so *every* response (including a 429 from
`RateLimitMiddleware` or a centrally-handled domain-exception response) still
carries correct CORS headers for a browser client to read; request ID is
bound before anything downstream logs; access logging wraps rate limiting so
a 429 still shows up in the access log.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ad_mri import __version__
from ad_mri.api.deps import get_inference_engine, inference_engine_was_built
from ad_mri.api.exception_handlers import register_exception_handlers
from ad_mri.api.middleware.logging_middleware import AccessLogMiddleware
from ad_mri.api.middleware.rate_limit import RateLimitMiddleware
from ad_mri.api.middleware.request_id import RequestIDMiddleware
from ad_mri.api.middleware.security_headers import SecurityHeadersMiddleware
from ad_mri.api.routers import explanations, health, meta, models, monitoring, predictions
from ad_mri.core.config import get_settings
from ad_mri.core.logging import configure_logging, get_logger

logger = get_logger(__name__)

_OPENAPI_TAGS = [
    {"name": "predictions", "description": "Run inference on an MRI study."},
    {"name": "explanations", "description": "Generate LIME/SHAP explanations for an MRI study."},
    {"name": "models", "description": "Inspect the currently-served model version."},
    {
        "name": "monitoring",
        "description": "Delayed model-performance monitoring: attach ground-truth "
        "labels to past predictions.",
    },
    {"name": "meta", "description": "Versioned liveness/readiness/version endpoints."},
    {
        "name": "health",
        "description": "Unversioned liveness/readiness/metrics probes (infra-friendly paths).",
    },
]


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    yield
    # Release the inference engine's worker thread pool on shutdown — only if
    # it was actually built (`get_inference_engine()` is lazy/cached; a
    # deployment that never configured INFERENCE_MODEL_PATH, or never
    # received a request, never built one, and there is nothing to release).
    if inference_engine_was_built():
        engine = get_inference_engine()
        if engine is not None:
            engine.shutdown()


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings)

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=(
            "Research / decision-support system for Alzheimer's Disease MRI classification. "
            "Not a certified medical device; makes no clinical diagnostic claims."
        ),
        contact={"name": "Project Maintainers"},
        license_info={"name": "Proprietary"},
        openapi_tags=_OPENAPI_TAGS,
        lifespan=_lifespan,
    )

    register_exception_handlers(app)

    # Added innermost-first — see the module docstring for the resulting order.
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=settings.rate_limit_per_minute,
        api_key_header_name=settings.api_key_header_name,
    )
    app.add_middleware(AccessLogMiddleware)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins or [],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key", "Authorization"],
        expose_headers=["X-Request-ID"],
    )
    app.add_middleware(SecurityHeadersMiddleware)

    app.include_router(health.router)
    app.include_router(meta.router, prefix=settings.api_v1_prefix)
    app.include_router(predictions.router, prefix=settings.api_v1_prefix)
    app.include_router(explanations.router, prefix=settings.api_v1_prefix)
    app.include_router(models.router, prefix=settings.api_v1_prefix)
    app.include_router(monitoring.router, prefix=settings.api_v1_prefix)

    logger.info("app_created", app_env=settings.app_env, version=__version__)
    return app


app = create_app()
