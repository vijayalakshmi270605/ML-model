"""Reusable dependency health checks used by readiness probes and monitoring jobs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import text
from sqlalchemy.engine import Engine

from ad_mri.core.logging import get_logger

if TYPE_CHECKING:
    from ad_mri.inference.predictor import InferenceEngine

logger = get_logger(__name__)


def check_database(engine: Engine) -> bool:
    """Return True if a trivial query succeeds against `engine`, False otherwise.

    Never raises — a failed health check is a signal, not an application error.
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:  # noqa: BLE001 - health checks must never raise
        logger.warning("database_health_check_failed", error=str(exc))
        return False


def check_inference_engine(engine: InferenceEngine | None) -> bool:
    """Return True if `engine` is present and its model can run a real
    forward pass, False otherwise (including "not configured": `engine is
    None`, e.g. `INFERENCE_MODEL_PATH` unset). Never raises.
    """
    if engine is None:
        return False
    try:
        status = engine.health_check(run_dummy_forward_pass=True)
        return bool(status.get("model_loaded") and status.get("forward_pass_ok"))
    except Exception as exc:  # noqa: BLE001 - health checks must never raise
        logger.warning("inference_engine_health_check_failed", error_type=type(exc).__name__)
        return False
