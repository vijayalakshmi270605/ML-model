"""Shared MLflow run-management helper: consistent fast-fail-on-unreachable-
server behavior and graceful degradation, used by `Trainer`, cross-validation,
and hyperparameter search alike — one implementation, not three copies.

`mlflow` is a training-phase dependency (requirements/train.txt), imported
lazily so this module stays importable without it.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any

from ad_mri.core.logging import get_logger

logger = get_logger(__name__)


def start_mlflow_run(
    *,
    enabled: bool,
    tracking_uri: str | None,
    experiment_name: str,
    run_name: str,
    nested: bool = False,
) -> tuple[Any, Any]:
    """Start (or no-op) an MLflow run.

    Returns `(mlflow_module, run_context_manager)` — `mlflow_module` is `None`
    when `enabled=False` or MLflow is unreachable/misconfigured, in which case
    `run_context_manager` is a no-op `contextlib.nullcontext()`. MLflow is
    supplementary tracking, not load-bearing: callers should treat a `None`
    module as "skip MLflow logging", never as a reason to abort.
    """
    if not enabled:
        return None, contextlib.nullcontext()

    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    # MLflow >=3.x refuses a plain local file-store tracking URI unless
    # explicitly opted in — restores the pre-3.x zero-config local-tracking
    # behavior most users expect when no remote server/DB backend is set.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    # MLflow's own defaults (120s timeout x 7 retries with backoff) mean an
    # unreachable HTTP tracking server can block a run for several minutes
    # before the try/except below ever gets a chance to degrade gracefully.
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")

    try:
        import mlflow

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        return mlflow, mlflow.start_run(run_name=run_name, nested=nested)
    except Exception as exc:
        logger.warning("mlflow_unavailable_continuing_without_tracking", error=str(exc))
        return None, contextlib.nullcontext()
