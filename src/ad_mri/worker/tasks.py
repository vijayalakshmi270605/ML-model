"""Async worker tasks."""

from __future__ import annotations

from ad_mri.worker.celery_app import celery_app


@celery_app.task(name="ad_mri.worker.run_explainability_job")
def run_explainability_job(prediction_id: str, method: str) -> dict[str, str]:
    """Raises: NotImplementedError — implemented in the explainability phase."""
    raise NotImplementedError("Async explainability jobs land in the explainability phase.")
