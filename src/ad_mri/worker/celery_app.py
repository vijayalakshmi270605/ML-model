"""Celery application for async explainability/batch jobs.

See docs/TECHNICAL_SPECIFICATION.md §7, §9. Runs in the dedicated worker
container (docker/worker.Dockerfile), never in the API process.
"""

from __future__ import annotations

from celery import Celery

from ad_mri.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "ad_mri",
    broker=settings.get_celery_broker_url(),
    backend=settings.get_celery_result_backend(),
)
celery_app.autodiscover_tasks(["ad_mri.worker"])
