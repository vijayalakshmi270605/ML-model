"""Celery application and async tasks (explainability jobs, batch predictions).

`celery`/`redis` are training/worker-phase dependencies (requirements/train.txt) —
this package is never imported by the API process.
"""
