"""Thin wrapper around the MLflow Model Registry.

`mlflow` is a training-phase dependency (requirements/train.txt). Methods import
it lazily so this module stays importable without it installed - independent of
whether the current deployment happens to have the `train` extra present (as
`docker/api.Dockerfile` does, since Phase 9's in-process inference architecture
means the api image needs torch/monai too - see pyproject.toml's `dependencies`
comment), a module that never *requires* an import at load time is more
resilient to a future, genuinely lean deployment that doesn't.
See docs/TECHNICAL_SPECIFICATION.md §8, §16.
"""

from __future__ import annotations

from ad_mri.core.constants import ModelStage


class MLflowModelRegistryClient:
    """Registers, promotes, and resolves model versions in the MLflow registry."""

    def __init__(self, tracking_uri: str | None = None) -> None:
        self._tracking_uri = tracking_uri

    def get_model_uri(self, name: str, stage: ModelStage = ModelStage.PRODUCTION) -> str:
        """Raises: NotImplementedError — implemented in the training/MLOps phase."""
        raise NotImplementedError("MLflow registry integration lands in the training phase.")

    def register_model(self, name: str, run_id: str, artifact_path: str) -> str:
        """Raises: NotImplementedError — implemented in the training/MLOps phase."""
        raise NotImplementedError("MLflow registry integration lands in the training phase.")

    def promote_model(self, name: str, version: str, stage: ModelStage) -> None:
        """Raises: NotImplementedError — implemented in the training/MLOps phase."""
        raise NotImplementedError("MLflow registry integration lands in the training phase.")
