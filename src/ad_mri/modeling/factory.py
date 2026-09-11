"""Model factory: builds a model instance from a `ModelConfig`.

Keeps architecture choice a configuration concern (`ModelConfig.architecture`),
not a code fork — see docs/TECHNICAL_SPECIFICATION.md §5.
"""

from __future__ import annotations

from typing import Any

from ad_mri.core.exceptions import ConfigurationError
from ad_mri.modeling.architectures import _ARCHITECTURE_REGISTRY
from ad_mri.modeling.model_config import ModelConfig


def build_model(config: ModelConfig) -> Any:
    """Instantiate the architecture named by `config.architecture`.

    Raises:
        ConfigurationError: `config.architecture` isn't registered (see
            `modeling.architectures.register_architecture`).
    """
    try:
        model_cls = _ARCHITECTURE_REGISTRY[config.architecture]
    except KeyError:
        available = sorted(_ARCHITECTURE_REGISTRY)
        raise ConfigurationError(
            f"Unknown architecture {config.architecture!r}; registered: {available}"
        ) from None
    return model_cls(config)
