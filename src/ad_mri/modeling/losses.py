"""Loss functions, including class-imbalance-aware variants."""

from __future__ import annotations

from typing import Any


def build_loss(name: str, class_weights: list[float] | None = None, **kwargs: Any) -> Any:
    """Raises: NotImplementedError — implemented in the modeling phase."""
    raise NotImplementedError("Loss functions are implemented in the modeling phase.")
