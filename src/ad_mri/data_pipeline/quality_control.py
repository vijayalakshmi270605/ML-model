"""Automated quality-control checks on preprocessed MRI volumes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class QualityControlReport:
    passed: bool
    brain_mask_coverage_ratio: float | None = None
    notes: list[str] = field(default_factory=list)


def run_quality_checks(image_path: Path) -> QualityControlReport:
    """Raises: NotImplementedError — implemented in the data-pipeline phase."""
    raise NotImplementedError("Quality control checks are implemented in the data-pipeline phase.")
