"""Structured, citation-backed comparison against published research results.

Populates *reported* literature numbers (with citations) alongside this
system's *measured* numbers. Never fabricates or assumes a performance figure
for either side — see docs/TECHNICAL_SPECIFICATION.md §0.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LiteratureBenchmark:
    citation: str
    dataset: str
    reported_metric_name: str
    reported_metric_value: float


def generate_literature_comparison_table(
    measured_metrics: dict[str, float], benchmarks: list[LiteratureBenchmark]
) -> list[dict[str, object]]:
    """Raises: NotImplementedError — implemented in the evaluation phase."""
    raise NotImplementedError("Literature comparison is implemented in the evaluation phase.")
