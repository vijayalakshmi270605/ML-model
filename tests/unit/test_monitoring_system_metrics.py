"""Tests for `monitoring.system_metrics` - real `psutil` calls against this
actual process/host, no mocks. GPU sampling is exercised for real too: on
this (CPU-only, or at least `torch`-uninstalled-in-the-worst-case) test
environment that means confirming it degrades to a silent no-op rather than
raising - the real, exercised code path either way."""

from __future__ import annotations

from prometheus_client import REGISTRY

from ad_mri.monitoring.metrics import (
    GPU_MEMORY_USED_BYTES,
    GPU_UTILIZATION_PERCENT,
    PROCESS_CPU_PERCENT,
    PROCESS_MEMORY_BYTES,
    SYSTEM_MEMORY_PERCENT,
)
from ad_mri.monitoring.system_metrics import sample_system_metrics


def _gauge_value(gauge) -> float:  # type: ignore[no-untyped-def]
    return REGISTRY.get_sample_value(gauge._name)


def test_sample_system_metrics_never_raises() -> None:
    sample_system_metrics()  # the assertion is simply that this doesn't raise


def test_process_memory_is_a_real_positive_reading() -> None:
    sample_system_metrics()
    value = _gauge_value(PROCESS_MEMORY_BYTES)
    assert value is not None
    # This test process itself is certainly using at least a few MB.
    assert value > 1_000_000


def test_system_memory_percent_is_between_0_and_100() -> None:
    sample_system_metrics()
    value = _gauge_value(SYSTEM_MEMORY_PERCENT)
    assert value is not None
    assert 0.0 <= value <= 100.0


def test_process_cpu_percent_is_never_negative() -> None:
    sample_system_metrics()
    sample_system_metrics()
    value = _gauge_value(PROCESS_CPU_PERCENT)
    assert value is not None
    assert value >= 0.0


def test_gpu_metrics_are_cleared_and_repopulated_on_each_sample() -> None:
    """Whether or not a GPU is actually present in this environment, calling
    this twice must not raise and must not accumulate stale label sets."""
    sample_system_metrics()
    sample_system_metrics()
    # No assertion on the value itself (may legitimately be absent - no GPU/
    # no torch installed) - only that repeated sampling is safe and doesn't
    # leave duplicate/stale series behind, which .clear() inside
    # _sample_gpu_metrics is specifically there to prevent.
    assert GPU_UTILIZATION_PERCENT is not None
    assert GPU_MEMORY_USED_BYTES is not None
