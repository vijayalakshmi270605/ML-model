"""System resource metrics: CPU, memory, and (best-effort) GPU utilization.

Pull-based, not a background sampler thread: `sample_system_metrics()` is
called once per `/metrics` scrape (`monitoring.metrics.render_latest_metrics`)
and updates the process/host/GPU gauges in place. This is deliberate, not an
oversight - a background thread would need its own lifecycle management
(start on app startup, stop on shutdown, survive worker respawns under a
process manager) for a benefit no different from sampling synchronously
inside the already-scheduled scrape: Prometheus scrapes every 10-30s in any
realistic deployment, which is a perfectly adequate resolution for CPU/memory
trend monitoring, and `psutil`'s calls here are sub-millisecond.

`psutil` is a base dependency (pyproject.toml) - process/host CPU and memory
metrics must work even in a deployment with no model configured at all. GPU
metrics go through `torch.cuda`, imported lazily and best-effort: a CPU-only
deployment (no `train` extra, or a `train`-extra deployment with no GPU) is
the common case, not an error, so its absence is silent - only an
*unexpected* failure while a GPU genuinely appears present logs a warning.
"""

from __future__ import annotations

import psutil

from ad_mri.core.logging import get_logger
from ad_mri.monitoring.metrics import (
    GPU_MEMORY_USED_BYTES,
    GPU_UTILIZATION_PERCENT,
    PROCESS_CPU_PERCENT,
    PROCESS_MEMORY_BYTES,
    SYSTEM_CPU_PERCENT,
    SYSTEM_MEMORY_PERCENT,
)

logger = get_logger(__name__)

#: One `Process` handle for the life of the interpreter - `cpu_percent()`
#: measures CPU time consumed *since the previous call on this same handle*;
#: a fresh `Process()` every scrape would always report ~0%.
_process = psutil.Process()


def _sample_gpu_metrics() -> None:
    """Best-effort - see module docstring. Clears any previously-set device
    labels first so a GPU that disappears (e.g. driver reset) doesn't leave a
    stale, no-longer-updating series behind."""
    GPU_UTILIZATION_PERCENT.clear()
    GPU_MEMORY_USED_BYTES.clear()
    try:
        import torch
    except ImportError:
        return  # no `train` extra installed - not an error, just nothing to report

    try:
        if not torch.cuda.is_available():
            return
        for index in range(torch.cuda.device_count()):
            device_label = f"cuda:{index}"
            try:
                utilization = torch.cuda.utilization(index)
                GPU_UTILIZATION_PERCENT.labels(device_label).set(utilization)
            except Exception as exc:  # noqa: BLE001 - one device's failure must not block the rest
                logger.warning(
                    "gpu_utilization_sample_failed",
                    device=device_label,
                    error_type=type(exc).__name__,
                )
            try:
                memory_used = torch.cuda.memory_allocated(index)
                GPU_MEMORY_USED_BYTES.labels(device_label).set(memory_used)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "gpu_memory_sample_failed", device=device_label, error_type=type(exc).__name__
                )
    except Exception as exc:  # noqa: BLE001 - a metrics scrape must never fail the request
        logger.warning("gpu_metrics_sample_failed", error_type=type(exc).__name__)


def sample_system_metrics() -> None:
    """Update every system-resource gauge with a fresh reading. Never
    raises - a failed sample is a gap in the series, not an application
    error (mirrors every other health/metrics function in this package)."""
    try:
        PROCESS_CPU_PERCENT.set(_process.cpu_percent(interval=None))
        PROCESS_MEMORY_BYTES.set(_process.memory_info().rss)
    except Exception as exc:  # noqa: BLE001
        logger.warning("process_metrics_sample_failed", error_type=type(exc).__name__)

    try:
        SYSTEM_CPU_PERCENT.set(psutil.cpu_percent(interval=None))
        SYSTEM_MEMORY_PERCENT.set(psutil.virtual_memory().percent)
    except Exception as exc:  # noqa: BLE001
        logger.warning("system_metrics_sample_failed", error_type=type(exc).__name__)

    _sample_gpu_metrics()
