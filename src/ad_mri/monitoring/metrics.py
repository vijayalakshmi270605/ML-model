"""Prometheus metrics definitions, scraped via `GET /metrics`.

Organized by what Phase 13 requires monitored, not by which module happens to
update them - see docs/monitoring.md for the full catalog (which metric
answers which requirement bullet, including ones answered by a PromQL
expression over an existing series rather than a dedicated one) and
deploy/prometheus/alerts.yml for the alert rules built on top of these.

- **System**: `monitoring.system_metrics` samples CPU/memory/GPU gauges on
  every scrape (pull-based, not a background thread - see that module).
- **HTTP/request**: `api.middleware.logging_middleware.AccessLogMiddleware`
  updates `REQUEST_COUNT`/`REQUEST_LATENCY_SECONDS` - request count and
  latency directly; throughput and error rate are PromQL `rate()`/ratio
  expressions over `REQUEST_COUNT`, not separate series (avoiding a
  redundant, driftable second counter for the same underlying events).
- **Model**: `inference.predictor.InferenceEngine.predict()` updates
  `INFERENCE_REQUESTS_TOTAL` (outcome, including "model latency" via
  `INFERENCE_DURATION_SECONDS`), `PREDICTED_CLASS_TOTAL` (class
  distribution), `PREDICTION_CONFIDENCE` (confidence distribution),
  `CLASS_PROBABILITY` (prediction/probability distribution per class).
- **Data drift**: `INPUT_MEAN_INTENSITY`/`INPUT_STD_INTENSITY`/
  `INPUT_NONZERO_FRACTION` are the real-time signal; `DRIFT_ALERT_ACTIVE`
  reflects the most recent offline `monitoring.drift` report's verdict.
- **Model drift**: `MODEL_DRIFT_ALERT_ACTIVE` reflects the most recent
  offline `monitoring.model_drift` report's verdict - see that module for why
  this can only ever be a periodically-refreshed gauge, never a per-request one
  (ground truth is never available at request time).
"""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

# --- HTTP / request (system: latency, throughput, error rate, request count) ---

REQUEST_COUNT = Counter(
    "ad_mri_http_requests_total",
    "Total HTTP requests processed.",
    ["method", "path", "status_code"],
)

REQUEST_LATENCY_SECONDS = Histogram(
    "ad_mri_http_request_duration_seconds",
    "HTTP request latency in seconds.",
    ["method", "path"],
)

# --- System resources (system: CPU, memory, GPU) --------------------------
# Updated by monitoring.system_metrics.sample_system_metrics(), called once
# per /metrics scrape - see that module for why pull-based beats a
# background-thread sampler here.

PROCESS_CPU_PERCENT = Gauge(
    "ad_mri_process_cpu_percent",
    "This process's CPU utilization, percent (can exceed 100 on multi-core).",
)
PROCESS_MEMORY_BYTES = Gauge(
    "ad_mri_process_memory_bytes", "This process's resident memory (RSS), bytes."
)
SYSTEM_CPU_PERCENT = Gauge("ad_mri_system_cpu_percent", "Host-wide CPU utilization, percent.")
SYSTEM_MEMORY_PERCENT = Gauge(
    "ad_mri_system_memory_percent", "Host-wide memory utilization, percent."
)
GPU_UTILIZATION_PERCENT = Gauge(
    "ad_mri_gpu_utilization_percent", "GPU utilization, percent.", ["device"]
)
GPU_MEMORY_USED_BYTES = Gauge(
    "ad_mri_gpu_memory_used_bytes", "GPU memory currently allocated, bytes.", ["device"]
)

# --- Model (prediction distribution, class distribution, confidence
# distribution, input validation failures, preprocessing failures, model
# latency) -------------------------------------------------------------

INFERENCE_REQUESTS_TOTAL = Counter(
    "ad_mri_inference_requests_total",
    "Total inference requests, by outcome.",
    # "validation_failed" and "preprocessing_failed" were a single conflated
    # "invalid_input" label before Phase 13 - split so each failure mode
    # (bad upload vs. a QC/preprocessing-pipeline problem) is independently
    # observable, per the requirement listing them as two distinct things to
    # monitor.
    ["outcome"],  # "success" | "validation_failed" | "preprocessing_failed" | "timeout" | "error"
)

INFERENCE_DURATION_SECONDS = Histogram(
    "ad_mri_inference_duration_seconds",
    "End-to-end inference pipeline latency in seconds (validation through "
    "calibration) - model latency.",
)

PREDICTED_CLASS_TOTAL = Counter(
    "ad_mri_predicted_class_total",
    "Count of predictions by predicted class - class distribution.",
    ["predicted_class"],
)

PREDICTION_CONFIDENCE = Histogram(
    "ad_mri_prediction_confidence",
    "The predicted (top-1/argmax) class's probability - confidence distribution.",
    buckets=(0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
)

CLASS_PROBABILITY = Histogram(
    "ad_mri_class_probability",
    "Predicted probability for each class on every request (not just the "
    "argmax) - prediction distribution.",
    ["class_name"],
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

# --- Data drift (real-time signal; see monitoring.drift for the offline,
# reference-comparison half) --------------------------------------------

INPUT_MEAN_INTENSITY = Histogram(
    "ad_mri_input_mean_intensity",
    "Mean voxel intensity of the preprocessed volume - intensity distribution.",
)
INPUT_STD_INTENSITY = Histogram(
    "ad_mri_input_std_intensity", "Voxel intensity standard deviation of the preprocessed volume."
)
INPUT_NONZERO_FRACTION = Histogram(
    "ad_mri_input_nonzero_fraction",
    "Fraction of nonzero voxels in the preprocessed volume - image quality signal "
    "(a near-empty or near-solid volume is usually a bad scan/crop, not a real one).",
)

# --- Drift alert state (set by the offline report jobs -
# monitoring.drift/monitoring.model_drift - not per-request) --------------

DRIFT_ALERT_ACTIVE = Gauge(
    "ad_mri_data_drift_alert_active",
    "1 if the most recent data-drift report exceeded the alert threshold for this feature, else 0.",
    ["feature"],
)
MODEL_DRIFT_ALERT_ACTIVE = Gauge(
    "ad_mri_model_drift_alert_active",
    "1 if the most recent (delayed, ground-truth-backed) model-drift report exceeded "
    "the degradation threshold for this metric, else 0.",
    ["metric"],
)


def render_latest_metrics() -> tuple[bytes, str]:
    """Return the current metrics snapshot and its content type for the /metrics response."""
    from ad_mri.monitoring.alert_state import sample_alert_state
    from ad_mri.monitoring.system_metrics import sample_system_metrics

    sample_system_metrics()
    sample_alert_state()
    return generate_latest(), CONTENT_TYPE_LATEST
