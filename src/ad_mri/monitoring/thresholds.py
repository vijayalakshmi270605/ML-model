"""Centralized alert thresholds and the `Alert` type every monitoring report
in this package produces its verdicts in.

Every threshold lives here, once - never re-typed as a bare literal in
`monitoring.drift`/`monitoring.model_drift`/`deploy/prometheus/alerts.yml`.
The Prometheus alert rules file is a separate document (PromQL, not Python)
and so cannot literally import this module, but every threshold value it
uses is a comment-annotated copy of the constant defined here - see that
file's header and docs/monitoring.md's threshold table, which is generated
by eye from this module and re-checked whenever a threshold here changes.

Defaults are industry-standard starting points (PSI's 0.1/0.25 bands are the
conventional ones from credit-risk-model monitoring, the field PSI
originates in), not values tuned against this project's own model. They are
plain module constants, not `core.config.Settings` fields - changing one for
a real deployment means editing this file (and `deploy/prometheus/alerts.yml`'s
matching PromQL literal) directly, not setting an environment variable.
Wiring these through `Settings` instead is a natural extension if a
deployment ever needs to tune sensitivity without a code change - not done
here to avoid a dozen new environment variables for values that, in
practice, get tuned by editing code and redeploying rather than by ops
toggling a knob at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

# --- Data drift (monitoring.drift) -----------------------------------------

#: PSI < this: no significant distribution shift.
PSI_WARNING_THRESHOLD = 0.1
#: PSI >= this: significant shift - `DriftReport.any_drifted` fires.
PSI_ALERT_THRESHOLD = 0.25
#: Two-sample KS test p-value below this is treated as statistically
#: significant evidence of a distribution change (independent of, and in
#: addition to, the PSI check - either one alone can trigger a drift verdict).
KS_PVALUE_ALERT_THRESHOLD = 0.01

# --- Model drift / delayed performance monitoring (monitoring.model_drift) -

#: Absolute drop in accuracy (production vs. baseline) that triggers an alert.
ACCURACY_DEGRADATION_ALERT_THRESHOLD = 0.05
#: Absolute drop in macro-F1 that triggers an alert.
F1_DEGRADATION_ALERT_THRESHOLD = 0.05
#: Minimum number of ground-truth-labeled predictions required before a
#: delayed-performance verdict is computed at all - below this, the sample is
#: too small for a percentage-point comparison to mean anything (see
#: monitoring.model_drift's module docstring for why this is a hard floor,
#: not a soft warning).
MIN_LABELED_SAMPLES_FOR_MODEL_DRIFT = 30

# --- Model / request quality (checked against Prometheus counters by the
# alert rules, thresholds duplicated here for docs/monitoring.md's table) --

#: Fraction of predictions with confidence below this, over a rolling
#: window, that triggers a "low-confidence rate" alert.
LOW_CONFIDENCE_THRESHOLD = 0.6
LOW_CONFIDENCE_RATE_ALERT_THRESHOLD = 0.2
#: Fraction of inference requests failing validation/preprocessing, over a
#: rolling window, that triggers an alert.
INPUT_FAILURE_RATE_ALERT_THRESHOLD = 0.1
#: HTTP 5xx rate, over a rolling window, that triggers an alert.
HTTP_ERROR_RATE_ALERT_THRESHOLD = 0.05
#: p95 inference latency (seconds) that triggers an alert.
INFERENCE_LATENCY_P95_ALERT_SECONDS = 10.0

# --- System resources --------------------------------------------------

CPU_PERCENT_ALERT_THRESHOLD = 90.0
MEMORY_PERCENT_ALERT_THRESHOLD = 90.0


class AlertSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class Alert:
    """One threshold check's outcome - produced by every `evaluate_*_alerts`
    function in this package and, for anything worth a durable record,
    written to `system_events` (`db.audit_service.AuditService.record_event`,
    `event_type="<category>_alert"`) by whichever CLI command generated it."""

    name: str
    severity: AlertSeverity
    message: str
    metric: str
    value: float
    threshold: float
