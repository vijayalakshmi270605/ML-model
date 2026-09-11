"""Delayed model-performance monitoring: has this model version's real,
ground-truth-backed accuracy degraded since it was registered?

**Ground truth is never assumed to be immediately available** - a prediction
is scored the moment it's made (`inference.predictor.InferenceEngine.predict`),
but *evaluated* only once a true label is later recorded
(`db.monitoring_service.MonitoringService.record_ground_truth`, via
`POST /api/v1/monitoring/ground-truth` or `ad-mri monitor record-ground-truth`
- days or weeks after the prediction, in a real clinical workflow). This
module is deliberately never called from the request path; it is always an
explicit, offline computation (`ad-mri monitor model-drift-report`, or a
scheduled job) over whatever labels have accumulated so far.

`MIN_LABELED_SAMPLES_FOR_MODEL_DRIFT` (`monitoring.thresholds`) is a hard
floor: with fewer labels than that, this returns a report with
`insufficient_samples=True` and **no drift verdict at all**, rather than
computing a technically-real-but-statistically-meaningless accuracy from 4
labeled predictions and calling it "model drift". The alternative - silently
lowering the bar because labels are scarce - would trade a true "we don't
know yet" for a false "everything's fine" or a false alarm, exactly the kind
of confident-sounding wrong answer this project's whole disclaimer posture
(README, docs/TECHNICAL_SPECIFICATION.md §0) exists to avoid.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ad_mri.core.logging import get_logger
from ad_mri.monitoring.thresholds import (
    ACCURACY_DEGRADATION_ALERT_THRESHOLD,
    F1_DEGRADATION_ALERT_THRESHOLD,
    MIN_LABELED_SAMPLES_FOR_MODEL_DRIFT,
)

logger = get_logger(__name__)

#: Which baseline metrics a degradation is actually checked against - kept
#: small and specific (not "every metric in the dict") so one noisy
#: per-class metric on a rare class can't trigger a false alert on its own.
MONITORED_METRICS = {
    "accuracy": ACCURACY_DEGRADATION_ALERT_THRESHOLD,
    "f1_macro": F1_DEGRADATION_ALERT_THRESHOLD,
}


@dataclass(frozen=True)
class ModelDriftReport:
    generated_at: str
    model_version_id: str
    n_labeled_samples: int
    insufficient_samples: bool
    current_metrics: dict[str, float]
    baseline_metrics: dict[str, float] | None
    degraded: bool
    degraded_metrics: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [
            "# Model Drift Report (Delayed Performance)",
            "",
            f"Generated: {self.generated_at}",
        ]
        lines.append(f"Model version: {self.model_version_id}")
        lines.append(f"Ground-truth-labeled predictions available: {self.n_labeled_samples}")

        if self.insufficient_samples:
            lines += [
                "",
                f"**Insufficient labeled samples** (need >= {MIN_LABELED_SAMPLES_FOR_MODEL_DRIFT}) "
                "- no drift verdict computed. This is expected in early production; "
                "ground truth accumulates over time, not immediately.",
            ]
            return "\n".join(lines) + "\n"

        verdict = "DEGRADED" if self.degraded else "no significant degradation"
        lines += ["", f"**Verdict: {verdict}**", ""]
        if self.baseline_metrics is None:
            lines.append(
                "No baseline_metrics recorded for this model version - current metrics computed, "
                "but no comparison was possible. Set one via `ad-mri monitor register-baseline`."
            )
        else:
            lines += ["| Metric | Baseline | Current | Delta | Degraded |", "|---|---|---|---|---|"]
            for metric in sorted(MONITORED_METRICS):
                baseline = self.baseline_metrics.get(metric)
                current = self.current_metrics.get(metric)
                if baseline is None or current is None:
                    continue
                delta = current - baseline
                flagged = metric in self.degraded_metrics
                lines.append(
                    f"| {metric} | {baseline:.4f} | {current:.4f} | {delta:+.4f} | "
                    f"{'yes' if flagged else 'no'} |"
                )
        return "\n".join(lines) + "\n"

    def save(self, output_dir: Path) -> dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "model_drift_report.json"
        md_path = output_dir / "model_drift_report.md"
        json_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        md_path.write_text(self.to_markdown(), encoding="utf-8")
        return {"json": json_path, "markdown": md_path}


def compute_delayed_performance(
    model_version_id: str,
    labeled_pairs: Sequence[tuple[str, str]],
    class_names: Sequence[str],
    baseline_metrics: dict[str, float] | None,
) -> ModelDriftReport:
    """`labeled_pairs`: `(predicted_class, true_class)` for every prediction
    made by `model_version_id` that has since received a ground-truth label
    (`db.monitoring_service.MonitoringService.get_labeled_predictions`).
    """
    n = len(labeled_pairs)
    if n < MIN_LABELED_SAMPLES_FOR_MODEL_DRIFT:
        return ModelDriftReport(
            generated_at=datetime.now(UTC).isoformat(),
            model_version_id=model_version_id,
            n_labeled_samples=n,
            insufficient_samples=True,
            current_metrics={},
            baseline_metrics=baseline_metrics,
            degraded=False,
            degraded_metrics=[],
        )

    from ad_mri.evaluation.metrics import compute_classification_metrics

    class_index = {name: i for i, name in enumerate(class_names)}
    y_pred: list[int] = []
    y_true: list[int] = []
    skipped = 0
    for predicted_class, true_class in labeled_pairs:
        if predicted_class not in class_index or true_class not in class_index:
            skipped += 1
            continue
        y_pred.append(class_index[predicted_class])
        y_true.append(class_index[true_class])

    if skipped:
        logger.warning(
            "model_drift_labels_skipped",
            skipped=skipped,
            reason="predicted_class or true_class not in the model's known class_names",
        )

    current_metrics = compute_classification_metrics(y_true, y_pred, list(class_names))

    degraded_metrics: list[str] = []
    if baseline_metrics is not None:
        for metric, threshold in MONITORED_METRICS.items():
            baseline_value = baseline_metrics.get(metric)
            current_value = current_metrics.get(metric)
            if baseline_value is None or current_value is None:
                continue
            if baseline_value - current_value >= threshold:
                degraded_metrics.append(metric)

    return ModelDriftReport(
        generated_at=datetime.now(UTC).isoformat(),
        model_version_id=model_version_id,
        n_labeled_samples=n,
        insufficient_samples=False,
        current_metrics=current_metrics,
        baseline_metrics=baseline_metrics,
        degraded=bool(degraded_metrics),
        degraded_metrics=degraded_metrics,
    )
