"""Data-drift detection: input distribution, image quality, and intensity
distribution changes, plus (where feasible) other feature distributions.

Two halves:

1. **Real-time signal** (no extra dependency, runs inside the API process):
   `compute_image_statistics()` runs on the *preprocessed* volume
   `inference.predictor.InferenceEngine` already has in memory for every
   prediction - free, no extra I/O, and measures exactly what the model
   actually sees. Recorded to Prometheus (`monitoring.metrics`) for live
   dashboards and, best-effort, to `input_statistics`
   (`db.monitoring_service`) for the offline report below.
2. **Offline drift report** (`train` extra - needs `scipy`/`numpy`; run via
   `ad-mri monitor drift-report`): compares a window of recorded
   `input_statistics` rows against a `DriftReference` built once from the
   training distribution (`ad-mri monitor build-drift-reference`), using the
   Population Stability Index (PSI) and a two-sample Kolmogorov-Smirnov test
   per feature - see `docs/monitoring.md` for why both, and the alert
   thresholds each one is checked against
   (`monitoring.thresholds`).

Class distribution drift (the *predicted*-class balance shifting) is
reported here too, via a categorical PSI over `predicted_class` counts -
`prediction_distribution`/`confidence_distribution` themselves are
real-time-only (`monitoring.metrics.PREDICTION_CONFIDENCE`/
`CLASS_PROBABILITY`); this module's `class_distribution` feature is the
drift-report treatment of the same signal Prometheus already exposes live.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ad_mri.core.exceptions import ConfigurationError
from ad_mri.core.logging import get_logger
from ad_mri.monitoring.thresholds import (
    KS_PVALUE_ALERT_THRESHOLD,
    PSI_ALERT_THRESHOLD,
    PSI_WARNING_THRESHOLD,
)

logger = get_logger(__name__)

FEATURE_NAMES = (
    "mean_intensity",
    "std_intensity",
    "p01_intensity",
    "p50_intensity",
    "p99_intensity",
    "nonzero_fraction",
)


@dataclass(frozen=True)
class ImageStatistics:
    """Aggregate statistics of one preprocessed volume - never the array
    itself. See `db.models.InputStatisticsRecord` for the persisted form."""

    mean_intensity: float
    std_intensity: float
    p01_intensity: float
    p50_intensity: float
    p99_intensity: float
    nonzero_fraction: float
    voxel_count: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def compute_image_statistics(array: Any) -> ImageStatistics:
    """Compute `ImageStatistics` for a preprocessed volume.

    Accepts a numpy array or anything with a `.numpy()` method (a torch
    tensor) - this module never imports torch itself (see the module-layering
    convention `explainability`/`storage.object_storage` already establish:
    a shared-concern module depends on the array-like value it's handed, not
    on whichever framework produced it).
    """
    import numpy as np

    np_array = array.numpy() if hasattr(array, "numpy") else np.asarray(array)
    flat = np_array.astype(np.float64).ravel()

    # An empty volume should never reach here in practice (the preprocessing
    # pipeline's own QC/shape checks reject it first) - guarded explicitly
    # anyway because np.mean/np.percentile on an empty array raise or return
    # NaN, and a monitoring function must never itself crash the request it's
    # only supposed to be observing.
    if flat.size == 0:
        return ImageStatistics(
            mean_intensity=0.0,
            std_intensity=0.0,
            p01_intensity=0.0,
            p50_intensity=0.0,
            p99_intensity=0.0,
            nonzero_fraction=0.0,
            voxel_count=0,
        )

    return ImageStatistics(
        mean_intensity=float(np.mean(flat)),
        std_intensity=float(np.std(flat)),
        p01_intensity=float(np.percentile(flat, 1)),
        p50_intensity=float(np.percentile(flat, 50)),
        p99_intensity=float(np.percentile(flat, 99)),
        nonzero_fraction=float(np.count_nonzero(flat)) / flat.size,
        voxel_count=int(flat.size),
    )


@dataclass(frozen=True)
class DriftReference:
    """The training-distribution baseline a current window is compared
    against. Built once (`build_drift_reference`), saved as JSON, versioned
    like `PreprocessingConfig`/`ModelConfig` (a content hash, not a
    timestamp-only version)."""

    #: `{feature_name: [sampled values from the reference set]}` - the raw
    #: samples, not just mean/std, because PSI/KS both need the underlying
    #: distribution shape, not a two-number summary of it.
    feature_samples: dict[str, list[float]]
    #: Predicted^ class counts is not available for a reference built from
    #: raw training images (they have no model prediction) - this holds the
    #: *label* distribution of the reference set instead, the closest
    #: available proxy for "what class balance did this model train on".
    class_counts: dict[str, int]
    n_samples: int
    built_at: str
    source_manifest: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DriftReference:
        return cls(**data)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> DriftReference:
        if not path.exists():
            raise ConfigurationError(
                f"No drift reference at {path} - run `ad-mri monitor build-drift-reference` first."
            )
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def build_drift_reference(
    image_paths: Sequence[Path],
    labels: Sequence[str] | None,
    preprocessing_config: Any,
    source_manifest: str,
    max_samples: int | None = None,
) -> DriftReference:
    """Run every image in `image_paths` through the *same* preprocessing
    pipeline inference uses, compute `ImageStatistics` for each, and collect
    them into a `DriftReference`. Offline, `train`-extra only (needs
    `datasets.transforms`, which needs monai/torch) - see
    `cli.main`'s `monitor build-drift-reference` command.
    """
    from ad_mri.datasets.transforms import apply_transforms, build_transforms

    if max_samples is not None:
        image_paths = image_paths[:max_samples]
        labels = labels[:max_samples] if labels is not None else None

    transform = build_transforms(preprocessing_config, train=False)
    feature_samples: dict[str, list[float]] = {name: [] for name in FEATURE_NAMES}
    class_counts: dict[str, int] = {}
    n_ok = 0

    for i, image_path in enumerate(image_paths):
        try:
            sample = apply_transforms(transform, {"image": str(image_path)})
            stats = compute_image_statistics(sample["image"])
        except (
            Exception
        ) as exc:  # noqa: BLE001 - one bad reference image must not abort the whole build
            logger.warning("drift_reference_image_skipped", index=i, error_type=type(exc).__name__)
            continue
        for name in FEATURE_NAMES:
            feature_samples[name].append(getattr(stats, name))
        n_ok += 1
        if labels is not None:
            label = labels[i]
            class_counts[label] = class_counts.get(label, 0) + 1

    if n_ok == 0:
        raise ConfigurationError(
            "No reference image could be preprocessed - cannot build a drift reference."
        )

    return DriftReference(
        feature_samples=feature_samples,
        class_counts=class_counts,
        n_samples=n_ok,
        built_at=datetime.now(UTC).isoformat(),
        source_manifest=source_manifest,
    )


def population_stability_index(
    reference: Sequence[float], current: Sequence[float], bins: int = 10
) -> float:
    """PSI between `reference` and `current`, using quantile bins computed
    from `reference` (the standard, "expected vs. actual" formulation - bin
    edges never move just because the current window's distribution did).

    Conventional interpretation (see `monitoring.thresholds`):
    < 0.1 no significant shift, 0.1-0.25 moderate (investigate), > 0.25
    significant (alert). Returns `0.0` if either sample is empty (nothing to
    compare) or every reference value is identical (no meaningful binning
    possible) - never raises, never fabricates a large score from degenerate
    input.
    """
    import numpy as np

    ref = np.asarray(reference, dtype=np.float64)
    cur = np.asarray(current, dtype=np.float64)
    if ref.size == 0 or cur.size == 0:
        return 0.0

    quantiles = np.linspace(0, 100, bins + 1)
    edges = np.unique(np.percentile(ref, quantiles))
    if edges.size < 2:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)

    ref_frac = ref_counts / ref.size
    cur_frac = cur_counts / cur.size
    # A zero-frequency bin makes the PSI term undefined (division/log by
    # zero) - the standard fix is a small floor, not skipping the bin
    # (skipping would silently ignore a bin the current window abandoned
    # entirely, exactly the kind of shift PSI exists to catch).
    epsilon = 1e-4
    ref_frac = np.where(ref_frac == 0, epsilon, ref_frac)
    cur_frac = np.where(cur_frac == 0, epsilon, cur_frac)

    return float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))


@dataclass(frozen=True)
class FeatureDriftResult:
    feature: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    reference_n: int
    current_n: int
    drifted: bool
    severity: str  # "none" | "warning" | "alert"


@dataclass(frozen=True)
class DriftReport:
    generated_at: str
    reference_built_at: str
    reference_n_samples: int
    current_n_samples: int
    window_start: str | None
    window_end: str | None
    feature_results: list[FeatureDriftResult]
    class_distribution_psi: float | None
    class_distribution_drifted: bool
    any_drifted: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            **{k: v for k, v in asdict(self).items() if k != "feature_results"},
            "feature_results": [asdict(r) for r in self.feature_results],
        }

    def to_markdown(self) -> str:
        lines = [
            "# Data Drift Report",
            "",
            f"Generated: {self.generated_at}",
            f"Reference: {self.reference_n_samples} samples, built {self.reference_built_at}",
            f"Current window: {self.current_n_samples} samples"
            + (f" ({self.window_start} to {self.window_end})" if self.window_start else ""),
            "",
            f"**Overall verdict: {'DRIFT DETECTED' if self.any_drifted else 'no drift detected'}**",
            "",
            "| Feature | PSI | KS statistic | KS p-value | Severity |",
            "|---|---|---|---|---|",
        ]
        for r in self.feature_results:
            lines.append(
                f"| {r.feature} | {r.psi:.4f} | {r.ks_statistic:.4f} | "
                f"{r.ks_pvalue:.4g} | {r.severity} |"
            )
        if self.class_distribution_psi is not None:
            lines += [
                "",
                f"Predicted-class distribution PSI: {self.class_distribution_psi:.4f} "
                f"({'DRIFTED' if self.class_distribution_drifted else 'stable'})",
            ]
        return "\n".join(lines) + "\n"

    def save(self, output_dir: Path) -> dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "drift_report.json"
        md_path = output_dir / "drift_report.md"
        json_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        md_path.write_text(self.to_markdown(), encoding="utf-8")
        return {"json": json_path, "markdown": md_path}


def _categorical_psi(reference: dict[str, int], current: dict[str, int]) -> float:
    """PSI over class-count distributions (a discrete analogue of the
    numeric-feature PSI above - same formula, categories instead of
    quantile bins)."""
    import numpy as np

    ref_total = sum(reference.values())
    cur_total = sum(current.values())
    if ref_total == 0 or cur_total == 0:
        return 0.0

    categories = sorted(set(reference) | set(current))
    epsilon = 1e-4
    psi = 0.0
    for category in categories:
        ref_frac = max(reference.get(category, 0) / ref_total, epsilon)
        cur_frac = max(current.get(category, 0) / cur_total, epsilon)
        psi += (cur_frac - ref_frac) * float(np.log(cur_frac / ref_frac))
    return float(psi)


def _severity(psi: float, ks_pvalue: float) -> tuple[bool, str]:
    drifted = psi >= PSI_ALERT_THRESHOLD or ks_pvalue < KS_PVALUE_ALERT_THRESHOLD
    if psi >= PSI_ALERT_THRESHOLD:
        return drifted, "alert"
    if psi >= PSI_WARNING_THRESHOLD:
        return drifted, "warning"
    return drifted, "none"


def generate_data_drift_report(
    reference: DriftReference,
    current_feature_samples: dict[str, list[float]],
    current_class_counts: dict[str, int] | None = None,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
) -> DriftReport:
    """Compare `current_feature_samples` (e.g. from
    `db.monitoring_service.MonitoringService.get_input_statistics_window`)
    against `reference`, per feature, via PSI + a two-sample KS test."""
    from scipy import stats as scipy_stats

    feature_results: list[FeatureDriftResult] = []
    current_n = 0
    for name in FEATURE_NAMES:
        ref_values = reference.feature_samples.get(name, [])
        cur_values = current_feature_samples.get(name, [])
        current_n = max(current_n, len(cur_values))

        psi = population_stability_index(ref_values, cur_values)
        if len(ref_values) >= 2 and len(cur_values) >= 2:
            ks_result = scipy_stats.ks_2samp(ref_values, cur_values)
            ks_statistic, ks_pvalue = float(ks_result.statistic), float(ks_result.pvalue)
        else:
            ks_statistic, ks_pvalue = 0.0, 1.0

        drifted, severity = _severity(psi, ks_pvalue)
        feature_results.append(
            FeatureDriftResult(
                feature=name,
                psi=psi,
                ks_statistic=ks_statistic,
                ks_pvalue=ks_pvalue,
                reference_n=len(ref_values),
                current_n=len(cur_values),
                drifted=drifted,
                severity=severity,
            )
        )

    class_psi: float | None = None
    class_drifted = False
    if current_class_counts is not None:
        class_psi = _categorical_psi(reference.class_counts, current_class_counts)
        class_drifted = class_psi >= PSI_ALERT_THRESHOLD

    return DriftReport(
        generated_at=datetime.now(UTC).isoformat(),
        reference_built_at=reference.built_at,
        reference_n_samples=reference.n_samples,
        current_n_samples=current_n,
        window_start=window_start.isoformat() if window_start else None,
        window_end=window_end.isoformat() if window_end else None,
        feature_results=feature_results,
        class_distribution_psi=class_psi,
        class_distribution_drifted=class_drifted,
        any_drifted=any(r.drifted for r in feature_results) or class_drifted,
    )
