"""Cross-fold / cross-trial / bootstrap statistical summaries: mean, standard
deviation, and confidence intervals — used to report cross-validation and
hyperparameter-search results without overstating precision from a handful
of folds/trials, and to report single-test-set evaluation metrics with a
resampling-based confidence interval instead of a bare point estimate. See
docs/TECHNICAL_SPECIFICATION.md §10, docs/model_selection.md, docs/evaluation.md.

`scipy` ships as a transitive dependency of `scikit-learn` (already in
requirements/train.txt), imported lazily so this module stays importable
without the training-phase extras.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MetricSummary:
    metric_name: str
    n: int
    mean: float
    std: float | None
    ci_lower: float | None
    ci_upper: float | None
    confidence_level: float
    values: list[float] = field(default_factory=list)


def summarize_values(
    metric_name: str, values: list[float], confidence_level: float = 0.95
) -> MetricSummary:
    """Mean/std/confidence-interval over `values` (e.g. one value per CV fold).

    Standard deviation and the CI require at least 2 values to be meaningful;
    with fewer, they're reported as `None` rather than a fabricated 0.0
    (requirement: "confidence intervals where appropriate"). Uses a
    t-distribution (not normal) critical value, appropriate for the typically
    small number of folds/trials involved.
    """
    n = len(values)
    if n == 0:
        raise ValueError(f"No values to summarize for {metric_name!r}.")

    mean = sum(values) / n
    if n < 2:
        return MetricSummary(metric_name, n, mean, None, None, None, confidence_level, list(values))

    variance = sum((v - mean) ** 2 for v in values) / (n - 1)  # sample variance (ddof=1)
    std = math.sqrt(variance)

    from scipy import stats

    standard_error = std / math.sqrt(n)
    t_critical = float(stats.t.ppf((1 + confidence_level) / 2, df=n - 1))
    margin = t_critical * standard_error

    return MetricSummary(
        metric_name=metric_name,
        n=n,
        mean=mean,
        std=std,
        ci_lower=mean - margin,
        ci_upper=mean + margin,
        confidence_level=confidence_level,
        values=list(values),
    )


def summarize_fold_metrics(
    fold_metrics: list[dict[str, float]], confidence_level: float = 0.95
) -> dict[str, MetricSummary]:
    """Summarize every metric key common to all fold result dicts."""
    if not fold_metrics:
        raise ValueError("No fold metrics to summarize.")

    common_keys: set[str] = set(fold_metrics[0])
    for row in fold_metrics[1:]:
        common_keys &= set(row)

    return {
        key: summarize_values(key, [row[key] for row in fold_metrics], confidence_level)
        for key in sorted(common_keys)
    }


def bootstrap_metric_ci(
    metric_fn: Callable[..., float],
    *arrays: Sequence[Any],
    metric_name: str,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> MetricSummary:
    """Non-parametric (percentile) bootstrap confidence interval for a metric
    computed on a *single* held-out test set, where the fold/trial-based
    t-distribution approach in `summarize_values` doesn't apply — there is
    only one dataset, so the resampling unit is individual test-set rows, not
    independent runs.

    `arrays` are one or more equal-length parallel sequences (e.g.
    `y_true, y_pred` for a label-only metric, or `y_true, y_proba` for an
    AUC-style metric) — each bootstrap replicate resamples row indices once
    and applies the *same* indices to every array, then calls
    `metric_fn(*resampled_arrays)`.

    `mean` in the returned `MetricSummary` is the point estimate on the
    original (unresampled) data, not the mean of the bootstrap distribution —
    the standard convention, avoiding bootstrap-induced bias in the headline
    number while still using the resampled distribution for the interval.
    `std` is the standard deviation of the bootstrap distribution. `values`
    holds the individual bootstrap replicate values.

    Reproducible: identical inputs + `seed` always produce identical resamples.
    """
    if not arrays:
        raise ValueError("bootstrap_metric_ci requires at least one array.")
    lengths = {len(array) for array in arrays}
    if len(lengths) > 1:
        raise ValueError(f"All arrays must be the same length, got lengths {sorted(lengths)}.")
    n = lengths.pop()
    if n == 0:
        raise ValueError(f"No values to bootstrap for {metric_name!r}.")

    import numpy as np

    point_estimate = float(metric_fn(*arrays))
    if n < 2:
        return MetricSummary(
            metric_name, n, point_estimate, None, None, None, confidence_level, [point_estimate]
        )

    array_objs = [np.asarray(array, dtype=object) for array in arrays]
    rng = np.random.default_rng(seed)

    replicates: list[float] = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        resampled = [arr[indices].tolist() for arr in array_objs]
        replicates.append(float(metric_fn(*resampled)))

    # nan-aware: a resample can be degenerate (e.g. missing a class), in which
    # case an AUC-style metric_fn legitimately returns NaN for that replicate
    # (mirroring evaluation.metrics' "omit, don't fabricate" convention) —
    # such replicates are excluded from the interval rather than poisoning it.
    lower_pct = (1 - confidence_level) / 2 * 100
    upper_pct = (1 + confidence_level) / 2 * 100
    ci_lower = float(np.nanpercentile(replicates, lower_pct))
    ci_upper = float(np.nanpercentile(replicates, upper_pct))
    std = float(np.nanstd(replicates, ddof=1))

    return MetricSummary(
        metric_name=metric_name,
        n=n,
        mean=point_estimate,
        std=std,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence_level=confidence_level,
        values=replicates,
    )
