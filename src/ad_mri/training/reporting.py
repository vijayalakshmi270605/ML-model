"""Comparison reports for cross-validation and hyperparameter-search results:
a markdown table (per-fold/per-trial metrics, mean +/- std, confidence
intervals) plus a plot, so model-selection results can be reviewed without
re-deriving them from raw MLflow runs. See docs/TECHNICAL_SPECIFICATION.md
§10, §11, and docs/model_selection.md.

`matplotlib` is a training-phase dependency (requirements/train.txt), imported
lazily so this module stays importable without it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ad_mri.core.logging import get_logger
from ad_mri.training.cross_validation import CrossValidationResult
from ad_mri.training.hpo import HPOResult

logger = get_logger(__name__)


def render_cv_report(cv_result: CrossValidationResult, output_dir: Path) -> dict[str, Path]:
    """Write a markdown comparison table (per-fold + mean/std/CI summary) and
    a per-fold bar-chart PNG for a handful of headline metrics.

    Returns `{"markdown": Path, "plot": Path}`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_names = sorted(cv_result.metric_summary)

    lines = ["# Cross-Validation Report", "", f"- Folds: {len(cv_result.fold_results)}", ""]
    lines += ["## Per-fold metrics", ""]
    lines.append("| Fold | " + " | ".join(metric_names) + " |")
    lines.append("|---|" + "|".join(["---"] * len(metric_names)) + "|")
    for fold in cv_result.fold_results:
        row = [f"fold {fold.fold_index}"]
        row += [f"{fold.final_val_metrics.get(name, float('nan')):.4f}" for name in metric_names]
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "## Summary (mean, sample std, 95% CI)",
        "",
        "| Metric | Mean | Std | 95% CI | n |",
        "|---|---|---|---|---|",
    ]
    for name in metric_names:
        summary = cv_result.metric_summary[name]
        std_str = f"{summary.std:.4f}" if summary.std is not None else "n/a (n<2)"
        ci_str = (
            f"[{summary.ci_lower:.4f}, {summary.ci_upper:.4f}]"
            if summary.ci_lower is not None
            else "n/a (n<2)"
        )
        lines.append(f"| {name} | {summary.mean:.4f} | {std_str} | {ci_str} | {summary.n} |")

    markdown_path = output_dir / "cv_report.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")

    plot_path = _plot_cv_summary(cv_result, output_dir / "cv_report.png")
    logger.info("cv_report_written", markdown=str(markdown_path), plot=str(plot_path))
    return {"markdown": markdown_path, "plot": plot_path}


def _plot_cv_summary(cv_result: CrossValidationResult, plot_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_names = sorted(cv_result.metric_summary)
    headline: list[str] = [
        m for m in ("accuracy", "f1_macro", "roc_auc_macro", "roc_auc") if m in metric_names
    ]
    headline = headline or metric_names[:4]

    fig, axes = plt.subplots(1, len(headline), figsize=(4 * len(headline), 4))
    axes_list: list[Any] = list(axes) if len(headline) > 1 else [axes]
    for ax, metric in zip(axes_list, headline, strict=True):
        values = [
            fold.final_val_metrics.get(metric, float("nan")) for fold in cv_result.fold_results
        ]
        ax.bar(range(len(values)), values, color="steelblue")
        summary = cv_result.metric_summary[metric]
        ax.axhline(
            summary.mean, color="darkorange", linestyle="--", label=f"mean={summary.mean:.3f}"
        )
        ax.set_title(metric)
        ax.set_xlabel("Fold")
        ax.set_xticks(range(len(values)))
        ax.legend(fontsize=8)
    fig.suptitle(f"Cross-validation ({len(cv_result.fold_results)} folds)")
    fig.tight_layout()

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    return plot_path


def render_hpo_report(hpo_result: HPOResult, output_dir: Path) -> dict[str, Path]:
    """Write a markdown table ranking every completed trial by its objective
    value (with sampled hyperparameters) and an optimization-history plot.

    Returns `{"markdown": Path, "plot": Path}`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked = sorted(
        hpo_result.trials, key=lambda t: t.metric_value, reverse=(hpo_result.metric_mode == "max")
    )
    param_names = sorted(ranked[0].params) if ranked else []

    lines = [
        f"# Hyperparameter Search Report — {hpo_result.study_name}",
        "",
        f"- Trials completed: {len(hpo_result.trials)}",
        f"- Objective: `{hpo_result.metric_name}` ({hpo_result.metric_mode})",
        f"- Best trial: #{hpo_result.best_trial_number} ({hpo_result.best_metric_value:.4f})",
        "",
        "## Trials, ranked",
        "",
        "| Rank | Trial | " + hpo_result.metric_name + " | " + " | ".join(param_names) + " |",
        "|---|---|---|" + "|".join(["---"] * len(param_names)) + "|",
    ]
    for rank, trial in enumerate(ranked, start=1):
        marker = " **(best)**" if trial.trial_number == hpo_result.best_trial_number else ""
        row = [str(rank), f"#{trial.trial_number}{marker}", f"{trial.metric_value:.4f}"]
        row += [str(trial.params[name]) for name in param_names]
        lines.append("| " + " | ".join(row) + " |")

    markdown_path = output_dir / "hpo_report.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")

    plot_path = _plot_hpo_history(hpo_result, output_dir / "hpo_report.png")
    logger.info("hpo_report_written", markdown=str(markdown_path), plot=str(plot_path))
    return {"markdown": markdown_path, "plot": plot_path}


def _plot_hpo_history(hpo_result: HPOResult, plot_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trial_numbers = [t.trial_number for t in hpo_result.trials]
    values = [t.metric_value for t in hpo_result.trials]

    running_best: list[float] = []
    best_so_far: float | None = None
    for value in values:
        if best_so_far is None:
            best_so_far = value
        elif hpo_result.metric_mode == "max":
            best_so_far = max(best_so_far, value)
        else:
            best_so_far = min(best_so_far, value)
        running_best.append(best_so_far)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(trial_numbers, values, label="trial value", color="steelblue")
    ax.plot(trial_numbers, running_best, label="best so far", color="darkorange")
    ax.set_xlabel("Trial")
    ax.set_ylabel(hpo_result.metric_name)
    ax.set_title(f"HPO optimization history — {hpo_result.study_name}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    return plot_path
