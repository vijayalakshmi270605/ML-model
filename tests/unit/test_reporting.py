"""Tests for training.reporting — comparison report generation.

Uses hand-built result objects (no real training) so the reporting logic
itself is tested in isolation and stays fast.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from ad_mri.evaluation.statistics import summarize_fold_metrics  # noqa: E402
from ad_mri.training.cross_validation import CrossValidationResult, FoldResult  # noqa: E402
from ad_mri.training.hpo import HPOResult, TrialResult  # noqa: E402
from ad_mri.training.reporting import render_cv_report, render_hpo_report  # noqa: E402


def _fake_cv_result() -> CrossValidationResult:
    fold_metrics = [
        {"loss": 0.5, "accuracy": 0.80, "f1_macro": 0.78},
        {"loss": 0.4, "accuracy": 0.85, "f1_macro": 0.83},
        {"loss": 0.6, "accuracy": 0.75, "f1_macro": 0.72},
    ]
    fold_results = [
        FoldResult(
            fold_index=i,
            train_manifest_path=Path(f"fold_{i}/train.csv"),
            val_manifest_path=Path(f"fold_{i}/val.csv"),
            final_val_metrics=metrics,
            best_checkpoint_path=Path(f"fold_{i}/best.pt"),
            mlflow_run_id=None,
        )
        for i, metrics in enumerate(fold_metrics)
    ]
    return CrossValidationResult(
        fold_results=fold_results, metric_summary=summarize_fold_metrics(fold_metrics)
    )


def _fake_hpo_result() -> HPOResult:
    trials = [
        TrialResult(
            trial_number=0,
            params={"learning_rate": 0.001, "batch_size": 8, "optimizer": "adamw"},
            metric_value=0.80,
            cv_result=_fake_cv_result(),
        ),
        TrialResult(
            trial_number=1,
            params={"learning_rate": 0.01, "batch_size": 4, "optimizer": "sgd"},
            metric_value=0.90,
            cv_result=_fake_cv_result(),
        ),
        TrialResult(
            trial_number=2,
            params={"learning_rate": 0.0001, "batch_size": 16, "optimizer": "adam"},
            metric_value=0.70,
            cv_result=_fake_cv_result(),
        ),
    ]
    return HPOResult(
        study_name="test-study",
        metric_name="f1_macro",
        metric_mode="max",
        best_trial_number=1,
        best_metric_value=0.90,
        best_model_config=None,  # type: ignore[arg-type]
        best_training_config=None,  # type: ignore[arg-type]
        trials=trials,
    )


def test_render_cv_report_writes_markdown_and_plot(tmp_path: Path) -> None:
    result = render_cv_report(_fake_cv_result(), tmp_path)
    assert result["markdown"].exists()
    assert result["plot"].exists()
    assert result["plot"].stat().st_size > 0


def test_cv_report_markdown_contains_per_fold_and_summary_rows(tmp_path: Path) -> None:
    result = render_cv_report(_fake_cv_result(), tmp_path)
    text = result["markdown"].read_text(encoding="utf-8")
    assert "fold 0" in text
    assert "fold 1" in text
    assert "fold 2" in text
    assert "Summary" in text
    assert "0.8000" in text  # mean accuracy = (0.80+0.85+0.75)/3 = 0.8


def test_render_hpo_report_writes_markdown_and_plot(tmp_path: Path) -> None:
    result = render_hpo_report(_fake_hpo_result(), tmp_path)
    assert result["markdown"].exists()
    assert result["plot"].exists()
    assert result["plot"].stat().st_size > 0


def test_hpo_report_ranks_trials_best_first(tmp_path: Path) -> None:
    result = render_hpo_report(_fake_hpo_result(), tmp_path)
    text = result["markdown"].read_text(encoding="utf-8")
    # trial #1 (0.90) should be ranked above trial #0 (0.80) above trial #2 (0.70)
    rank_1 = text.index("#1")
    rank_0 = text.index("#0")
    rank_2 = text.index("#2")
    assert rank_1 < rank_0 < rank_2
    assert "(best)" in text
