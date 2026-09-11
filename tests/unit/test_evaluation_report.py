"""Tests for evaluation.report — the EvaluationReport orchestration and its
JSON/CSV/markdown/PNG rendering. Built on synthetic predictions directly (no
model training needed) so the report machinery is tested in isolation from
the CLI's inference loop (see test_evaluate_cli.py for the real end-to-end path)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("matplotlib")

from ad_mri.evaluation.report import evaluate_predictions, render_evaluation_report  # noqa: E402

_Y_TRUE = [0, 0, 0, 1, 1, 1, 1, 1]
_Y_PRED = [0, 0, 1, 1, 1, 1, 1, 0]  # 6/8 correct -> accuracy 0.75
_Y_PROBA = [
    [0.9, 0.1],
    [0.8, 0.2],
    [0.4, 0.6],
    [0.3, 0.7],
    [0.2, 0.8],
    [0.1, 0.9],
    [0.35, 0.65],
    [0.6, 0.4],
]
_CLASS_NAMES = ["CN", "AD"]


def _build_report(**overrides: object):
    defaults: dict[str, object] = dict(
        y_true=_Y_TRUE,
        y_pred=_Y_PRED,
        y_proba=_Y_PROBA,
        class_names=_CLASS_NAMES,
        model_version="1.2.3",
        model_config_hash="abc123",
        checkpoint_path="artifacts/checkpoints/run/best.pt",
        dataset_manifest_path="data/splits/demo/test.csv",
        dataset_manifest_sha256="deadbeef",
        preprocessing_version="1.0.0",
        preprocessing_config_hash="feedface",
        subject_ids=[f"subj-{i}" for i in range(len(_Y_TRUE))],
        n_bootstrap=100,
    )
    defaults.update(overrides)
    return evaluate_predictions(**defaults)  # type: ignore[arg-type]


def test_accuracy_matches_hand_calculation() -> None:
    report = _build_report()
    assert report.metrics["accuracy"] == pytest.approx(0.75)


def test_n_samples_and_class_names_are_recorded() -> None:
    report = _build_report()
    assert report.n_samples == 8
    assert report.class_names == ["CN", "AD"]


def test_confusion_matrix_matches_hand_calculation() -> None:
    # true=0: pred=[0,0,1] -> row [2,1]; true=1: pred=[1,1,1,1,0] -> row [1,4]
    report = _build_report()
    assert report.confusion_matrix == [[2, 1], [1, 4]]


def test_model_and_preprocessing_provenance_is_carried_through() -> None:
    report = _build_report()
    assert report.model_version == "1.2.3"
    assert report.model_config_hash == "abc123"
    assert report.preprocessing_version == "1.0.0"
    assert report.preprocessing_config_hash == "feedface"
    assert report.dataset_manifest_sha256 == "deadbeef"


def test_headline_metrics_have_bootstrap_confidence_intervals() -> None:
    report = _build_report()
    for key in ("accuracy", "f1_macro", "roc_auc"):
        assert key in report.confidence_intervals
        summary = report.confidence_intervals[key]
        assert summary.ci_lower is not None
        assert summary.ci_lower <= report.metrics[key] <= summary.ci_upper  # type: ignore[operator]


def test_threshold_none_by_default_and_recorded_when_given() -> None:
    assert _build_report().threshold is None
    assert _build_report(threshold=0.7).threshold == 0.7


def test_predictions_are_labeled_by_class_name_not_bare_index() -> None:
    report = _build_report()
    first = report.predictions[0]
    assert first["true_label"] in _CLASS_NAMES
    assert first["predicted_label"] in _CLASS_NAMES
    assert first["subject_id"] == "subj-0"
    assert "proba_CN" in first and "proba_AD" in first


def test_classification_report_text_mentions_every_class() -> None:
    report = _build_report()
    for name in _CLASS_NAMES:
        assert name in report.classification_report_text


def test_render_evaluation_report_writes_every_deliverable(tmp_path: Path) -> None:
    report = _build_report()
    written = render_evaluation_report(report, tmp_path / "eval_out")

    for key in (
        "confusion_matrix",
        "roc_curve",
        "pr_curve",
        "classification_report",
        "json",
        "csv",
        "predictions",
        "markdown",
    ):
        assert key in written, f"missing deliverable: {key}"
        assert written[key].exists()
        assert written[key].stat().st_size > 0


def test_json_report_round_trips_the_headline_metrics(tmp_path: Path) -> None:
    report = _build_report()
    written = render_evaluation_report(report, tmp_path / "eval_out")
    payload = json.loads(written["json"].read_text(encoding="utf-8"))

    assert payload["model_version"] == "1.2.3"
    assert payload["metrics"]["accuracy"] == pytest.approx(0.75)
    assert payload["confusion_matrix"] == [[2, 1], [1, 4]]
    assert payload["class_names"] == ["CN", "AD"]


def test_csv_report_has_a_row_per_metric_with_ci_columns(tmp_path: Path) -> None:
    report = _build_report()
    written = render_evaluation_report(report, tmp_path / "eval_out")

    with written["csv"].open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    row_by_metric = {row["metric"]: row for row in rows}
    assert float(row_by_metric["accuracy"]["value"]) == pytest.approx(0.75)
    assert row_by_metric["accuracy"]["ci_lower"] != ""


def test_predictions_csv_has_one_row_per_sample(tmp_path: Path) -> None:
    report = _build_report()
    written = render_evaluation_report(report, tmp_path / "eval_out")
    with written["predictions"].open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 8
    assert rows[0]["subject_id"] == "subj-0"


def test_markdown_report_documents_class_labels(tmp_path: Path) -> None:
    report = _build_report()
    written = render_evaluation_report(report, tmp_path / "eval_out")
    text = written["markdown"].read_text(encoding="utf-8")
    assert "0=CN" in text
    assert "1=AD" in text
    assert "Evaluation Report" in text


def test_render_is_deterministic_given_the_same_report(tmp_path: Path) -> None:
    report = _build_report()
    written_a = render_evaluation_report(report, tmp_path / "run_a")
    written_b = render_evaluation_report(report, tmp_path / "run_b")
    assert written_a["json"].read_text() == written_b["json"].read_text()
    assert written_a["csv"].read_text() == written_b["csv"].read_text()
