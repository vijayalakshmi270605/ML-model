"""Tests for `api.audit` - the best-effort persistence boundary. Uses a real
SQLite database wired in through the same `db.session.get_sessionmaker()`
these functions call, so the "actually writes when the database is
reachable" half is genuinely exercised, not assumed. The "never raises when
it isn't" half points `DATABASE_URL` at a nonexistent path with an unwritable
parent instead of mocking anything."""

from __future__ import annotations

from pathlib import Path

import pytest

from ad_mri.api import audit
from ad_mri.core.config import get_settings
from ad_mri.db.base import Base
from ad_mri.db.models import ExplanationRecord, ModelVersionRecord, PredictionRequestRecord
from ad_mri.db.session import get_engine, get_sessionmaker, reset_engine_cache
from ad_mri.explainability.service import ExplanationBundle, PredictionSummary
from ad_mri.inference.predictor import ModelVersionInfo, PredictionResult


def _model_info(**overrides: object) -> ModelVersionInfo:
    defaults = dict(
        architecture="ad-mri-convnet3d",
        model_version="1.0.0",
        model_config_hash="cfg-hash",
        preprocessing_version="1.0.0",
        preprocessing_config_hash="prep-hash",
        calibration_method="none",
        device="cpu",
        checkpoint_path="artifacts/checkpoints/model.pt",
        loaded_at="2026-01-01T00:00:00+00:00",
    )
    defaults.update(overrides)
    return ModelVersionInfo(**defaults)  # type: ignore[arg-type]


def _prediction_result(**overrides: object) -> PredictionResult:
    defaults = dict(
        request_id="external-request-id",
        model_version="1.0.0",
        preprocessing_version="1.0.0",
        prediction="AD",
        confidence=0.88,
        probabilities={"AD": 0.88, "CN": 0.12},
        processing_time_ms=42.0,
        explanation_available=False,
        explanation_id=None,
        timestamp="2026-01-01T00:00:01+00:00",
    )
    defaults.update(overrides)
    return PredictionResult(**defaults)  # type: ignore[arg-type]


@pytest.fixture
def real_sqlite_database(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Points the process-wide DB engine at a real, schema-migrated SQLite
    file for the duration of one test."""
    db_path = tmp_path / "audit.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    get_settings.cache_clear()
    reset_engine_cache()
    Base.metadata.create_all(get_engine())
    yield
    reset_engine_cache()
    get_settings.cache_clear()


class TestRecordPrediction:
    def test_writes_a_real_row_when_the_database_is_reachable(
        self, real_sqlite_database: None
    ) -> None:
        result = _prediction_result()
        audit.record_prediction(
            result=result,
            model_info=_model_info(),
            image_ref="0123456789abcdef",
            content_type="application/octet-stream",
            api_key="a-real-key",
        )

        session = get_sessionmaker()()
        try:
            rows = session.query(PredictionRequestRecord).all()
            assert len(rows) == 1
            assert rows[0].external_request_id == "external-request-id"
        finally:
            session.close()

    def test_never_raises_when_the_database_is_unreachable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # A sqlite path under a directory that does not exist - opening a
        # connection to it fails for real, no mock involved.
        bad_path = tmp_path / "missing-dir" / "x" / "db.sqlite"
        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{bad_path}")
        get_settings.cache_clear()
        reset_engine_cache()
        try:
            audit.record_prediction(
                result=_prediction_result(),
                model_info=_model_info(),
                image_ref="0123456789abcdef",
                content_type=None,
                api_key=None,
            )  # must not raise
        finally:
            reset_engine_cache()
            get_settings.cache_clear()

    def test_is_a_no_op_when_audit_is_disabled(
        self, monkeypatch: pytest.MonkeyPatch, real_sqlite_database: None
    ) -> None:
        monkeypatch.setenv("AUDIT_ENABLED", "false")
        get_settings.cache_clear()
        audit.record_prediction(
            result=_prediction_result(),
            model_info=_model_info(),
            image_ref="0123456789abcdef",
            content_type=None,
            api_key=None,
        )
        session = get_sessionmaker()()
        try:
            assert session.query(PredictionRequestRecord).count() == 0
        finally:
            session.close()
        get_settings.cache_clear()


class TestRecordExplanation:
    def test_writes_a_real_row_with_the_manifest_uri(
        self, real_sqlite_database: None, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "explanations" / "expl-1"
        output_dir.mkdir(parents=True)
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text("{}", encoding="utf-8")

        bundle = ExplanationBundle(
            input_id="expl-1",
            image_path="ignored",
            model_version="1.0.0",
            model_config_hash="cfg-hash",
            checkpoint_path="artifacts/checkpoints/model.pt",
            preprocessing_version="1.0.0",
            preprocessing_config_hash="prep-hash",
            class_names=["AD", "CN"],
            prediction=PredictionSummary(
                predicted_class="AD", confidence=0.9, class_probabilities={"AD": 0.9, "CN": 0.1}
            ),
            target_class_index=0,
            lime=None,
            shap=None,
            generated_at="2026-01-01T00:00:00+00:00",
            output_dir=output_dir,
            artifacts={"manifest": manifest_path},
        )

        audit.record_explanation(
            bundle=bundle, model_info=_model_info(), methods="lime", target_class=None
        )

        session = get_sessionmaker()()
        try:
            rows = session.query(ExplanationRecord).all()
            assert len(rows) == 1
            assert rows[0].external_explanation_id == "expl-1"
            assert rows[0].artifact_uri.endswith("manifest.json")
            assert rows[0].request_id is None
        finally:
            session.close()


class TestRecordModelVersionSeen:
    def test_registers_the_model_version(self, real_sqlite_database: None) -> None:
        audit.record_model_version_seen(_model_info())
        session = get_sessionmaker()()
        try:
            rows = session.query(ModelVersionRecord).all()
            assert len(rows) == 1
            assert rows[0].model_config_hash == "cfg-hash"
        finally:
            session.close()


class TestRecordEvent:
    def test_writes_a_system_event(self, real_sqlite_database: None) -> None:
        audit.record_event(event_type="rate_limit_exceeded", severity="warning", detail={"n": 1})
        from ad_mri.db.models import SystemEventRecord

        session = get_sessionmaker()()
        try:
            rows = session.query(SystemEventRecord).all()
            assert len(rows) == 1
            assert rows[0].event_type == "rate_limit_exceeded"
        finally:
            session.close()
