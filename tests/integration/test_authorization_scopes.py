"""Integration tests for Phase 14's authorization architecture
(`api.deps.require_scope`, `Settings.api_keys`) - real HTTP requests against
a real FastAPI app + a real, tiny checkpoint, no mocks."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.helpers import write_synthetic_nifti


def _upload(image_path: Path) -> dict[str, tuple[str, bytes, str]]:
    return {"file": (image_path.name, image_path.read_bytes(), "application/octet-stream")}


@pytest.fixture
def multi_key_client(
    monkeypatch: pytest.MonkeyPatch, checkpoint: tuple[Path, Path], tmp_path: Path
) -> TestClient:
    checkpoint_path, preprocessing_path = checkpoint
    monkeypatch.setenv("INFERENCE_MODEL_PATH", str(checkpoint_path))
    monkeypatch.setenv("INFERENCE_PREPROCESSING_CONFIG_PATH", str(preprocessing_path))
    monkeypatch.setenv("API_KEY_REQUIRED", "true")
    monkeypatch.setenv(
        "API_KEYS",
        '{"predict-only-key": ["predict"], "clinician-key": ["predict", "label"]}',
    )
    # A real, reachable database: /monitoring/ground-truth (unlike
    # /predictions'/'explanations' best-effort audit writes) is a *loud* DB
    # read+write - it must look up the given request_id to answer
    # authoritatively whether it exists, so this endpoint's own tests need a
    # database that's actually there to look it up in, not the default
    # (unreachable in this sandbox, see docs/database.md §9) PostgreSQL.
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'authz.db'}")

    from ad_mri.db.base import Base
    from ad_mri.db.session import get_engine

    Base.metadata.create_all(get_engine())

    from ad_mri.api.main import create_app

    with TestClient(create_app()) as client:
        yield client


class TestPredictScope:
    def test_a_predict_only_key_can_run_predictions(
        self, multi_key_client: TestClient, tmp_path: Path
    ) -> None:
        image = write_synthetic_nifti(tmp_path / "scan.nii.gz", shape=(16, 16, 16))
        response = multi_key_client.post(
            "/api/v1/predictions",
            files=_upload(image),
            headers={"X-API-Key": "predict-only-key"},
        )
        assert response.status_code == 201

    def test_an_unknown_key_is_401_not_403(
        self, multi_key_client: TestClient, tmp_path: Path
    ) -> None:
        image = write_synthetic_nifti(tmp_path / "scan.nii.gz", shape=(16, 16, 16))
        response = multi_key_client.post(
            "/api/v1/predictions", files=_upload(image), headers={"X-API-Key": "not-a-real-key"}
        )
        assert response.status_code == 401

    def test_no_key_at_all_is_401(self, multi_key_client: TestClient, tmp_path: Path) -> None:
        image = write_synthetic_nifti(tmp_path / "scan.nii.gz", shape=(16, 16, 16))
        response = multi_key_client.post("/api/v1/predictions", files=_upload(image))
        assert response.status_code == 401


class TestLabelScope:
    def test_a_predict_only_key_is_forbidden_from_recording_ground_truth(
        self, multi_key_client: TestClient
    ) -> None:
        """A real, meaningful 403 - this key genuinely authenticated, it
        just isn't authorized for this more sensitive capability. Distinct
        from a 401 (see the unknown-key test above)."""
        response = multi_key_client.post(
            "/api/v1/monitoring/ground-truth",
            json={"request_id": "whatever", "true_class": "AD", "label_source": "clinician_review"},
            headers={"X-API-Key": "predict-only-key"},
        )
        assert response.status_code == 403

    def test_a_clinician_key_can_reach_the_label_scope_check(
        self, multi_key_client: TestClient
    ) -> None:
        """Passes authorization (gets past the 401/403 checks); the 404 that
        follows is this endpoint's own, separate "no such prediction was
        ever recorded" domain error for a request_id that was never real -
        proof the scope check itself let this key through."""
        response = multi_key_client.post(
            "/api/v1/monitoring/ground-truth",
            json={
                "request_id": "never-actually-recorded",
                "true_class": "AD",
                "label_source": "clinician_review",
            },
            headers={"X-API-Key": "clinician-key"},
        )
        assert response.status_code == 404


class TestLegacySingleKeyBackwardCompatibility:
    def test_the_legacy_single_key_still_holds_every_scope(
        self, monkeypatch: pytest.MonkeyPatch, checkpoint: tuple[Path, Path], tmp_path: Path
    ) -> None:
        """No API_KEYS configured at all - Settings.api_keys is None, so
        require_scope falls all the way back to the pre-Phase-14
        single-API_KEY behavior for every scope, unchanged."""
        checkpoint_path, preprocessing_path = checkpoint
        monkeypatch.setenv("INFERENCE_MODEL_PATH", str(checkpoint_path))
        monkeypatch.setenv("INFERENCE_PREPROCESSING_CONFIG_PATH", str(preprocessing_path))
        monkeypatch.setenv("API_KEY_REQUIRED", "true")
        monkeypatch.setenv("API_KEY", "the-one-legacy-key")
        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'authz_legacy.db'}")

        from ad_mri.db.base import Base
        from ad_mri.db.session import get_engine

        Base.metadata.create_all(get_engine())

        from ad_mri.api.main import create_app

        with TestClient(create_app()) as client:
            image = write_synthetic_nifti(tmp_path / "scan.nii.gz", shape=(16, 16, 16))
            predict_response = client.post(
                "/api/v1/predictions",
                files=_upload(image),
                headers={"X-API-Key": "the-one-legacy-key"},
            )
            assert predict_response.status_code == 201

            label_response = client.post(
                "/api/v1/monitoring/ground-truth",
                json={
                    "request_id": "whatever",
                    "true_class": "AD",
                    "label_source": "clinician_review",
                },
                headers={"X-API-Key": "the-one-legacy-key"},
            )
            # 404 (unknown request_id), not 401/403 - the legacy key was
            # authorized for the "label" scope too.
            assert label_response.status_code == 404
