"""Tests for `explainability.llm_summary` — the optional, best-effort LLM
narrative-summary feature. No real network call is made here (that would
cost money and require a real key on every test run); the always-on tests
below cover the "not configured" no-op path and the privacy-critical payload
construction. A separate, manually-invoked live check (not part of this
suite) exercised the real OpenAI API once during Phase 14's follow-up review
— see the session notes; `generate_narrative_summary`'s own contract (never
raises, `None` on any failure) is what every test here actually verifies.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ad_mri.explainability import llm_summary
from ad_mri.explainability.lime_explainer import LimeExplanationResult
from ad_mri.explainability.service import ExplanationBundle, PredictionSummary
from ad_mri.explainability.shap_explainer import ShapExplanationResult


def _bundle(
    *,
    lime: LimeExplanationResult | None = None,
    shap: ShapExplanationResult | None = None,
) -> ExplanationBundle:
    return ExplanationBundle(
        input_id="expl-1",
        image_path="ignored",
        model_version="1.0.0",
        model_config_hash="cfg-hash",
        checkpoint_path="artifacts/checkpoints/model.pt",
        preprocessing_version="1.0.0",
        preprocessing_config_hash="prep-hash",
        class_names=["AD", "CN"],
        prediction=PredictionSummary(
            predicted_class="AD", confidence=0.91, class_probabilities={"AD": 0.91, "CN": 0.09}
        ),
        target_class_index=0,
        lime=lime,
        shap=shap,
        generated_at="2026-01-01T00:00:00+00:00",
        output_dir=Path("artifacts/explanations/expl-1"),
        artifacts={},
    )


def _lime_result() -> LimeExplanationResult:
    return LimeExplanationResult(
        target_class_index=0,
        target_class_name="AD",
        slice_axis=2,
        slice_index=64,
        num_samples=1000,
        segmentation_algorithm="slic",
        num_segments=50,
        top_superpixels=[
            {"segment_id": 7.0, "weight": 0.42},
            {"segment_id": 3.0, "weight": 0.31},
        ],
        local_prediction=0.88,
        surrogate_r2=0.76,
        config_hash="lime-cfg-hash",
    )


def _shap_result() -> ShapExplanationResult:
    return ShapExplanationResult(
        target_class_index=0,
        target_class_name="AD",
        explainer_type="GradientExplainer",
        n_background_samples=20,
        nsamples=200,
        slice_axis=2,
        slice_index=64,
        attribution_shape=(128, 128, 128),
        attribution_sum=2.35,
        config_hash="shap-cfg-hash",
    )


@pytest.mark.asyncio
async def test_no_op_when_openai_api_key_is_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default, and every deployment that never opted in: the feature is
    fully inert, not an error - see `Settings.openai_api_key`'s docstring.
    Patches `get_settings` directly (rather than deleting the env var) since
    this sandbox's own `.env` has a real key configured for the live check
    below - the point of this test is the `openai_api_key is None` branch,
    not this process's actual configuration."""
    from ad_mri.core.config import Settings

    monkeypatch.setattr(llm_summary, "get_settings", lambda: Settings(openai_api_key=None))
    result = await llm_summary.generate_narrative_summary(_bundle())
    assert result is None


def test_payload_never_includes_a_file_path_or_raw_pixel_data() -> None:
    """The privacy-critical guarantee this module exists to keep: the built
    payload is the exhaustive set of what would be sent to the third-party
    API - assert it contains only the numeric/label fields it claims to,
    never `image_path`/`checkpoint_path`/`output_dir` or anything shaped like
    a filesystem path."""
    bundle = _bundle(lime=_lime_result(), shap=_shap_result())
    payload = llm_summary._build_payload(bundle)

    serialized = str(payload)
    assert "ignored" not in serialized  # bundle.image_path's value
    assert "checkpoints" not in serialized  # bundle.checkpoint_path's value
    assert "expl-1" not in serialized  # bundle.input_id

    assert payload["predicted_class"] == "AD"
    assert payload["target_class"] == "AD"
    assert payload["class_probabilities"] == {"AD": 0.91, "CN": 0.09}
    assert payload["lime"]["top_regions"] == [
        {"segment_id": 7, "weight": 0.42},
        {"segment_id": 3, "weight": 0.31},
    ]
    assert payload["shap"]["explainer_type"] == "GradientExplainer"
    assert payload["shap"]["attribution_sum"] == 2.35


def test_payload_omits_lime_and_shap_keys_when_that_method_was_not_run() -> None:
    payload = llm_summary._build_payload(_bundle())
    assert "lime" not in payload
    assert "shap" not in payload


@pytest.mark.asyncio
async def test_returns_none_rather_than_raising_when_the_client_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real API failure (bad key, network error, rate limit, timeout) must
    degrade to `None`, never propagate and fail the /explanations request
    whose actual job already succeeded."""
    from ad_mri.core.config import Settings

    monkeypatch.setattr(
        llm_summary, "get_settings", lambda: Settings(openai_api_key="sk-test-not-a-real-key")
    )

    class _BoomClient:
        def __init__(self, **_kwargs: object) -> None:
            self.chat = self

        class completions:
            @staticmethod
            async def create(**_kwargs: object) -> None:
                raise RuntimeError("simulated network failure")

    import openai

    monkeypatch.setattr(openai, "AsyncOpenAI", _BoomClient)

    result = await llm_summary.generate_narrative_summary(_bundle())
    assert result is None
