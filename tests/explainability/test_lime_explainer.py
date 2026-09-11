"""Integration tests for explainability.lime_explainer — real LIME runs (no
mocks) against a real, tiny 3D CNN.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lime")
pytest.importorskip("skimage")

from ad_mri.core.exceptions import ConfigurationError  # noqa: E402
from ad_mri.explainability.config import LimeConfig  # noqa: E402
from ad_mri.explainability.lime_explainer import explain_with_lime  # noqa: E402
from ad_mri.modeling.checkpoint import load_checkpoint  # noqa: E402

# Tuned quickshift parameters that reliably produce several superpixels on a
# tiny 16x16 test slice — the module default (kernel_size=4) is tuned for a
# realistic MRI slice and collapses a 16x16 image to ~1 superpixel.
_TEST_LIME_CONFIG = LimeConfig(
    num_samples=40,
    batch_size=8,
    num_features=5,
    quickshift_kernel_size=2.0,
    quickshift_max_dist=10.0,
    quickshift_ratio=0.2,
    seed=42,
)


def _structured_volume() -> object:
    """A deterministic 16^3 volume with four distinct-intensity quadrants (in
    every axis-aligned mid-slice) plus a little noise — unlike flat random
    noise, this reliably gives quickshift real structure to segment,
    regardless of the RNG draw."""
    generator = torch.Generator().manual_seed(0)
    volume = torch.rand(1, 16, 16, 16, generator=generator) * 0.1
    volume[0, :8, :8, :] += 0.0
    volume[0, :8, 8:, :] += 0.3
    volume[0, 8:, :8, :] += 0.6
    volume[0, 8:, 8:, :] += 0.9
    return volume


@pytest.fixture
def model_and_volume(checkpoint: tuple[Path, Path]) -> tuple[object, object]:
    checkpoint_path, _preprocessing_path = checkpoint
    model, _model_config, _metadata = load_checkpoint(checkpoint_path)
    model.eval()
    return model, _structured_volume()


def test_lime_explanation_produces_more_than_one_superpixel(
    tmp_path: Path, model_and_volume: tuple[object, object]
) -> None:
    model, volume = model_and_volume
    result = explain_with_lime(
        model,
        volume,
        ["AD", "CN"],
        target_class_index=0,
        config=_TEST_LIME_CONFIG,
        output_dir=tmp_path,
        device=torch.device("cpu"),
    )
    assert result.num_segments > 1
    assert len(result.top_superpixels) > 1


def test_lime_result_identifies_the_requested_target_class(
    tmp_path: Path, model_and_volume: tuple[object, object]
) -> None:
    model, volume = model_and_volume
    result = explain_with_lime(
        model,
        volume,
        ["AD", "CN"],
        target_class_index=1,
        config=_TEST_LIME_CONFIG,
        output_dir=tmp_path,
        device=torch.device("cpu"),
    )
    assert result.target_class_index == 1
    assert result.target_class_name == "CN"


def test_lime_writes_an_overlay_artifact(
    tmp_path: Path, model_and_volume: tuple[object, object]
) -> None:
    model, volume = model_and_volume
    result = explain_with_lime(
        model,
        volume,
        ["AD", "CN"],
        target_class_index=0,
        config=_TEST_LIME_CONFIG,
        output_dir=tmp_path,
        device=torch.device("cpu"),
    )
    overlay_path = result.artifacts["lime_overlay"]
    assert overlay_path.exists()
    assert overlay_path.stat().st_size > 0


def test_lime_config_hash_is_recorded_on_the_result(
    tmp_path: Path, model_and_volume: tuple[object, object]
) -> None:
    model, volume = model_and_volume
    result = explain_with_lime(
        model,
        volume,
        ["AD", "CN"],
        target_class_index=0,
        config=_TEST_LIME_CONFIG,
        output_dir=tmp_path,
        device=torch.device("cpu"),
    )
    assert result.config_hash == _TEST_LIME_CONFIG.compute_hash()


def test_lime_is_reproducible_given_the_same_seed(
    tmp_path: Path, model_and_volume: tuple[object, object]
) -> None:
    model, volume = model_and_volume
    result_a = explain_with_lime(
        model,
        volume,
        ["AD", "CN"],
        target_class_index=0,
        config=_TEST_LIME_CONFIG,
        output_dir=tmp_path / "a",
        device=torch.device("cpu"),
    )
    result_b = explain_with_lime(
        model,
        volume,
        ["AD", "CN"],
        target_class_index=0,
        config=_TEST_LIME_CONFIG,
        output_dir=tmp_path / "b",
        device=torch.device("cpu"),
    )
    assert result_a.top_superpixels == result_b.top_superpixels
    assert result_a.local_prediction == result_b.local_prediction
    assert result_a.surrogate_r2 == result_b.surrogate_r2


def test_lime_top_superpixels_are_ranked_by_absolute_weight(
    tmp_path: Path, model_and_volume: tuple[object, object]
) -> None:
    model, volume = model_and_volume
    result = explain_with_lime(
        model,
        volume,
        ["AD", "CN"],
        target_class_index=0,
        config=_TEST_LIME_CONFIG,
        output_dir=tmp_path,
        device=torch.device("cpu"),
    )
    weights = [abs(item["weight"]) for item in result.top_superpixels]
    assert weights == sorted(weights, reverse=True)


def test_lime_num_features_caps_the_reported_superpixels(
    tmp_path: Path, model_and_volume: tuple[object, object]
) -> None:
    model, volume = model_and_volume
    small_config = _TEST_LIME_CONFIG.model_copy(update={"num_features": 2})
    result = explain_with_lime(
        model,
        volume,
        ["AD", "CN"],
        target_class_index=0,
        config=small_config,
        output_dir=tmp_path,
        device=torch.device("cpu"),
    )
    assert len(result.top_superpixels) <= 2


def test_lime_rejects_multi_channel_volume(
    tmp_path: Path, model_and_volume: tuple[object, object]
) -> None:
    model, _volume = model_and_volume
    multi_channel_volume = torch.rand(2, 16, 16, 16)
    with pytest.raises(ConfigurationError, match="single-channel"):
        explain_with_lime(
            model,
            multi_channel_volume,
            ["AD", "CN"],
            target_class_index=0,
            config=_TEST_LIME_CONFIG,
            output_dir=tmp_path,
            device=torch.device("cpu"),
        )
