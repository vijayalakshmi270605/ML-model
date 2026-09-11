"""Integration tests for explainability.shap_explainer — real SHAP runs (no
mocks) against a real, tiny 3D CNN.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("shap")

from ad_mri.core.exceptions import ConfigurationError  # noqa: E402
from ad_mri.explainability.config import ShapConfig  # noqa: E402
from ad_mri.explainability.shap_explainer import (  # noqa: E402
    compute_global_shap_importance,
    explain_with_shap,
)
from ad_mri.modeling.checkpoint import load_checkpoint  # noqa: E402

_TEST_SHAP_CONFIG = ShapConfig(n_background_samples=4, nsamples=10, seed=42)


@pytest.fixture
def model(checkpoint: tuple[Path, Path]) -> object:
    checkpoint_path, _preprocessing_path = checkpoint
    model, _model_config, _metadata = load_checkpoint(checkpoint_path)
    model.eval()
    return model


def test_shap_attribution_matches_the_volume_spatial_shape(tmp_path: Path, model: object) -> None:
    volume = torch.rand(1, 16, 16, 16)
    background = torch.rand(4, 1, 16, 16, 16)
    result = explain_with_shap(
        model,
        volume,
        ["AD", "CN"],
        target_class_index=0,
        background=background,
        config=_TEST_SHAP_CONFIG,
        output_dir=tmp_path,
        device=torch.device("cpu"),
    )
    assert result.attribution_shape == (16, 16, 16)


def test_shap_writes_values_and_overlay_artifacts(tmp_path: Path, model: object) -> None:
    volume = torch.rand(1, 16, 16, 16)
    background = torch.rand(4, 1, 16, 16, 16)
    result = explain_with_shap(
        model,
        volume,
        ["AD", "CN"],
        target_class_index=0,
        background=background,
        config=_TEST_SHAP_CONFIG,
        output_dir=tmp_path,
        device=torch.device("cpu"),
    )
    values_path = result.artifacts["shap_values"]
    overlay_path = result.artifacts["shap_overlay"]
    assert values_path.exists() and values_path.stat().st_size > 0
    assert overlay_path.exists() and overlay_path.stat().st_size > 0


def test_shap_saved_values_match_the_reported_sum(tmp_path: Path, model: object) -> None:
    import numpy as np

    volume = torch.rand(1, 16, 16, 16)
    background = torch.rand(4, 1, 16, 16, 16)
    result = explain_with_shap(
        model,
        volume,
        ["AD", "CN"],
        target_class_index=0,
        background=background,
        config=_TEST_SHAP_CONFIG,
        output_dir=tmp_path,
        device=torch.device("cpu"),
    )
    saved = np.load(result.artifacts["shap_values"])
    assert saved.shape == (16, 16, 16)
    assert float(saved.sum()) == pytest.approx(result.attribution_sum, abs=1e-4)


def test_shap_config_hash_and_background_size_are_recorded(tmp_path: Path, model: object) -> None:
    volume = torch.rand(1, 16, 16, 16)
    background = torch.rand(4, 1, 16, 16, 16)
    result = explain_with_shap(
        model,
        volume,
        ["AD", "CN"],
        target_class_index=0,
        background=background,
        config=_TEST_SHAP_CONFIG,
        output_dir=tmp_path,
        device=torch.device("cpu"),
    )
    assert result.config_hash == _TEST_SHAP_CONFIG.compute_hash()
    assert result.n_background_samples == 4


def test_shap_rejects_unknown_explainer_type(tmp_path: Path, model: object) -> None:
    volume = torch.rand(1, 16, 16, 16)
    background = torch.rand(4, 1, 16, 16, 16)
    bad_config = _TEST_SHAP_CONFIG.model_copy(update={"explainer": "not-a-real-explainer"})
    with pytest.raises(ConfigurationError, match="Unknown SHAP explainer"):
        explain_with_shap(
            model,
            volume,
            ["AD", "CN"],
            target_class_index=0,
            background=background,
            config=bad_config,
            output_dir=tmp_path,
            device=torch.device("cpu"),
        )


def test_global_shap_importance_produces_a_downsampled_grid(tmp_path: Path, model: object) -> None:
    samples = torch.rand(3, 1, 16, 16, 16)
    background = torch.rand(4, 1, 16, 16, 16)
    result = compute_global_shap_importance(
        model,
        samples,
        background,
        ["AD", "CN"],
        target_class_index=0,
        config=_TEST_SHAP_CONFIG,
        output_dir=tmp_path,
        device=torch.device("cpu"),
        region_pool_size=4,
    )
    # 16 / 4 = 4 along each spatial axis — strictly smaller than the 16^3 input
    assert result.region_grid_shape == (4, 4, 4)
    assert result.n_samples_used == 3


def test_global_shap_importance_writes_artifacts(tmp_path: Path, model: object) -> None:
    samples = torch.rand(3, 1, 16, 16, 16)
    background = torch.rand(4, 1, 16, 16, 16)
    result = compute_global_shap_importance(
        model,
        samples,
        background,
        ["AD", "CN"],
        target_class_index=1,
        config=_TEST_SHAP_CONFIG,
        output_dir=tmp_path,
        device=torch.device("cpu"),
        region_pool_size=4,
    )
    grid_path = result.artifacts["shap_global_importance"]
    overlay_path = result.artifacts["shap_global_importance_overlay"]
    assert grid_path.exists() and overlay_path.exists()


def test_global_shap_importance_values_are_non_negative(tmp_path: Path, model: object) -> None:
    """Aggregated importance is a mean of |SHAP value| — never negative,
    unlike a single local attribution."""
    import numpy as np

    samples = torch.rand(3, 1, 16, 16, 16)
    background = torch.rand(4, 1, 16, 16, 16)
    result = compute_global_shap_importance(
        model,
        samples,
        background,
        ["AD", "CN"],
        target_class_index=0,
        config=_TEST_SHAP_CONFIG,
        output_dir=tmp_path,
        device=torch.device("cpu"),
        region_pool_size=4,
    )
    grid = np.load(result.artifacts["shap_global_importance"])
    assert (grid >= 0).all()
