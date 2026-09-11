"""SHAP-based explanations (expected-gradients) for individual predictions,
plus an aggregated/global importance summary across multiple samples.

Uses `shap.GradientExplainer` — an expected-gradients method that computes
attributions directly from the model's gradients w.r.t. the **full input
tensor** (the entire preprocessed volume, every voxel, in one pass per
background/interpolation sample) rather than any hand-picked 2D projection or
a flattened feature vector. This is the "appropriate SHAP method for the CNN
architecture" requirement: gradient-based explainers scale to CNNs operating
on high-dimensional image/volume input in a way perturbation-based explainers
(KernelExplainer) do not — no combinatorial feature-flattening is involved.

`shap`/`torch` are explainability-phase dependencies (requirements/train.txt
via the `train` extra), imported lazily so this module stays importable
without them. See docs/TECHNICAL_SPECIFICATION.md §7, docs/explainability.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ad_mri.core.exceptions import ConfigurationError
from ad_mri.core.logging import get_logger
from ad_mri.explainability.config import ShapConfig
from ad_mri.explainability.overlays import render_global_importance_overlay, render_shap_overlay
from ad_mri.explainability.slicing import extract_slice

logger = get_logger(__name__)


@dataclass(frozen=True)
class ShapExplanationResult:
    """One instance's SHAP attribution. The full per-voxel attribution array
    is written to `artifacts["shap_values"]` (a `.npy` file) — not embedded
    inline, since even a modest MRI volume is too large for a JSON field."""

    target_class_index: int
    target_class_name: str
    explainer_type: str
    n_background_samples: int
    nsamples: int
    slice_axis: int
    slice_index: int
    attribution_shape: tuple[int, ...]
    #: Sum of per-voxel attributions for the target class — a diagnostic, not
    #: a guaranteed completeness/additivity identity (GradientExplainer's
    #: expected-gradients approximation does not provide the exact
    #: prediction-minus-base-rate guarantee DeepExplainer/KernelExplainer do
    #: — see docs/explainability.md "Limitations").
    attribution_sum: float
    config_hash: str
    artifacts: dict[str, Path] = field(default_factory=dict)


@dataclass(frozen=True)
class GlobalShapImportanceResult:
    """Aggregated importance across multiple samples for one target class —
    "technically meaningful" only in this multi-sample, downsampled-region
    form (see module docstring and docs/explainability.md "Limitations" for
    why a per-voxel global ranking is not)."""

    target_class_index: int
    target_class_name: str
    n_samples_used: int
    region_pool_size: int
    region_grid_shape: tuple[int, ...]
    config_hash: str
    artifacts: dict[str, Path] = field(default_factory=dict)


def _build_explainer(model: Any, background: Any, explainer_type: str) -> Any:
    import shap

    if explainer_type == "gradient":
        return shap.GradientExplainer(model, background)
    if explainer_type == "deep":
        return shap.DeepExplainer(model, background)
    raise ConfigurationError(f"Unknown SHAP explainer type: {explainer_type!r}")  # pragma: no cover


def _per_class_attribution(shap_values: Any, target_class_index: int) -> Any:
    """`GradientExplainer`/`DeepExplainer` return one array shaped
    `(n_instances, *input_shape, n_classes)` — slice out one instance's one
    target class, dropping the leading instance and channel axes."""
    import numpy as np

    array = np.asarray(shap_values)
    return array[0, 0, ..., target_class_index]  # -> (*spatial_dims,)


def explain_with_shap(
    model: Any,
    volume: Any,
    class_names: list[str],
    target_class_index: int,
    background: Any,
    config: ShapConfig,
    output_dir: Path,
    device: Any,
) -> ShapExplanationResult:
    """Generate one SHAP explanation for `volume` (a single preprocessed,
    channel-first sample) for `target_class_index`, against `background`
    (a stack of preprocessed reference samples — see
    `explainability.sampling.load_volume_batch`).

    Reproducible given identical `model`/`volume`/`background`/`config`
    (`shap.GradientExplainer` draws its interpolation points from a seeded
    RNG when `config.seed` is set globally before this call — see
    `core.seeding.set_global_seed`, used by the service).

    Raises:
        ConfigurationError: `config.explainer` is not `"gradient"`/`"deep"`.
    """
    import numpy as np
    import torch

    model = model.to(device)
    background = background.to(device)
    instance = volume.unsqueeze(0).to(device)  # add the batch dim GradientExplainer expects

    explainer = _build_explainer(model, background, config.explainer)
    with torch.enable_grad():  # GradientExplainer needs gradients even though we're not training
        shap_values = explainer.shap_values(instance, nsamples=config.nsamples)
    attribution = _per_class_attribution(shap_values, target_class_index)

    spatial_dims = volume.ndim - 1
    raw_slice, resolved_index = extract_slice(
        volume, spatial_dims, config.slice_axis, config.slice_index
    )
    attribution_slice = (
        attribution
        if spatial_dims == 2
        else np.take(attribution, resolved_index, axis=config.slice_axis)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    values_path = output_dir / "shap_values.npy"
    np.save(values_path, attribution)
    overlay_path = render_shap_overlay(
        raw_slice.detach().cpu().numpy(), attribution_slice, output_dir / "shap_overlay.png"
    )

    logger.info(
        "shap_explanation_generated",
        target_class=class_names[target_class_index],
        explainer=config.explainer,
        n_background_samples=background.shape[0],
        attribution_sum=float(attribution.sum()),
    )

    return ShapExplanationResult(
        target_class_index=target_class_index,
        target_class_name=class_names[target_class_index],
        explainer_type=config.explainer,
        n_background_samples=int(background.shape[0]),
        nsamples=config.nsamples,
        slice_axis=config.slice_axis,
        slice_index=resolved_index,
        attribution_shape=tuple(attribution.shape),
        attribution_sum=float(attribution.sum()),
        config_hash=config.compute_hash(),
        artifacts={"shap_values": values_path, "shap_overlay": overlay_path},
    )


def compute_global_shap_importance(
    model: Any,
    samples: Any,
    background: Any,
    class_names: list[str],
    target_class_index: int,
    config: ShapConfig,
    output_dir: Path,
    device: Any,
    region_pool_size: int = 4,
) -> GlobalShapImportanceResult:
    """Aggregate |SHAP value| across `samples` (a batch of preprocessed
    volumes, e.g. a slice of the validation set) into one **downsampled
    region-importance grid** for `target_class_index`.

    Global importance for a per-voxel MRI attribution is deliberately *not*
    reported as a flat ranked list of voxels — with millions of voxels and
    only a handful of samples, such a ranking is not statistically meaningful
    (see docs/explainability.md "Limitations"). Average-pooling to a coarse
    `region_pool_size`-voxel grid trades per-voxel precision for a summary
    that is stable enough to interpret across samples.
    """
    import numpy as np
    import torch
    import torch.nn.functional as functional

    model = model.to(device)
    background = background.to(device)
    batch = samples.to(device)

    explainer = _build_explainer(model, background, config.explainer)
    with torch.enable_grad():
        shap_values = explainer.shap_values(batch, nsamples=config.nsamples)
    array = np.asarray(shap_values)[:, 0, ..., target_class_index]  # (n_samples, *spatial)

    abs_attribution = torch.from_numpy(np.abs(array)).float()
    spatial_dims = abs_attribution.ndim - 1
    pool_fn = functional.avg_pool3d if spatial_dims == 3 else functional.avg_pool2d
    # avg_pool expects a channel dim: (n_samples, 1, *spatial)
    pooled_per_sample = pool_fn(abs_attribution.unsqueeze(1), kernel_size=region_pool_size)
    mean_region_importance = pooled_per_sample.mean(dim=0).squeeze(0).numpy()  # (*region_grid)

    output_dir.mkdir(parents=True, exist_ok=True)
    grid_path = output_dir / "shap_global_importance.npy"
    np.save(grid_path, mean_region_importance)

    representative_axis = config.slice_axis if spatial_dims == 3 else 0
    resolved_index = mean_region_importance.shape[representative_axis] // 2
    region_slice = (
        mean_region_importance
        if spatial_dims == 2
        else np.take(mean_region_importance, resolved_index, axis=representative_axis)
    )
    overlay_path = render_global_importance_overlay(
        region_slice, output_dir / "shap_global_importance.png"
    )

    logger.info(
        "shap_global_importance_computed",
        target_class=class_names[target_class_index],
        n_samples_used=int(batch.shape[0]),
        region_grid_shape=tuple(mean_region_importance.shape),
    )

    return GlobalShapImportanceResult(
        target_class_index=target_class_index,
        target_class_name=class_names[target_class_index],
        n_samples_used=int(batch.shape[0]),
        region_pool_size=region_pool_size,
        region_grid_shape=tuple(mean_region_importance.shape),
        config_hash=config.compute_hash(),
        artifacts={
            "shap_global_importance": grid_path,
            "shap_global_importance_overlay": overlay_path,
        },
    )
