"""LIME-based local explanations for individual predictions.

Applies `lime`'s `LimeImageExplainer` — designed for 2D images — to one
representative 2D slice of the (possibly 3D) preprocessed volume, perturbing
only that slice's superpixels while every other voxel stays the real,
preprocessed input; each perturbed candidate is scored by a genuine forward
pass through the real 3D model (see `explainability.slicing` for the full
rationale, and docs/explainability.md "Limitations" for what this does and
does not tell you about the rest of the volume).

`lime`/`torch`/`skimage` are explainability-phase dependencies
(requirements/train.txt via the `train` extra), imported lazily so this
module stays importable without them. See docs/TECHNICAL_SPECIFICATION.md §7.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ad_mri.core.logging import get_logger
from ad_mri.explainability.config import LimeConfig
from ad_mri.explainability.overlays import render_lime_overlay
from ad_mri.explainability.slicing import extract_slice, insert_slice

logger = get_logger(__name__)


@dataclass(frozen=True)
class LimeExplanationResult:
    """Everything about one LIME run: which class/slice it explains, the
    regions it found, and diagnostics for judging how much to trust it."""

    target_class_index: int
    target_class_name: str
    slice_axis: int
    slice_index: int
    num_samples: int
    segmentation_algorithm: str
    num_segments: int
    #: Superpixels ranked by |weight|, most influential first —
    #: "identify image regions/features contributing to the prediction".
    top_superpixels: list[dict[str, float]]
    #: The local surrogate (Ridge) model's own prediction for the *unperturbed*
    #: slice, for the target class — compare against the real model's
    #: probability (in the prediction bundle) as a faithfulness sanity check.
    local_prediction: float
    #: R^2 of the local surrogate model fit — low values mean the linear
    #: surrogate fits the neighborhood poorly and the explanation should be
    #: trusted less (see docs/explainability.md "Limitations").
    surrogate_r2: float
    config_hash: str
    artifacts: dict[str, Path] = field(default_factory=dict)


def _build_segmentation_fn(config: LimeConfig) -> Any:
    from lime.wrappers.scikit_image import SegmentationAlgorithm

    if config.segmentation_algorithm == "quickshift":
        return SegmentationAlgorithm(
            "quickshift",
            kernel_size=config.quickshift_kernel_size,
            max_dist=config.quickshift_max_dist,
            ratio=config.quickshift_ratio,
            random_seed=config.seed,
        )
    return SegmentationAlgorithm(
        "slic",
        n_segments=config.slic_n_segments,
        compactness=config.slic_compactness,
        random_seed=config.seed,
    )


def _build_classifier_fn(
    model: Any,
    volume: Any,
    spatial_dims: int,
    axis: int,
    resolved_index: int,
    slice_lo: float,
    slice_range: float,
    device: Any,
    batch_size: int,
) -> Any:
    """Returns a `(N, H, W, 3) -> (N, num_classes)` function: for each
    perturbed slice LIME hands us, reinsert it into a *copy* of the real
    volume at the representative slice and run one real forward pass."""
    import numpy as np
    import torch

    def classifier_fn(images: Any) -> Any:
        # LIME hands us float RGB in the *display* range [0, 1] (see
        # `explain_with_lime`) — collapse the 3 duplicated channels back to
        # one, then de-normalize into the model's actual input value range.
        gray = images.mean(axis=-1)  # (N, H, W), in [0, 1]
        denormalized = gray.astype(np.float32) * slice_range + slice_lo

        all_probs = []
        for start in range(0, len(denormalized), batch_size):
            chunk = denormalized[start : start + batch_size]
            batch_slices = torch.from_numpy(chunk)
            volumes = torch.stack(
                [insert_slice(volume, spatial_dims, axis, resolved_index, s) for s in batch_slices]
            ).to(device)
            with torch.no_grad():
                logits = model(volumes)
                probs = torch.softmax(logits.float(), dim=1)
            all_probs.append(probs.cpu().numpy())
        return np.concatenate(all_probs, axis=0)

    return classifier_fn


def explain_with_lime(
    model: Any,
    volume: Any,
    class_names: list[str],
    target_class_index: int,
    config: LimeConfig,
    output_dir: Path,
    device: Any,
) -> LimeExplanationResult:
    """Generate one LIME explanation for `volume` (a single preprocessed,
    channel-first sample — see `slicing.extract_slice`) for `target_class_index`.

    Reproducible: identical `model`, `volume`, `target_class_index`, and
    `config` (same `config.seed`) always produce identical superpixel weights.

    Raises:
        ConfigurationError: `volume` is multi-channel (unsupported — see
            `slicing.extract_slice`), or `config.slice_index`/`slice_axis`
            resolves out of range.
    """
    import numpy as np
    from lime.lime_image import LimeImageExplainer

    spatial_dims = volume.ndim - 1  # volume is channel-first: (1, *spatial)
    slice_2d, resolved_index = extract_slice(
        volume, spatial_dims, config.slice_axis, config.slice_index
    )

    slice_np = slice_2d.detach().cpu().numpy()
    slice_lo = float(slice_np.min())
    slice_hi = float(slice_np.max())
    slice_range = slice_hi - slice_lo if slice_hi > slice_lo else 1.0
    display_slice = (slice_np - slice_lo) / slice_range  # [0, 1], for segmentation + display
    rgb_image = np.stack([display_slice] * 3, axis=-1)

    classifier_fn = _build_classifier_fn(
        model,
        volume,
        spatial_dims,
        config.slice_axis,
        resolved_index,
        slice_lo,
        slice_range,
        device,
        config.batch_size,
    )
    segmentation_fn = _build_segmentation_fn(config)

    explainer = LimeImageExplainer(random_state=config.seed)
    explanation = explainer.explain_instance(
        rgb_image,
        classifier_fn,
        labels=(target_class_index,),
        top_labels=None,
        num_features=config.num_features,
        num_samples=config.num_samples,
        batch_size=config.batch_size,
        segmentation_fn=segmentation_fn,
        random_seed=config.seed,
    )

    weights = dict(explanation.local_exp[target_class_index])
    ranked = sorted(weights.items(), key=lambda item: abs(item[1]), reverse=True)
    top_superpixels = [
        {"segment_id": float(segment_id), "weight": float(weight)}
        for segment_id, weight in ranked[: config.num_features]
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    temp, mask = explanation.get_image_and_mask(
        target_class_index, positive_only=False, hide_rest=False, num_features=config.num_features
    )
    overlay_path = render_lime_overlay(temp, mask, output_dir / "lime_overlay.png")

    logger.info(
        "lime_explanation_generated",
        target_class=class_names[target_class_index],
        slice_axis=config.slice_axis,
        slice_index=resolved_index,
        num_samples=config.num_samples,
        num_segments=int(explanation.segments.max()) + 1,
    )

    return LimeExplanationResult(
        target_class_index=target_class_index,
        target_class_name=class_names[target_class_index],
        slice_axis=config.slice_axis,
        slice_index=resolved_index,
        num_samples=config.num_samples,
        segmentation_algorithm=config.segmentation_algorithm,
        num_segments=int(explanation.segments.max()) + 1,
        top_superpixels=top_superpixels,
        local_prediction=float(explanation.local_pred[0]),
        surrogate_r2=float(explanation.score),
        config_hash=config.compute_hash(),
        artifacts={"lime_overlay": overlay_path},
    )
