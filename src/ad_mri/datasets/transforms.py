"""MONAI transform pipeline: the single preprocessing implementation shared by
training and inference.

`build_transforms(config, train=...)` is the one function both call — training
passes `train=True` (deterministic preprocessing + augmentation), inference
passes `train=False` (deterministic preprocessing only). There is no second
copy of the preprocessing steps anywhere; augmentation is strictly additive on
top of the same deterministic sequence, never a fork of it. See
docs/TECHNICAL_SPECIFICATION.md §4, §5, §6.

Design notes:

- **No cross-image statistics.** Every normalization mode operates on a single
  volume's own values (per-volume z-score, or percentile/min-max rescaling
  read off that volume). Nothing here computes or depends on a dataset-wide
  mean/std, so there is no channel for validation/test statistics to leak into
  training (requirement: "avoid data leakage").
- **Deterministic vs. augmentation are separate, composable step lists**
  (`build_preprocessing_transforms` / `build_augmentation_transforms`) so the
  same deterministic steps are provably identical whether or not augmentation
  is appended.
- **Augmentation ranges are deliberately conservative** — see
  `PreprocessingConfig`/`AugmentationConfig` docstrings for the rationale
  behind each default (small-angle rotation instead of 90-degree rotation,
  left-right flip off by default, intensity jitter bounded to a fraction of
  the normalized scale).

`monai`/`torch` are training-phase dependencies (requirements/train.txt),
imported lazily so this module stays importable without them.
"""

from __future__ import annotations

from typing import Any

from ad_mri.core.exceptions import ConfigurationError
from ad_mri.core.logging import get_logger
from ad_mri.datasets.preprocessing_config import PreprocessingConfig

logger = get_logger(__name__)


def _resolve_config(config: PreprocessingConfig | dict[str, Any] | None) -> PreprocessingConfig:
    if config is None:
        return PreprocessingConfig()
    if isinstance(config, PreprocessingConfig):
        return config
    return PreprocessingConfig(**config)


def _build_normalization_transform(config: PreprocessingConfig) -> Any:
    from monai.transforms import (
        NormalizeIntensityd,
        ScaleIntensityd,
        ScaleIntensityRangePercentilesd,
    )

    if config.normalization == "zscore_nonzero":
        # Per-volume z-score over nonzero (foreground) voxels only — the
        # standard choice for MRI, since background is a large, uninformative
        # zero-intensity region that would otherwise dominate the statistics.
        return NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True)
    if config.normalization == "zscore":
        return NormalizeIntensityd(keys="image", nonzero=False, channel_wise=True)
    if config.normalization == "percentile_clip":
        # Robust to outlier-bright voxels and to the arbitrary-unit intensity
        # scale of MRI (unlike CT, there's no fixed Hounsfield reference) —
        # useful when different source scanners/protocols produce different
        # intensity distributions (requirement: "handle different MRI
        # intensity distributions").
        lower, upper = config.percentile_clip
        return ScaleIntensityRangePercentilesd(
            keys="image", lower=lower, upper=upper, b_min=0.0, b_max=1.0, clip=True, relative=False
        )
    if config.normalization == "min_max":
        return ScaleIntensityd(keys="image", minv=0.0, maxv=1.0)
    raise ConfigurationError(f"Unknown normalization mode: {config.normalization!r}")


def build_preprocessing_transforms(
    config: PreprocessingConfig | dict[str, Any] | None = None,
) -> list[Any]:
    """The deterministic pipeline: load -> orient -> resample -> normalize ->
    resize -> (optional QC) -> cast to tensor.

    Used, unmodified, by both training (wrapped by `build_transforms(train=True)`,
    which appends augmentation) and inference (`build_transforms(train=False)`).
    """
    import torch
    from monai.transforms import (
        EnsureChannelFirstd,
        EnsureTyped,
        Orientationd,
        ResizeWithPadOrCropd,
        Spacingd,
    )

    from ad_mri.datasets.custom_transforms import QualityControld, SafeLoadImaged

    config = _resolve_config(config)

    steps: list[Any] = [
        SafeLoadImaged(keys="image", image_only=True, ensure_channel_first=False),
        EnsureChannelFirstd(keys="image"),
        # `labels` pinned explicitly (not left to MONAI's default) to avoid a
        # pending default-value change in a future MONAI release silently
        # altering axis-label interpretation.
        Orientationd(
            keys="image",
            axcodes=config.target_orientation,
            labels=(("L", "R"), ("P", "A"), ("I", "S")),
        ),
        Spacingd(keys="image", pixdim=config.target_spacing, mode=config.interpolation_mode),
        _build_normalization_transform(config),
        ResizeWithPadOrCropd(keys="image", spatial_size=config.target_shape),
    ]

    if config.enable_quality_control:
        steps.append(
            QualityControld(
                keys="image", min_nonzero_ratio=config.qc_min_nonzero_ratio, mode=config.qc_mode
            )
        )

    steps.append(EnsureTyped(keys="image", dtype=torch.float32))
    return steps


def build_augmentation_transforms(
    config: PreprocessingConfig | dict[str, Any] | None = None,
) -> list[Any]:
    """Training-only randomized augmentation. Never applied at val/test/inference
    — callers only ever add these on top of `build_preprocessing_transforms`
    when `train=True` (see `build_transforms`).

    Deliberately excludes anything that could produce a medically unrealistic
    sample for this task: no 90-degree/arbitrary rotation (only a small tilt),
    no left-right flip by default (see `AugmentationConfig.allow_flip`), no
    aggressive intensity remapping.
    """
    import math

    from monai.transforms import (
        RandAffined,
        RandFlipd,
        RandGaussianNoised,
        RandScaleIntensityd,
        RandShiftIntensityd,
    )

    config = _resolve_config(config)
    aug = config.augmentation
    if not aug.enabled:
        return []

    rotation_rad = math.radians(aug.rotation_range_deg)
    steps: list[Any] = [
        RandAffined(
            keys="image",
            prob=aug.probability,
            rotate_range=(rotation_rad, rotation_rad, rotation_rad),
            translate_range=(aug.translation_range_voxels,) * 3,
            scale_range=(aug.scale_range,) * 3,
            mode="bilinear",
            padding_mode="border",
        ),
        RandGaussianNoised(
            keys="image", prob=aug.probability, mean=0.0, std=aug.gaussian_noise_std
        ),
        RandScaleIntensityd(keys="image", factors=aug.intensity_scale_range, prob=aug.probability),
        RandShiftIntensityd(keys="image", offsets=aug.intensity_shift_range, prob=aug.probability),
    ]
    if aug.allow_flip:
        steps.append(RandFlipd(keys="image", prob=aug.flip_probability, spatial_axis=0))
    return steps


def build_transforms(
    config: PreprocessingConfig | dict[str, Any] | None = None,
    *,
    train: bool,
    seed: int | None = None,
) -> Any:
    """Build the full MONAI `Compose` pipeline operating on `{"image", "label"}` dicts.

    `train=True`: deterministic preprocessing + augmentation (training only).
    `train=False`: deterministic preprocessing only — the exact same steps,
    same parameters, used for validation, test, and production inference.

    `seed` (or `config.seed`) pins the augmentation RNG for reproducibility
    (e.g. in tests/demos); training normally relies on the global seed
    (`core.seeding.set_global_seed`) instead.

    Raises:
        ImportError: monai/torch are not installed (install `requirements/train.txt`).
    """
    from monai.transforms import Compose

    config = _resolve_config(config)
    steps = build_preprocessing_transforms(config)
    if train:
        steps = steps + build_augmentation_transforms(config)

    compose = Compose(steps)
    resolved_seed = seed if seed is not None else config.seed
    if resolved_seed is not None:
        compose.set_random_state(seed=resolved_seed)
    return compose


def apply_transforms(transform: Any, data: dict[str, Any]) -> dict[str, Any]:
    """Run a `Compose` pipeline built by `build_transforms`, applied to one sample.

    MONAI's `Compose` wraps *any* exception raised inside a transform step in a
    generic `RuntimeError("applying transform ...")`, which would otherwise
    force every caller to inspect `__cause__` to tell a real failure
    (`PreprocessingError`/`DataValidationError`) apart from a MONAI internal
    error. This unwraps that so callers can catch our specific exception types
    directly — used by every call site (`MRIDataset`, `Predictor`,
    the visualization utility) so the exception surface stays consistent.
    """
    from ad_mri.core.exceptions import DataValidationError, PreprocessingError

    try:
        return dict(transform(data))
    except RuntimeError as exc:
        cause = exc.__cause__
        if isinstance(cause, (PreprocessingError, DataValidationError)):
            raise cause from exc
        raise
