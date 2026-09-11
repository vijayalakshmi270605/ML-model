"""Shared 2D-slice extraction/insertion for applying 2D-image explanation
tooling (LIME's image explainer, slice-level SHAP overlays) to a 3D MRI
volume **without ever flattening the whole volume**.

Both `lime_explainer` and `shap_explainer`/`overlays` import from here so the
"which slice is 'representative'" decision is made exactly once and
consistently, rather than each explainer picking its own convention.

Design note (see docs/explainability.md "Limitations" for the full
rationale): a full 3D volume forward pass is cheap (it's what training/
inference already does), but LIME's algorithm needs *hundreds* of forward
passes over *perturbed* input to fit a local surrogate model, and each
perturbation only makes sense as a 2D superpixel mask. Perturbing a single
representative 2D slice — while leaving the rest of the real, preprocessed
3D volume untouched — keeps every forward pass a genuine call to the real 3D
model (not a separate, unfaithful 2D proxy model) while keeping the sample
budget tractable. This is a deliberate, documented representation choice, not
an accidental dimensionality reduction.

`torch`/`numpy` are training-phase dependencies (requirements/train.txt),
imported lazily so this module stays importable without them.
"""

from __future__ import annotations

from typing import Any

from ad_mri.core.exceptions import ConfigurationError


def resolve_slice_index(axis_size: int, slice_index: str) -> int:
    """`slice_index` is `"middle"` or a base-10 integer string (so it round-trips
    cleanly through YAML/JSON config as a single `str` field). Negative indices
    (Python-style, from the end) are accepted."""
    if slice_index == "middle":
        return axis_size // 2
    try:
        resolved = int(slice_index)
    except ValueError as exc:
        raise ConfigurationError(
            f"slice_index must be 'middle' or an integer string, got {slice_index!r}."
        ) from exc
    if resolved < 0:
        resolved += axis_size
    if not (0 <= resolved < axis_size):
        raise ConfigurationError(
            f"slice_index {slice_index!r} resolves to {resolved}, out of range "
            f"for an axis of size {axis_size}."
        )
    return resolved


def extract_slice(volume: Any, spatial_dims: int, axis: int, slice_index: str) -> tuple[Any, int]:
    """Extract one 2D slice from a preprocessed volume tensor.

    `volume`: shape `(1, H, W)` (2D model) or `(1, D, H, W)` (3D model) — a
    single-channel MRI volume, channel-first, no batch dimension (the exact
    shape `datasets.transforms.build_transforms(..., train=False)` produces).
    Multi-channel input is out of scope here (raises `ConfigurationError`) —
    documented as a limitation, not silently mishandled.

    Returns `(slice_2d, resolved_index)`: `slice_2d` has shape `(H, W)`.
    """
    if volume.shape[0] != 1:
        raise ConfigurationError(
            f"slicing.extract_slice only supports single-channel volumes, got "
            f"{volume.shape[0]} channels — multi-channel MRI explanation is a "
            "documented limitation (see docs/explainability.md)."
        )
    core = volume[0]  # drop the channel dim: (H, W) or (D, H, W)

    if spatial_dims == 2:
        return core, 0
    if spatial_dims != 3:
        raise ConfigurationError(f"Unsupported spatial_dims for slicing: {spatial_dims}")

    if not (0 <= axis <= 2):
        raise ConfigurationError(f"slice_axis must be 0, 1, or 2 for a 3D volume, got {axis}.")
    axis_size = core.shape[axis]
    resolved_index = resolve_slice_index(axis_size, slice_index)
    slice_2d = core.select(dim=axis, index=resolved_index)
    return slice_2d, resolved_index


def insert_slice(
    volume: Any, spatial_dims: int, axis: int, resolved_index: int, new_slice: Any
) -> Any:
    """Return a **copy** of `volume` with the 2D slice at `(axis, resolved_index)`
    replaced by `new_slice` — every other voxel is untouched, real,
    preprocessed data. Used by LIME to perturb only the representative slice
    while keeping the rest of the volume faithful to the actual input."""
    updated = volume.clone()
    if spatial_dims == 2:
        updated[0] = new_slice
        return updated
    updated[0].select(dim=axis, index=resolved_index).copy_(new_slice)
    return updated
