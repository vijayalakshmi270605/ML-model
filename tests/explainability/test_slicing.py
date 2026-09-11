"""Tests for explainability.slicing — hand-verifiable extract/insert round-trips."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ad_mri.core.exceptions import ConfigurationError  # noqa: E402
from ad_mri.explainability.slicing import (  # noqa: E402
    extract_slice,
    insert_slice,
    resolve_slice_index,
)


def test_resolve_slice_index_middle() -> None:
    assert resolve_slice_index(16, "middle") == 8
    assert resolve_slice_index(15, "middle") == 7


def test_resolve_slice_index_explicit_integer() -> None:
    assert resolve_slice_index(16, "3") == 3


def test_resolve_slice_index_negative_wraps_from_the_end() -> None:
    assert resolve_slice_index(16, "-1") == 15
    assert resolve_slice_index(16, "-16") == 0


def test_resolve_slice_index_out_of_range_raises() -> None:
    with pytest.raises(ConfigurationError, match="out of range"):
        resolve_slice_index(16, "16")


def test_resolve_slice_index_non_integer_raises() -> None:
    with pytest.raises(ConfigurationError, match="'middle' or an integer"):
        resolve_slice_index(16, "not-an-int")


def test_extract_slice_3d_returns_expected_shape_and_index() -> None:
    volume = torch.arange(1 * 4 * 5 * 6, dtype=torch.float32).reshape(1, 4, 5, 6)
    slice_2d, resolved_index = extract_slice(volume, spatial_dims=3, axis=0, slice_index="middle")
    assert resolved_index == 2
    assert slice_2d.shape == (5, 6)
    assert torch.equal(slice_2d, volume[0, 2])


def test_extract_slice_2d_returns_the_whole_image() -> None:
    volume = torch.rand(1, 8, 8)
    slice_2d, resolved_index = extract_slice(volume, spatial_dims=2, axis=0, slice_index="middle")
    assert resolved_index == 0
    assert torch.equal(slice_2d, volume[0])


def test_extract_slice_rejects_multi_channel_volume() -> None:
    volume = torch.rand(2, 4, 5, 6)
    with pytest.raises(ConfigurationError, match="single-channel"):
        extract_slice(volume, spatial_dims=3, axis=0, slice_index="middle")


def test_insert_slice_only_changes_the_targeted_slice() -> None:
    volume = torch.zeros(1, 4, 5, 6)
    new_slice = torch.ones(5, 6)
    updated = insert_slice(volume, spatial_dims=3, axis=0, resolved_index=2, new_slice=new_slice)

    assert torch.equal(updated[0, 2], new_slice)
    for index in (0, 1, 3):
        assert torch.equal(updated[0, index], torch.zeros(5, 6))
    # the original volume is untouched (insert_slice returns a copy)
    assert torch.equal(volume, torch.zeros(1, 4, 5, 6))


def test_extract_then_insert_the_same_slice_is_a_no_op() -> None:
    volume = torch.rand(1, 4, 5, 6)
    original = volume.clone()
    slice_2d, resolved_index = extract_slice(volume, spatial_dims=3, axis=1, slice_index="middle")
    updated = insert_slice(
        volume, spatial_dims=3, axis=1, resolved_index=resolved_index, new_slice=slice_2d
    )
    assert torch.equal(updated, original)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_insert_slice_works_on_every_axis(axis: int) -> None:
    volume = torch.zeros(1, 4, 5, 6)
    axis_size = volume.shape[1 + axis]
    other_dims = [volume.shape[1 + d] for d in range(3) if d != axis]
    new_slice = torch.ones(*other_dims)
    updated = insert_slice(
        volume, spatial_dims=3, axis=axis, resolved_index=axis_size // 2, new_slice=new_slice
    )
    assert torch.equal(updated[0].select(dim=axis, index=axis_size // 2), new_slice)
