"""Tests for `db.privacy` - the enforcement layer behind "do not store
patient names / unnecessary identifiers / raw MRI data / secrets"."""

from __future__ import annotations

import hashlib

import pytest

from ad_mri.db.privacy import ensure_no_raw_path, ensure_redacted_image_reference, hash_value
from ad_mri.inference.predictor import redact_image_reference


def test_hash_value_is_a_stable_sha256_hex_digest() -> None:
    assert hash_value("super-secret-key") == hashlib.sha256(b"super-secret-key").hexdigest()


def test_hash_value_never_returns_the_input() -> None:
    assert hash_value("my-api-key") != "my-api-key"


def test_ensure_redacted_image_reference_accepts_a_real_redacted_hash(tmp_path) -> None:
    real_hash = redact_image_reference(tmp_path / "john_doe_1957.nii.gz")
    assert ensure_redacted_image_reference(real_hash) == real_hash


@pytest.mark.parametrize(
    "bad_value",
    [
        "john_doe_1957.nii.gz",
        "/home/user/scans/jane_smith.nii.gz",
        r"C:\Users\clinician\jane_smith.nii.gz",
        "",
        "not-16-hex-chars",
        "UPPERCASE1234567",  # 16 chars but not lowercase hex
    ],
)
def test_ensure_redacted_image_reference_rejects_anything_not_shaped_like_a_hash(
    bad_value: str,
) -> None:
    with pytest.raises(ValueError, match="image_ref"):
        ensure_redacted_image_reference(bad_value)


def test_ensure_no_raw_path_accepts_a_plain_identifier() -> None:
    assert ensure_no_raw_path("inference_failed", field_name="event_type") == "inference_failed"


@pytest.mark.parametrize(
    "path_like",
    ["/tmp/upload.nii.gz", r"C:\temp\upload.nii.gz", "subdir/file.txt"],
)
def test_ensure_no_raw_path_rejects_anything_with_a_separator(path_like: str) -> None:
    with pytest.raises(ValueError, match="field_x"):
        ensure_no_raw_path(path_like, field_name="field_x")
