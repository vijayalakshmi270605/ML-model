"""Tests for api.upload_utils — real streaming, size-limit enforcement, and
cleanup behavior (no mocks)."""

from __future__ import annotations

import io
import shutil
import tempfile as tempfile_module
from pathlib import Path

import pytest
from fastapi import HTTPException, UploadFile

from ad_mri.api.upload_utils import save_upload_within_limit, upload_suffix


def _upload(data: bytes, filename: str | None = "scan.nii.gz") -> UploadFile:
    return UploadFile(io.BytesIO(data), filename=filename)


def test_upload_suffix_preserves_double_extension() -> None:
    assert upload_suffix("scan.nii.gz") == ".nii.gz"
    assert upload_suffix("SCAN.NII.GZ") == ".nii.gz"


def test_upload_suffix_preserves_single_extension() -> None:
    assert upload_suffix("scan.dcm") == ".dcm"
    assert upload_suffix("scan.nii") == ".nii"


def test_upload_suffix_defaults_when_no_filename() -> None:
    assert upload_suffix(None) == ".nii.gz"
    assert upload_suffix("") == ".nii.gz"


def test_upload_suffix_defaults_when_no_extension() -> None:
    assert upload_suffix("scan") == ".nii.gz"


@pytest.mark.asyncio
async def test_save_upload_writes_the_full_content() -> None:
    payload = b"a real scan's bytes" * 1000
    temp_dir, temp_path = await save_upload_within_limit(
        _upload(payload), max_bytes=len(payload) + 1
    )
    try:
        assert temp_path.read_bytes() == payload
        assert temp_dir in temp_path.parents
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_save_upload_never_writes_the_original_filename() -> None:
    temp_dir, temp_path = await save_upload_within_limit(
        _upload(b"data", filename="john_doe_1957_confidential.nii.gz"), max_bytes=1000
    )
    try:
        assert temp_path.name == "upload.nii.gz"
        assert "john_doe" not in str(temp_path)
        assert "john_doe" not in str(temp_dir)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_save_upload_rejects_oversized_upload() -> None:
    payload = b"x" * 100
    with pytest.raises(HTTPException) as excinfo:
        await save_upload_within_limit(_upload(payload), max_bytes=50)
    assert excinfo.value.status_code == 413


@pytest.mark.asyncio
async def test_save_upload_oversized_upload_leaves_no_temp_dir_behind() -> None:
    before = set(Path(tempfile_module.gettempdir()).glob("ad-mri-upload-*"))
    payload = b"x" * 100
    with pytest.raises(HTTPException):
        await save_upload_within_limit(_upload(payload), max_bytes=50)
    after = set(Path(tempfile_module.gettempdir()).glob("ad-mri-upload-*"))
    assert after - before == set()  # no new leftover directories


@pytest.mark.asyncio
async def test_save_upload_rejects_empty_file() -> None:
    with pytest.raises(HTTPException) as excinfo:
        await save_upload_within_limit(_upload(b""), max_bytes=1000)
    assert excinfo.value.status_code == 422


@pytest.mark.asyncio
async def test_save_upload_accepts_exactly_at_the_limit() -> None:
    payload = b"y" * 100
    temp_dir, temp_path = await save_upload_within_limit(_upload(payload), max_bytes=100)
    try:
        assert temp_path.read_bytes() == payload
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# --- Path traversal (Phase 14 security review) ------------------------------
#
# `upload_suffix()` only ever contributes to a *suffix* string appended to
# the fixed literal "upload" - real, adversarial filenames are exercised
# below, proving (not assuming) that a crafted name can never make the
# resulting temp file land outside its own fresh `mkdtemp()` directory.


@pytest.mark.parametrize(
    "malicious_filename",
    [
        "../../../etc/passwd",
        "..\\..\\..\\Windows\\System32\\config\\SAM",
        "../../etc/passwd.nii.gz",
        "....//....//etc/passwd",
        "/etc/passwd",
        "C:\\Windows\\System32\\drivers\\etc\\hosts",
        "a" * 5000 + ".nii.gz",  # pathologically long "filename"
    ],
)
def test_upload_suffix_never_contains_a_path_separator(malicious_filename: str) -> None:
    suffix = upload_suffix(malicious_filename)
    assert "/" not in suffix
    assert "\\" not in suffix


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "malicious_filename",
    ["../../../etc/passwd", "..\\..\\..\\Windows\\System32\\config\\SAM", "/etc/passwd"],
)
async def test_save_upload_with_a_path_traversal_filename_stays_inside_its_temp_dir(
    malicious_filename: str,
) -> None:
    payload = b"whatever bytes - the filename is the attack, not the content"
    temp_dir, temp_path = await save_upload_within_limit(
        _upload(payload, filename=malicious_filename), max_bytes=len(payload) + 1
    )
    try:
        # The resolved file is a direct, single-level child of the fresh
        # temp directory - never escaped it via `../` traversal.
        assert temp_path.parent == temp_dir
        assert temp_path.resolve().parent == temp_dir.resolve()
        assert temp_path.name.startswith("upload")
        assert temp_path.read_bytes() == payload
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
