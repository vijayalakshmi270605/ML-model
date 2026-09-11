"""Secure file-upload handling shared by every endpoint that accepts an MRI
file (`predictions`, `explanations`) — one implementation, not copied per
router.

Three concerns, each addressed once here:

- **Maximum upload size** (`Settings.max_upload_size_mb`): read in bounded
  chunks and abort as soon as the running total exceeds the limit, so an
  oversized upload is rejected without ever buffering the whole thing in
  memory.
- **Secure file handling**: the file is written to a fresh, per-request
  temporary directory (`tempfile.mkdtemp`) under a **generic** name
  (`upload<ext>`) — the client's original filename is never written to disk,
  logged, or reflected in an error message; see `inference.predictor`'s
  module docstring for why. Callers are responsible for deleting the temp
  directory (`shutil.rmtree`) once done — see `api.routers.predictions`/
  `api.routers.explanations`'s `finally` blocks.
- Only the file **extension** (needed for NIfTI/DICOM format detection) is
  taken from the client-supplied filename.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import HTTPException, UploadFile, status

_CHUNK_SIZE = 1024 * 1024  # 1 MiB


def upload_suffix(filename: str | None) -> str:
    """The file extension nibabel/pydicom need to detect format — the only
    part of an uploaded filename this project ever looks at or keeps."""
    if not filename:
        return ".nii.gz"
    lowered = filename.lower()
    if lowered.endswith(".nii.gz"):
        return ".nii.gz"
    return Path(filename).suffix or ".nii.gz"


async def save_upload_within_limit(file: UploadFile, max_bytes: int) -> tuple[Path, Path]:
    """Stream `file` to a fresh temp directory, generically named, aborting
    with `HTTPException(413)` if it exceeds `max_bytes` before ever holding
    more than one chunk of it in memory.

    Returns `(temp_dir, temp_path)` — the caller owns cleanup
    (`shutil.rmtree(temp_dir, ignore_errors=True)`).

    Raises:
        HTTPException: 413 if the upload exceeds `max_bytes`.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="ad-mri-upload-"))
    temp_path = temp_dir / f"upload{upload_suffix(file.filename)}"

    total = 0
    try:
        with temp_path.open("wb") as fh:
            while True:
                chunk = await file.read(_CHUNK_SIZE)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail=f"Upload exceeds the maximum allowed size of {max_bytes} bytes.",
                    )
                fh.write(chunk)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    if total == 0:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="The uploaded file is empty."
        )

    return temp_dir, temp_path
