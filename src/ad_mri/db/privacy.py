"""Privacy/security guards enforced at the database boundary itself.

The rest of the codebase already avoids logging patient-sensitive data by
convention (`inference.predictor.redact_image_reference`,
`api.exception_handlers` never logging `str(exc)`, ...). This module makes
one of those conventions a *hard, testable invariant of the audit trail*
rather than something reviewers must remember: `ad_mri.db.repositories`
calls `ensure_redacted_image_reference` on every value it is about to
persist as `image_ref`, so a caller that accidentally passed a raw upload
path or filename fails loudly (`ValueError`) instead of writing patient-
identifying text into PostgreSQL.
"""

from __future__ import annotations

import hashlib
import re

#: `redact_image_reference()` always returns exactly this shape - the first
#: 16 hex characters of a sha256 digest. Anything else (a path, a filename,
#: an empty string) is rejected before it can reach the database.
_REDACTED_REFERENCE_RE = re.compile(r"^[0-9a-f]{16}$")


def hash_value(value: str) -> str:
    """One-way sha256 hex digest - used to store a stable reference to a
    secret or identifier (e.g. a presented API key, a client IP) without
    ever persisting the raw value itself."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def ensure_redacted_image_reference(value: str) -> str:
    """Raise `ValueError` unless `value` is already in
    `redact_image_reference()`'s output shape. Returns `value` unchanged
    otherwise (a pass-through validator, not a transform) - it is the
    caller's job to redact; this only refuses to store what was not."""
    if not _REDACTED_REFERENCE_RE.match(value):
        raise ValueError(
            "image_ref must be a redact_image_reference() hash, not a raw path or "
            f"filename; got a value of length {len(value)} that does not match the "
            "expected 16-hex-character shape."
        )
    return value


def ensure_no_raw_path(value: str, *, field_name: str) -> str:
    """Raise `ValueError` if `value` looks like a filesystem path (contains a
    path separator) - a cheap, defensive backstop for any other freeform
    string field (e.g. `system_events.detail` values) that must never carry
    an uploaded file's original path or name."""
    if "/" in value or "\\" in value:
        raise ValueError(
            f"{field_name} must not contain a filesystem path or filename; "
            "store a hash or identifier instead."
        )
    return value
