"""Object storage for large medical-imaging artifacts.

Requirement: "Do NOT store raw MRI images in PostgreSQL. Use object storage
architecture for large medical imaging files if persistence is required."
Neither requirement is theoretical hand-waving here: `db.models.ModelVersionRecord.checkpoint_uri`
and `db.models.ExplanationRecord.artifact_uri` are exactly the columns that
would otherwise tempt someone to inline bytes into PostgreSQL - they store a
URI produced by this module instead, and the Postgres column widths
(`String(512)`) are deliberately too narrow to hold anything else.

`LocalFilesystemObjectStorage` is real and is what a single-node/dev
deployment uses today - it is exactly the filesystem
`inference_explanation_output_dir` (see `core.config.Settings`) already
writes explanation manifests to; this class only adds URI framing around
that existing behavior, it does not relocate it. `S3ObjectStorageClient` is
the same kind of documented-but-unimplemented seam as
`model_registry.client.MLflowModelRegistryClient` - swapping the backend a
production deployment uses is a config change (`OBJECT_STORAGE_BACKEND`),
not a rewrite of any caller, because every caller only ever depends on the
`ObjectStorageClient` protocol below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


def local_file_uri(path: Path) -> str:
    """A `file://` URI for a path that already exists on disk - used to
    reference an artifact (a checkpoint, an explanation manifest) without
    instantiating a full `LocalFilesystemObjectStorage` (which creates its
    root directory as a side effect) just to describe where something
    already is."""
    return f"file://{Path(path).resolve().as_posix()}"


@runtime_checkable
class ObjectStorageClient(Protocol):
    """What every caller (`db.audit_service`, `inference.predictor`) depends
    on - never a concrete backend class directly."""

    def put_bytes(self, key: str, data: bytes) -> str:
        """Write `data` under `key` and return its URI."""
        ...

    def get_bytes(self, key: str) -> bytes:
        """Read back the bytes written under `key`."""
        ...

    def uri_for(self, path: Path) -> str:
        """Return the URI for an artifact already on disk at `path` (e.g. an
        explanation manifest `ExplainabilityService` already wrote), without
        re-uploading it."""
        ...


class LocalFilesystemObjectStorage:
    """The real, default backend: a directory on local (or a mounted
    network/NFS) disk, addressed by `file://` URIs. Suitable for a
    single-node deployment; `S3ObjectStorageClient` below is the seam for a
    multi-node one."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def put_bytes(self, key: str, data: bytes) -> str:
        destination = self._root / key
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        return self.uri_for(destination)

    def get_bytes(self, key: str) -> bytes:
        return (self._root / key).read_bytes()

    def uri_for(self, path: Path) -> str:
        return local_file_uri(path)


class S3ObjectStorageClient:
    """Multi-node/production seam - not implemented in this phase (no `boto3`
    dependency has been added). Mirrors
    `model_registry.client.MLflowModelRegistryClient`'s pattern: the
    interface a real S3/MinIO-backed client will satisfy is fixed now
    (`ObjectStorageClient`), so adopting one later is additive."""

    def __init__(self, bucket: str, prefix: str = "") -> None:
        self._bucket = bucket
        self._prefix = prefix

    def put_bytes(self, key: str, data: bytes) -> str:
        raise NotImplementedError("S3 object storage lands with the multi-node deployment phase.")

    def get_bytes(self, key: str) -> bytes:
        raise NotImplementedError("S3 object storage lands with the multi-node deployment phase.")

    def uri_for(self, path: Path) -> str:
        raise NotImplementedError("S3 object storage lands with the multi-node deployment phase.")
