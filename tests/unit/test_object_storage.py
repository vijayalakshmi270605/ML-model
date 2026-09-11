"""Tests for `storage.object_storage` - real file I/O against a real
temp directory, no mocks. `S3ObjectStorageClient` is checked only for the
documented-seam contract (raises `NotImplementedError`), matching
`model_registry.client.MLflowModelRegistryClient`'s test coverage."""

from __future__ import annotations

from pathlib import Path

import pytest

from ad_mri.storage.object_storage import (
    LocalFilesystemObjectStorage,
    S3ObjectStorageClient,
    local_file_uri,
)


def test_local_file_uri_is_an_absolute_file_scheme_uri(tmp_path: Path) -> None:
    target = tmp_path / "checkpoint.pt"
    target.write_bytes(b"fake-weights")
    uri = local_file_uri(target)
    assert uri.startswith("file://")
    assert uri.endswith("checkpoint.pt")


def test_put_bytes_writes_a_real_file_and_returns_its_uri(tmp_path: Path) -> None:
    storage = LocalFilesystemObjectStorage(tmp_path / "store")
    uri = storage.put_bytes("explanations/expl-1/manifest.json", b'{"ok": true}')

    assert uri.startswith("file://")
    written_path = Path(uri.removeprefix("file://"))
    assert written_path.read_bytes() == b'{"ok": true}'


def test_get_bytes_reads_back_exactly_what_was_written(tmp_path: Path) -> None:
    storage = LocalFilesystemObjectStorage(tmp_path / "store")
    storage.put_bytes("a/b.bin", b"\x00\x01\x02binary-data")
    assert storage.get_bytes("a/b.bin") == b"\x00\x01\x02binary-data"


def test_uri_for_an_existing_path_does_not_copy_or_modify_it(tmp_path: Path) -> None:
    existing = tmp_path / "already-here.json"
    existing.write_text("{}", encoding="utf-8")
    storage = LocalFilesystemObjectStorage(tmp_path / "store")

    uri = storage.uri_for(existing)
    assert uri == local_file_uri(existing)
    # The store's own root was created, but nothing was copied into it.
    assert list((tmp_path / "store").iterdir()) == []


def test_constructor_creates_the_root_directory(tmp_path: Path) -> None:
    root = tmp_path / "does" / "not" / "exist" / "yet"
    assert not root.exists()
    LocalFilesystemObjectStorage(root)
    assert root.is_dir()


class TestS3ObjectStorageClientSeam:
    def test_put_bytes_is_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            S3ObjectStorageClient(bucket="ad-mri-artifacts").put_bytes("key", b"data")

    def test_get_bytes_is_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            S3ObjectStorageClient(bucket="ad-mri-artifacts").get_bytes("key")

    def test_uri_for_is_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            S3ObjectStorageClient(bucket="ad-mri-artifacts").uri_for(Path("x"))
