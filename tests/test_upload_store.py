import tempfile
from pathlib import Path
from upload_store import UploadStore


def test_create_returns_uuid_and_storage_uri():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = UploadStore(Path(tmpdir))
        upload_id, storage_uri = store.create()
        assert storage_uri.startswith("storage://")
        assert upload_id in storage_uri


def test_save_and_resolve():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = UploadStore(Path(tmpdir))
        upload_id, storage_uri = store.create()
        store.save(upload_id, b"fake image data")
        resolved = store.resolve(storage_uri)
        assert resolved.exists()
        assert resolved.read_bytes() == b"fake image data"


def test_resolve_unknown_uri_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = UploadStore(Path(tmpdir))
        try:
            store.resolve("storage://nonexistent")
            assert False, "Should have raised"
        except FileNotFoundError:
            pass
