import re
import uuid
from pathlib import Path

_VALID_ID = re.compile(r"^[0-9a-f]{32}$")

MAX_UPLOAD_BYTES = 1024 * 1024 * 1024  # 1 GB


class UploadStore:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_id(upload_id: str) -> None:
        if not _VALID_ID.match(upload_id):
            raise ValueError(f"Invalid upload ID: {upload_id!r}")

    def create(self) -> tuple[str, str]:
        upload_id = uuid.uuid4().hex
        return upload_id, f"storage://{upload_id}"

    def save(self, upload_id: str, data: bytes) -> Path:
        self._validate_id(upload_id)
        path = self.base_dir / upload_id
        path.write_bytes(data)
        return path

    def resolve(self, storage_uri: str) -> Path:
        if not storage_uri.startswith("storage://"):
            raise ValueError(f"Invalid storage URI: {storage_uri}")
        upload_id = storage_uri[len("storage://"):]
        self._validate_id(upload_id)
        path = self.base_dir / upload_id
        if not path.exists():
            raise FileNotFoundError(f"Upload not found: {storage_uri}")
        return path
