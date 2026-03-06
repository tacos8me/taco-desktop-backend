import uuid
from pathlib import Path


class UploadStore:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create(self) -> tuple[str, str]:
        upload_id = uuid.uuid4().hex
        return upload_id, f"storage://{upload_id}"

    def save(self, upload_id: str, data: bytes) -> Path:
        path = self.base_dir / upload_id
        path.write_bytes(data)
        return path

    def resolve(self, storage_uri: str) -> Path:
        if not storage_uri.startswith("storage://"):
            raise ValueError(f"Invalid storage URI: {storage_uri}")
        upload_id = storage_uri[len("storage://"):]
        path = self.base_dir / upload_id
        if not path.exists():
            raise FileNotFoundError(f"Upload not found: {storage_uri}")
        return path
