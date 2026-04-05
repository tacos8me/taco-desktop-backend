"""LoRA registry: flat-directory storage with JSON index."""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class LoRAInfo:
    id: str
    name: str
    filename: str
    base_model: str
    size_bytes: int
    uploaded_at: str
    description: str = ""
    trigger_word: str | None = None
    strategy: str | None = None


class LoRARegistry:
    """Manages LoRA files in a flat directory with a registry.json index."""

    def __init__(self, loras_dir: Path) -> None:
        self._dir = loras_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._registry_path = self._dir / "registry.json"
        self._loras: dict[str, LoRAInfo] = {}
        self._load()

    def _load(self) -> None:
        if not self._registry_path.exists():
            self._save()
            return
        data = json.loads(self._registry_path.read_text())
        self._loras = {
            entry["id"]: LoRAInfo(**entry)
            for entry in data.get("loras", [])
        }
        logger.info("LoRA registry loaded: %d loras", len(self._loras))

    def _save(self) -> None:
        data = {
            "loras": [
                {
                    "id": l.id, "name": l.name, "filename": l.filename,
                    "base_model": l.base_model, "size_bytes": l.size_bytes,
                    "uploaded_at": l.uploaded_at, "description": l.description,
                    "trigger_word": l.trigger_word, "strategy": l.strategy,
                }
                for l in self._loras.values()
            ]
        }
        tmp = self._registry_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, self._registry_path)

    def list_all(self) -> list[LoRAInfo]:
        return list(self._loras.values())

    def get(self, lora_id: str) -> LoRAInfo | None:
        return self._loras.get(lora_id)

    def resolve_path(self, lora_id: str) -> Path:
        info = self._loras.get(lora_id)
        if info is None:
            raise FileNotFoundError(f"LoRA not found: {lora_id}")
        return self._dir / f"{lora_id}.safetensors"

    def add(self, name: str, filename: str, data: bytes, description: str = "", base_model: str = "ltx-2.3",
            trigger_word: str | None = None, strategy: str | None = None) -> LoRAInfo:
        """Validate and store a LoRA file. Returns the new LoRAInfo."""
        _validate_safetensors(data)

        lora_id = uuid.uuid4().hex
        dest = self._dir / f"{lora_id}.safetensors"
        dest.write_bytes(data)

        info = LoRAInfo(
            id=lora_id,
            name=name,
            filename=filename,
            base_model=base_model,
            size_bytes=len(data),
            uploaded_at=datetime.now(timezone.utc).isoformat(),
            description=description,
            trigger_word=trigger_word,
            strategy=strategy,
        )
        self._loras[lora_id] = info
        self._save()
        logger.info("LoRA added: %s (%s, %d bytes)", name, lora_id, len(data))
        return info

    def delete(self, lora_id: str) -> bool:
        info = self._loras.pop(lora_id, None)
        if info is None:
            return False
        path = self._dir / f"{lora_id}.safetensors"
        path.unlink(missing_ok=True)
        self._save()
        logger.info("LoRA deleted: %s (%s)", info.name, lora_id)
        return True

    def count(self) -> int:
        return len(self._loras)


def _validate_safetensors(data: bytes) -> None:
    """Validate that data is a safetensors file with LoRA keys."""
    if len(data) < 8:
        raise ValueError("File too small to be a valid safetensors file")

    # SafeTensors format: first 8 bytes = little-endian u64 header size
    header_size = int.from_bytes(data[:8], "little")
    if header_size > len(data) - 8 or header_size > 100_000_000:
        raise ValueError("Invalid safetensors header")

    try:
        header = json.loads(data[8:8 + header_size])
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise ValueError("Invalid safetensors header: not valid JSON")

    # Check for LoRA keys (lora_A / lora_B pattern)
    keys = [k for k in header if k != "__metadata__"]
    has_lora_a = any("lora_A" in k or "lora_a" in k for k in keys)
    has_lora_b = any("lora_B" in k or "lora_b" in k for k in keys)
    if not (has_lora_a and has_lora_b):
        raise ValueError("File does not contain LoRA weights (missing lora_A/lora_B keys)")
