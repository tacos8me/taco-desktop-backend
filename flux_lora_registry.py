"""Flux 2 LoRA registry: folder-drop discovery.

Source of truth is the filesystem itself. Drop `.safetensors` files into
`config.FLUX_LORAS_DIR` and they appear in `GET /v1/flux-loras`. Optional
sidecar `.json` alongside the `.safetensors` provides display metadata;
absent sidecar falls back to the filename stem.

Unlike the LTX `LoRARegistry`, there is no upload endpoint, no `registry.json`
index, and no ID-remapping: the LoRA's `id` is the slugified filename stem.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FluxLoRAInfo:
    id: str                          # slug of filename stem
    name: str                        # display name (sidecar or stem)
    filename: str                    # on-disk filename
    size_bytes: int
    model_compat: list[str] = field(default_factory=lambda: ["flux2-dev", "flux2-klein"])
    description: str = ""
    trigger_word: str | None = None


class FluxLoRARegistry:
    """Filesystem-backed Flux 2 LoRA registry.

    Scans `loras_dir` on construction and on demand. IDs are stable slugs
    derived from filenames — renaming a file changes its ID by design.
    """

    def __init__(self, loras_dir: Path) -> None:
        self._dir = loras_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._loras: dict[str, FluxLoRAInfo] = {}
        self._lock = threading.Lock()
        self.rescan()

    def rescan(self) -> int:
        """Re-scan directory for `.safetensors` files. Returns count of valid LoRAs."""
        with self._lock:
            self._loras = {}
            for path in sorted(self._dir.glob("*.safetensors")):
                try:
                    info = self._load_info(path)
                except Exception as exc:
                    logger.warning("flux_lora_registry: skipping %s: %s", path.name, exc)
                    continue
                if info.id in self._loras:
                    logger.warning(
                        "flux_lora_registry: duplicate slug %r (already from %s), skipping %s",
                        info.id, self._loras[info.id].filename, path.name,
                    )
                    continue
                self._loras[info.id] = info
            logger.info("flux_lora_registry: scanned %d LoRA(s) in %s", len(self._loras), self._dir)
            return len(self._loras)

    def list_all(self) -> list[FluxLoRAInfo]:
        with self._lock:
            return list(self._loras.values())

    def get(self, lora_id: str) -> FluxLoRAInfo | None:
        with self._lock:
            return self._loras.get(lora_id)

    def resolve_path(self, lora_id: str) -> Path:
        info = self.get(lora_id)
        if info is None:
            raise FileNotFoundError(f"Flux LoRA not found: {lora_id}")
        return self._dir / info.filename

    def count(self) -> int:
        with self._lock:
            return len(self._loras)

    # ---- internals ----

    def _load_info(self, path: Path) -> FluxLoRAInfo:
        """Build FluxLoRAInfo from a `.safetensors` file on disk. Raises on invalid."""
        _validate_safetensors(path)

        stem = path.stem
        lora_id = self._slugify(stem)

        sidecar = path.with_suffix(".json")
        meta: dict = {}
        if sidecar.exists():
            try:
                meta = json.loads(sidecar.read_text())
                if not isinstance(meta, dict):
                    raise ValueError("sidecar is not a JSON object")
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
                logger.warning("flux_lora_registry: invalid sidecar %s: %s", sidecar.name, exc)
                meta = {}

        model_compat = meta.get("model_compat")
        if not isinstance(model_compat, list) or not all(isinstance(m, str) for m in model_compat):
            model_compat = ["flux2-dev", "flux2-klein"]

        return FluxLoRAInfo(
            id=lora_id,
            name=str(meta.get("name", stem)),
            filename=path.name,
            size_bytes=path.stat().st_size,
            model_compat=model_compat,
            description=str(meta.get("description", "")),
            trigger_word=meta.get("trigger_word"),
        )

    @staticmethod
    def _slugify(stem: str) -> str:
        """Deterministic URL-safe ID from filename stem."""
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", stem).strip("-").lower()
        return slug or "unnamed"


def _validate_safetensors(path: Path) -> None:
    """Parse safetensors header and verify lora_A/lora_B keys exist.

    Mirrors lora_registry._validate_safetensors but reads from a path instead
    of in-memory bytes (since LoRA files are dropped as files, not uploaded).
    """
    size = path.stat().st_size
    if size < 8:
        raise ValueError("file too small to be a valid safetensors file")

    with path.open("rb") as f:
        header_size_bytes = f.read(8)
        header_size = int.from_bytes(header_size_bytes, "little")
        if header_size > size - 8 or header_size > 100_000_000:
            raise ValueError("invalid safetensors header size")
        header_bytes = f.read(header_size)

    try:
        header = json.loads(header_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"invalid safetensors header JSON: {exc}") from exc

    keys = [k for k in header if k != "__metadata__"]
    has_lora_a = any("lora_A" in k or "lora_a" in k for k in keys)
    has_lora_b = any("lora_B" in k or "lora_b" in k for k in keys)
    if not (has_lora_a and has_lora_b):
        raise ValueError("no LoRA weights found (missing lora_A/lora_B keys)")
