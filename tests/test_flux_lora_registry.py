"""Tests for flux_lora_registry — filesystem-backed Flux 2 LoRA discovery."""

import json
import struct
import tempfile
from pathlib import Path

import pytest

from flux_lora_registry import FluxLoRARegistry, FluxLoRAInfo, _validate_safetensors


def _make_lora_safetensors(path: Path, include_lora_b: bool = True) -> None:
    """Write a minimal valid safetensors file with lora_A/lora_B keys (no tensor data)."""
    header = {
        "double_blocks.0.img_attn.to_q.lora_A.weight": {"dtype": "BF16", "shape": [16, 128], "data_offsets": [0, 0]},
    }
    if include_lora_b:
        header["double_blocks.0.img_attn.to_q.lora_B.weight"] = {"dtype": "BF16", "shape": [128, 16], "data_offsets": [0, 0]}
    header["__metadata__"] = {"format": "pt"}
    header_bytes = json.dumps(header).encode("utf-8")
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        # No tensor data — offsets are [0,0] so validation won't read past header


def _make_non_lora_safetensors(path: Path) -> None:
    """Write a safetensors file with no LoRA keys."""
    header = {
        "some.weight": {"dtype": "BF16", "shape": [16, 16], "data_offsets": [0, 0]},
        "__metadata__": {"format": "pt"},
    }
    header_bytes = json.dumps(header).encode("utf-8")
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)


# ---- slug ----


def test_slugify_basic():
    assert FluxLoRARegistry._slugify("MyStyle") == "mystyle"


def test_slugify_strips_special_chars():
    assert FluxLoRARegistry._slugify("Cool Style v2.0!") == "cool-style-v2-0"


def test_slugify_collapses_runs():
    assert FluxLoRARegistry._slugify("a   b   c") == "a-b-c"


def test_slugify_preserves_underscore_and_dash():
    assert FluxLoRARegistry._slugify("foo_bar-baz") == "foo_bar-baz"


def test_slugify_empty_fallback():
    assert FluxLoRARegistry._slugify("") == "unnamed"
    assert FluxLoRARegistry._slugify("...") == "unnamed"


# ---- header validation ----


def test_validate_rejects_tiny_file(tmp_path):
    p = tmp_path / "bogus.safetensors"
    p.write_bytes(b"nope")
    with pytest.raises(ValueError, match="too small"):
        _validate_safetensors(p)


def test_validate_rejects_no_lora_keys(tmp_path):
    p = tmp_path / "not_a_lora.safetensors"
    _make_non_lora_safetensors(p)
    with pytest.raises(ValueError, match="lora_A/lora_B"):
        _validate_safetensors(p)


def test_validate_accepts_valid_lora(tmp_path):
    p = tmp_path / "valid.safetensors"
    _make_lora_safetensors(p)
    _validate_safetensors(p)  # should not raise


def test_validate_rejects_missing_lora_b(tmp_path):
    p = tmp_path / "half.safetensors"
    _make_lora_safetensors(p, include_lora_b=False)
    with pytest.raises(ValueError, match="lora_A/lora_B"):
        _validate_safetensors(p)


# ---- registry ----


def test_empty_dir_gives_zero_count(tmp_path):
    reg = FluxLoRARegistry(tmp_path / "flux_loras")
    assert reg.count() == 0
    assert reg.list_all() == []


def test_discovers_safetensors_files(tmp_path):
    d = tmp_path / "flux_loras"
    d.mkdir()
    _make_lora_safetensors(d / "MyStyle.safetensors")
    _make_lora_safetensors(d / "OtherStyle.safetensors")
    reg = FluxLoRARegistry(d)
    assert reg.count() == 2
    ids = sorted(l.id for l in reg.list_all())
    assert ids == ["mystyle", "otherstyle"]


def test_sidecar_json_overrides_name_and_metadata(tmp_path):
    d = tmp_path / "flux_loras"
    d.mkdir()
    _make_lora_safetensors(d / "lora1.safetensors")
    (d / "lora1.json").write_text(json.dumps({
        "name": "Custom Display Name",
        "description": "A custom lora",
        "trigger_word": "trigger",
        "model_compat": ["flux2-dev"],
    }))
    reg = FluxLoRARegistry(d)
    info = reg.get("lora1")
    assert info is not None
    assert info.name == "Custom Display Name"
    assert info.description == "A custom lora"
    assert info.trigger_word == "trigger"
    assert info.model_compat == ["flux2-dev"]


def test_missing_sidecar_defaults_to_stem(tmp_path):
    d = tmp_path / "flux_loras"
    d.mkdir()
    _make_lora_safetensors(d / "FancyLora.safetensors")
    reg = FluxLoRARegistry(d)
    info = reg.get("fancylora")
    assert info is not None
    assert info.name == "FancyLora"
    assert info.model_compat == ["flux2-dev", "flux2-klein"]


def test_invalid_sidecar_json_falls_back_to_defaults(tmp_path):
    d = tmp_path / "flux_loras"
    d.mkdir()
    _make_lora_safetensors(d / "lora1.safetensors")
    (d / "lora1.json").write_text("not json at all { {")
    reg = FluxLoRARegistry(d)
    info = reg.get("lora1")
    assert info is not None
    assert info.name == "lora1"  # falls back to stem


def test_skips_invalid_safetensors(tmp_path):
    d = tmp_path / "flux_loras"
    d.mkdir()
    _make_lora_safetensors(d / "good.safetensors")
    _make_non_lora_safetensors(d / "bad.safetensors")
    reg = FluxLoRARegistry(d)
    assert reg.count() == 1
    assert reg.get("good") is not None
    assert reg.get("bad") is None


def test_resolve_path_returns_file(tmp_path):
    d = tmp_path / "flux_loras"
    d.mkdir()
    _make_lora_safetensors(d / "abc.safetensors")
    reg = FluxLoRARegistry(d)
    p = reg.resolve_path("abc")
    assert p == d / "abc.safetensors"
    assert p.exists()


def test_resolve_path_unknown_raises(tmp_path):
    reg = FluxLoRARegistry(tmp_path / "flux_loras")
    with pytest.raises(FileNotFoundError, match="Flux LoRA not found"):
        reg.resolve_path("nonexistent")


def test_rescan_picks_up_new_files(tmp_path):
    d = tmp_path / "flux_loras"
    d.mkdir()
    reg = FluxLoRARegistry(d)
    assert reg.count() == 0

    _make_lora_safetensors(d / "new.safetensors")
    assert reg.count() == 0  # not yet scanned

    reg.rescan()
    assert reg.count() == 1
    assert reg.get("new") is not None


def test_rescan_drops_removed_files(tmp_path):
    d = tmp_path / "flux_loras"
    d.mkdir()
    _make_lora_safetensors(d / "temp.safetensors")
    reg = FluxLoRARegistry(d)
    assert reg.count() == 1

    (d / "temp.safetensors").unlink()
    reg.rescan()
    assert reg.count() == 0
    assert reg.get("temp") is None


def test_rescan_is_idempotent(tmp_path):
    d = tmp_path / "flux_loras"
    d.mkdir()
    _make_lora_safetensors(d / "a.safetensors")
    _make_lora_safetensors(d / "b.safetensors")
    reg = FluxLoRARegistry(d)
    ids_before = sorted(l.id for l in reg.list_all())
    reg.rescan()
    reg.rescan()
    ids_after = sorted(l.id for l in reg.list_all())
    assert ids_before == ids_after


def test_duplicate_slugs_keep_first(tmp_path):
    d = tmp_path / "flux_loras"
    d.mkdir()
    _make_lora_safetensors(d / "My-Style.safetensors")
    _make_lora_safetensors(d / "my_style.safetensors")
    reg = FluxLoRARegistry(d)
    # Both slugify to "my-style" / "my_style" — actually slugify preserves _ so these differ
    # Better test: "MyStyle" and "mystyle" both → "mystyle"
    (d / "My-Style.safetensors").unlink()
    (d / "my_style.safetensors").unlink()
    _make_lora_safetensors(d / "MyStyle.safetensors")
    _make_lora_safetensors(d / "mystyle.safetensors")
    reg.rescan()
    assert reg.count() == 1  # one kept, one skipped as duplicate


def test_creates_dir_if_missing(tmp_path):
    d = tmp_path / "does_not_exist_yet" / "flux_loras"
    assert not d.exists()
    FluxLoRARegistry(d)
    assert d.exists()
