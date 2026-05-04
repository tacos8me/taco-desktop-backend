"""Tests for gen_config stage 1 sigma overrides + VAE tiling knobs (F-1)."""
import copy
import json
import os
from pathlib import Path

import config

config.GPU_DEVICES = []
config.API_KEYS = set()  # disable auth for tests

import split_model_manager  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from server import app  # noqa: E402

client = TestClient(app)


def _reset_config() -> None:
    """Restore defaults between tests so cross-test state can't leak."""
    split_model_manager._gen_config.clear()
    split_model_manager._gen_config.update(copy.deepcopy(split_model_manager._DEFAULT_GEN_CONFIG))


def setup_function(_func) -> None:
    _reset_config()


# ---------------------------------------------------------------------------
# stage1_sigmas validation
# ---------------------------------------------------------------------------

def test_stage1_sigmas_length_mismatch_returns_422():
    """Length must equal fast_stage1_steps + 1."""
    resp = client.post("/v1/system/config", json={"stage1_sigmas": [1.0, 0.5, 0.0]})
    assert resp.status_code == 422
    body = resp.json()
    assert body["error_code"] == "INVALID_SIGMA_LENGTH"


def test_stage1_sigmas_out_of_range_returns_422():
    """Values must be in [0, 1]."""
    bad = [1.5] + [0.5] * (split_model_manager._gen_config["fast_stage1_steps"])
    resp = client.post("/v1/system/config", json={"stage1_sigmas": bad})
    assert resp.status_code == 422
    assert resp.json()["error_code"] == "INVALID_SIGMA_RANGE"


def test_stage1_sigmas_valid_returns_200_and_persists():
    steps = split_model_manager._gen_config["fast_stage1_steps"]
    payload = [1.0 - i / steps for i in range(steps + 1)]
    resp = client.post("/v1/system/config", json={"stage1_sigmas": payload})
    assert resp.status_code == 200, resp.json()
    assert split_model_manager._gen_config["stage1_sigmas"] == payload


def test_stage1_sigmas_non_monotonic_returns_200_with_warning():
    steps = split_model_manager._gen_config["fast_stage1_steps"]
    # Strictly increasing sequence in [0,1] — non-monotonic warning, not an error.
    payload = [i / steps for i in range(steps + 1)]
    resp = client.post("/v1/system/config", json={"stage1_sigmas": payload})
    assert resp.status_code == 200
    body = resp.json()
    assert "warnings" in body
    assert any("monotonically" in w for w in body["warnings"])


def test_stage1_sigmas_null_restores_scheduler_path():
    """Setting override then null reverts _get_stage1_sigmas to LTX2Scheduler."""
    import torch

    steps = split_model_manager._gen_config["fast_stage1_steps"]
    # 1. baseline: scheduler-computed sigmas
    base = split_model_manager._get_stage1_sigmas(
        is_fast=True,
        steps=steps,
        max_shift=split_model_manager._gen_config["scheduler_max_shift"],
        base_shift=split_model_manager._gen_config["scheduler_base_shift"],
        device="cpu",
    )
    # 2. apply override
    payload = [1.0 - i / steps for i in range(steps + 1)]
    resp = client.post("/v1/system/config", json={"stage1_sigmas": payload})
    assert resp.status_code == 200
    overridden = split_model_manager._get_stage1_sigmas(
        is_fast=True,
        steps=steps,
        max_shift=split_model_manager._gen_config["scheduler_max_shift"],
        base_shift=split_model_manager._gen_config["scheduler_base_shift"],
        device="cpu",
    )
    assert torch.allclose(overridden, torch.tensor(payload, dtype=torch.float32))
    # 3. null restores scheduler computation
    resp = client.post("/v1/system/config", json={"stage1_sigmas": None})
    assert resp.status_code == 200
    restored = split_model_manager._get_stage1_sigmas(
        is_fast=True,
        steps=steps,
        max_shift=split_model_manager._gen_config["scheduler_max_shift"],
        base_shift=split_model_manager._gen_config["scheduler_base_shift"],
        device="cpu",
    )
    assert torch.allclose(restored, base)


def test_get_stage1_sigmas_override_returns_expected_tensor():
    import torch
    steps = 4
    override = [1.0, 0.75, 0.5, 0.25, 0.0]
    split_model_manager._gen_config["stage1_sigmas"] = override
    out = split_model_manager._get_stage1_sigmas(
        is_fast=True, steps=steps, max_shift=2.0, base_shift=0.95, device="cpu",
    )
    assert torch.allclose(out, torch.tensor(override, dtype=torch.float32))


# ---------------------------------------------------------------------------
# VAE tiling validation
# ---------------------------------------------------------------------------

def test_vae_spatial_tile_divisibility_rejects_500():
    resp = client.post("/v1/system/config", json={"vae_spatial_tile_px": 500})
    assert resp.status_code == 422
    assert resp.json()["error_code"] == "INVALID_VAE_TILE_SIZE"


def test_vae_spatial_tile_512_accepted():
    resp = client.post("/v1/system/config", json={"vae_spatial_tile_px": 768})
    assert resp.status_code == 200
    assert split_model_manager._gen_config["vae_spatial_tile_px"] == 768


def test_vae_spatial_overlap_exceeding_tile_rejected():
    resp = client.post("/v1/system/config", json={
        "vae_spatial_tile_px": 512, "vae_spatial_overlap_px": 512,
    })
    assert resp.status_code == 422
    assert resp.json()["error_code"] == "INVALID_VAE_OVERLAP"


def test_vae_temporal_tile_above_256_rejected():
    resp = client.post("/v1/system/config", json={"vae_temporal_tile_frames": 264})
    assert resp.status_code == 422
    assert resp.json()["error_code"] == "INVALID_VAE_TILE_SIZE"


def test_get_decode_tiling_threshold_gating():
    """num_frames <= threshold returns None; num_frames > threshold returns TilingConfig."""
    split_model_manager._gen_config["vae_tiling_threshold_frames"] = 100
    assert split_model_manager._get_decode_tiling(99) is None
    assert split_model_manager._get_decode_tiling(100) is None
    cfg = split_model_manager._get_decode_tiling(200)
    assert cfg is not None
    assert cfg.spatial_config.tile_size_in_pixels == split_model_manager._gen_config["vae_spatial_tile_px"]
    assert cfg.temporal_config.tile_size_in_frames == split_model_manager._gen_config["vae_temporal_tile_frames"]


# ---------------------------------------------------------------------------
# Atomic write + history.jsonl
# ---------------------------------------------------------------------------

def test_atomic_write_uses_tmp_then_rename(tmp_path, monkeypatch):
    cfg_path = tmp_path / ".gen_config.json"
    history_path = tmp_path / ".gen_config.history.jsonl"
    monkeypatch.setattr(split_model_manager, "_CONFIG_PATH", cfg_path)
    monkeypatch.setattr(split_model_manager, "_CONFIG_HISTORY_PATH", history_path)

    seen_tmp: list[bool] = []
    real_replace = os.replace

    def spy_replace(src, dst):
        # The .tmp file must exist on disk at the moment os.replace is called —
        # this is the atomicity contract.
        seen_tmp.append(Path(src).exists())
        return real_replace(src, dst)

    monkeypatch.setattr(split_model_manager.os, "replace", spy_replace)
    split_model_manager._save_gen_config()
    assert seen_tmp == [True]
    assert cfg_path.exists()
    assert not Path(str(cfg_path) + ".tmp").exists()


def test_history_jsonl_appends_diff_record(tmp_path, monkeypatch):
    cfg_path = tmp_path / ".gen_config.json"
    history_path = tmp_path / ".gen_config.history.jsonl"
    monkeypatch.setattr(split_model_manager, "_CONFIG_PATH", cfg_path)
    monkeypatch.setattr(split_model_manager, "_CONFIG_HISTORY_PATH", history_path)

    prev = copy.deepcopy(split_model_manager._gen_config)
    split_model_manager._gen_config["vae_spatial_tile_px"] = 384
    split_model_manager._save_gen_config(prev=prev)

    assert history_path.exists()
    lines = history_path.read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert "vae_spatial_tile_px" in rec["changed_keys"]
    assert rec["old"]["vae_spatial_tile_px"] == prev["vae_spatial_tile_px"]
    assert rec["new"]["vae_spatial_tile_px"] == 384


# ---------------------------------------------------------------------------
# Backward-compat: legacy callers without overrides see unchanged behavior
# ---------------------------------------------------------------------------

def test_legacy_config_load_fills_new_keys_from_defaults(tmp_path, monkeypatch):
    """A pre-F-1 .gen_config.json without the 7 new keys must still load + merge."""
    cfg_path = tmp_path / ".gen_config.json"
    cfg_path.write_text(json.dumps({"sampler": "cfg_pp", "fast_stage1_steps": 8}))
    monkeypatch.setattr(split_model_manager, "_CONFIG_PATH", cfg_path)

    loaded = split_model_manager._load_gen_config()
    assert loaded["stage1_sigmas"] is None
    assert loaded["vae_spatial_tile_px"] == 512
    assert loaded["vae_tiling_threshold_frames"] == 257
