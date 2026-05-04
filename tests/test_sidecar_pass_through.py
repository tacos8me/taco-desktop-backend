"""F-2 sidecar pass-through tests — operator gen_config overrides reach Modal/RunPod.

Covers:
- LtxSidecarClient.generate forwards gen_config_overrides in the JSON body when set.
- Backward compat: omitting the kwarg produces the same body as today.
- _dispatch_job_turbo_remote builds the whitelisted overrides dict from
  split_model_manager._gen_config and threads it through to the client.
- Modal/RunPod save-apply-restore pattern leaves _gen_config unchanged across requests.
- Local sidecar's _reload_gen_config picks up F-1 default keys via _load_gen_config.
"""
from __future__ import annotations

import copy
import json

import config

config.GPU_DEVICES = []
config.API_KEYS = set()

import httpx  # noqa: E402
import pytest  # noqa: E402

import split_model_manager  # noqa: E402
from ltx_sidecar_client import LtxSidecarClient  # noqa: E402


def _install_mock(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    real_cls = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs.pop("transport", None)
        return real_cls(transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)


def _reset_gen_config() -> None:
    split_model_manager._gen_config.clear()
    split_model_manager._gen_config.update(copy.deepcopy(split_model_manager._DEFAULT_GEN_CONFIG))


# ---------------------------------------------------------------------------
# Client-level: forwards overrides when set, byte-identical body when omitted
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ltx_sidecar_client_forwards_overrides(monkeypatch):
    captured: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/generate"
        captured["body"] = json.loads(req.content.decode())
        return httpx.Response(200, content=b"\x00" * 16, headers={"content-type": "video/mp4"})

    _install_mock(monkeypatch, handler)
    c = LtxSidecarClient(base_url="http://ltx.test")

    overrides = {"stage1_sigmas": [1.0, 0.5, 0.0], "vae_spatial_tile_px": 768}
    await c.generate(
        job_type="text-to-video", prompt="hi", model="ltx-2-3-fast",
        width=512, height=512, num_frames=9, fps=24.0, seed=1,
        gen_config_overrides=overrides,
    )

    body = captured["body"]
    assert "gen_config_overrides" in body
    assert body["gen_config_overrides"] == overrides


@pytest.mark.asyncio
async def test_ltx_sidecar_client_backward_compat(monkeypatch):
    """Omitting the new kwarg must NOT introduce gen_config_overrides into the payload."""
    captured: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(req.content.decode())
        return httpx.Response(200, content=b"\x00" * 16, headers={"content-type": "video/mp4"})

    _install_mock(monkeypatch, handler)
    c = LtxSidecarClient(base_url="http://ltx.test")

    await c.generate(
        job_type="text-to-video", prompt="hi", model="ltx-2-3-fast",
        width=512, height=512, num_frames=9, fps=24.0, seed=1,
    )

    body = captured["body"]
    assert "gen_config_overrides" not in body


@pytest.mark.asyncio
async def test_ltx_sidecar_client_empty_overrides_omitted(monkeypatch):
    """Empty dict is falsy — should be treated like None for backward compat."""
    captured: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(req.content.decode())
        return httpx.Response(200, content=b"\x00" * 16, headers={"content-type": "video/mp4"})

    _install_mock(monkeypatch, handler)
    c = LtxSidecarClient(base_url="http://ltx.test")

    await c.generate(
        job_type="text-to-video", prompt="hi", model="ltx-2-3-fast",
        width=512, height=512, num_frames=9, fps=24.0, seed=1,
        gen_config_overrides={},
    )

    body = captured["body"]
    assert "gen_config_overrides" not in body


# ---------------------------------------------------------------------------
# Dispatch-level: turbo remote dispatch builds whitelisted overrides dict
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_turbo_dispatch_threads_overrides_through(monkeypatch):
    """_dispatch_job_turbo_remote should ship every OPERATOR_TUNABLE_GEN_CONFIG_KEY."""
    import server
    from job_queue import Job, JobType

    _reset_gen_config()
    # Mutate a F-1 key so we can verify it's actually picked up.
    split_model_manager._gen_config["vae_spatial_tile_px"] = 768
    split_model_manager._gen_config["stage1_sigmas"] = [1.0, 0.5, 0.0]

    captured: dict = {}

    class _FakeClient:
        async def generate(self, **kwargs):
            captured["kwargs"] = kwargs
            return b"\x00" * 16

    monkeypatch.setitem(server.ltx_remote_sidecars, "modal", _FakeClient())

    job = Job(
        id="test-job",
        type=JobType.TEXT_TO_VIDEO,
        params={
            "prompt": "hello",
            "model": "ltx-2-3-fast",
            "width": 512, "height": 512, "num_frames": 9,
            "fps": 24, "seed": 1, "generate_audio": False,
        },
    )

    await server._dispatch_job_turbo_remote(job, provider="modal")

    overrides = captured["kwargs"]["gen_config_overrides"]
    # Every operator-tunable key should be present.
    for key in server.OPERATOR_TUNABLE_GEN_CONFIG_KEYS:
        assert key in overrides, f"missing {key}"
    # Mutated values arrive intact.
    assert overrides["vae_spatial_tile_px"] == 768
    assert overrides["stage1_sigmas"] == [1.0, 0.5, 0.0]


def test_operator_tunable_keys_subset_of_defaults():
    """Whitelist must not name keys that don't exist in _DEFAULT_GEN_CONFIG."""
    import server

    defaults = set(split_model_manager._DEFAULT_GEN_CONFIG.keys())
    extras = server.OPERATOR_TUNABLE_GEN_CONFIG_KEYS - defaults
    assert not extras, f"whitelist names unknown keys: {extras}"


# ---------------------------------------------------------------------------
# Save/apply/restore pattern (Modal + RunPod parity)
# ---------------------------------------------------------------------------

def test_save_apply_restore_pattern():
    """Simulates the Modal/RunPod handler: apply overrides, run, restore."""
    _reset_gen_config()
    pre = copy.deepcopy(split_model_manager._gen_config)

    overrides = {
        "vae_spatial_tile_px": 768,
        "stage1_sigmas": [1.0, 0.5, 0.0],
        "cfg_scale": 4.5,
        "unknown_key_should_be_ignored": "garbage",
    }

    # --- mirror of the handler logic ---
    saved = {
        k: split_model_manager._gen_config.get(k)
        for k in overrides
        if k in split_model_manager._gen_config
    }
    for k, v in overrides.items():
        if k in split_model_manager._gen_config:
            split_model_manager._gen_config[k] = v

    # Mid-request: overrides applied.
    assert split_model_manager._gen_config["vae_spatial_tile_px"] == 768
    assert split_model_manager._gen_config["stage1_sigmas"] == [1.0, 0.5, 0.0]
    assert split_model_manager._gen_config["cfg_scale"] == 4.5
    # Unknown key never landed.
    assert "unknown_key_should_be_ignored" not in split_model_manager._gen_config

    # Restore.
    for k, v in saved.items():
        split_model_manager._gen_config[k] = v

    # Post-handler: state must match pre-handler.
    assert split_model_manager._gen_config == pre


# ---------------------------------------------------------------------------
# Local sidecar verification: _load_gen_config returns F-1 keys from defaults
# ---------------------------------------------------------------------------

def test_local_sidecar_load_picks_up_new_keys(tmp_path, monkeypatch):
    """_load_gen_config (used by sidecar's _reload_gen_config) must surface F-1 keys."""
    # Point at a nonexistent path — _load_gen_config should fall back to
    # _DEFAULT_GEN_CONFIG which now includes the F-1 keys.
    monkeypatch.setattr(split_model_manager, "_CONFIG_PATH", tmp_path / "missing.json")
    cfg = split_model_manager._load_gen_config()

    for key in (
        "stage1_sigmas",
        "pro_stage1_sigmas",
        "vae_spatial_tile_px",
        "vae_spatial_overlap_px",
        "vae_temporal_tile_frames",
        "vae_temporal_overlap_frames",
        "vae_tiling_threshold_frames",
    ):
        assert key in cfg, f"_load_gen_config missing F-1 key {key}"
