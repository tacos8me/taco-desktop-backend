"""Phase 1 — L1 threshold endpoints + recommend-loras session_boosts.

Tests for the v1.19.0+ Phase 1 ship:

  - GET / POST /v1/api-keys/me/validator-thresholds — round-trip + clamp
    + inversion 422.
  - composite() honors per-bearer (pass, retake) overrides; an in-range
    override actually flips a borderline clip's recommendation.
  - POST /v2/embeddings/recommend-loras accepts optional ``session_boosts``;
    each per-LoRA boost is clamped to ±0.10; missing/null map → byte-
    identical behavior to pre-rc1.
"""
from __future__ import annotations

import struct
import time
from unittest.mock import patch

import pytest

import config

# Mirror existing v1_18 + v1_19 test patterns: disable auth + GPU, then
# import server.
config.GPU_DEVICES = []
config.API_KEYS = set()

from fastapi.testclient import TestClient  # noqa: E402

import history_store  # noqa: E402
from history_store import HistoryStore, _hash_key  # noqa: E402
import server as server_mod  # noqa: E402
from server import app  # noqa: E402


client = TestClient(app)


EMBED_DIM = 4096


def _pack_embed(vec: list[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def _unit_embed(seed: int) -> bytes:
    import random
    rng = random.Random(seed)
    return _pack_embed([rng.uniform(-1.0, 1.0) for _ in range(EMBED_DIM)])


def _seed_opt_in(history: HistoryStore, api_key: str) -> None:
    now = time.time()
    history._conn.execute(
        """INSERT OR REPLACE INTO api_key_metadata
           (api_key_hash, training_opt_in, tier, notes, created_at, updated_at)
           VALUES (?, 1, 'pro', NULL, ?, ?)""",
        (_hash_key(api_key), now, now),
    )
    history._conn.commit()


def _insert_clip(
    history: HistoryStore,
    *,
    clip_id: str,
    api_key: str,
    prompt: str,
    embedding: bytes,
    validator_score: float = 0.85,
    lora_applied_id: str | None = None,
    lora_applied_strength: float | None = None,
) -> None:
    now = time.time()
    history._conn.execute(
        """INSERT OR REPLACE INTO generations
           (id, api_key_hash, job_type, prompt, model, width, height, turbo,
            status, result_uri, created_at,
            validator_score, validator_version,
            lora_applied_id, lora_applied_strength)
           VALUES (?, ?, 'text-to-video', ?, 'ltx-2-3-fast', 512, 512, 0,
                   'completed', ?, ?, ?, ?, ?, ?)""",
        (
            clip_id, _hash_key(api_key), prompt, f"storage://{clip_id}", now,
            validator_score, config.VALIDATOR_VERSION,
            lora_applied_id, lora_applied_strength,
        ),
    )
    history._conn.execute(
        """INSERT OR IGNORE INTO clip_embeddings (id, embedding, embedding_model_version)
           VALUES (?, ?, ?)""",
        (clip_id, embedding, "gemma-3-12b-nvfp4"),
    )
    history._conn.commit()


@pytest.fixture(autouse=True)
def _reset_metrics():
    server_mod._embeddings_buckets.clear()
    for k in server_mod._embeddings_metrics:
        server_mod._embeddings_metrics[k] = 0
    server_mod._embeddings_search_latencies_ms.clear()
    yield


@pytest.fixture
def fresh_history(tmp_path, monkeypatch):
    db = tmp_path / "history.db"
    h = HistoryStore(db_path=db)
    monkeypatch.setattr(server_mod, "history", h)
    yield h
    h._conn.close()


@pytest.fixture
def patch_embed():
    async def fake_embed(text: str) -> bytes:
        seed = abs(hash(text)) % (2 ** 31)
        return _unit_embed(seed)

    async def fake_embed_batch(texts: list[str]) -> list[bytes]:
        return [await fake_embed(t) for t in texts]

    with patch.object(server_mod.chat, "embed", side_effect=fake_embed), \
         patch.object(server_mod.chat, "embed_batch", side_effect=fake_embed_batch):
        yield


# ---------------------------------------------------------------------------
# Threshold endpoints
# ---------------------------------------------------------------------------


def test_threshold_round_trip(fresh_history):
    """GET returns nulls + globals before any POST; POST persists; GET reads back."""
    api_key = "bearer-thresh-1"
    _seed_opt_in(fresh_history, api_key)

    resp = client.get(
        "/v1/api-keys/me/validator-thresholds",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["pass"] is None
    assert data["retake"] is None
    assert data["fallback_pass"] == 0.65
    assert data["fallback_retake"] == 0.45

    resp2 = client.post(
        "/v1/api-keys/me/validator-thresholds",
        json={"pass": 0.70, "retake": 0.40},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp2.status_code == 200, resp2.text
    data2 = resp2.json()
    assert data2["pass"] == pytest.approx(0.70)
    assert data2["retake"] == pytest.approx(0.40)

    resp3 = client.get(
        "/v1/api-keys/me/validator-thresholds",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp3.json()["pass"] == pytest.approx(0.70)
    assert resp3.json()["retake"] == pytest.approx(0.40)

    # Clear via null
    resp4 = client.post(
        "/v1/api-keys/me/validator-thresholds",
        json={"pass": None, "retake": None},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp4.status_code == 200, resp4.text
    assert resp4.json()["pass"] is None
    assert resp4.json()["retake"] is None


def test_threshold_clamp_rejects_outside_range(fresh_history):
    """global ±0.15: 0.85 (=0.65+0.20) is outside; 422.

    # safety-critical: invariant-14 (threshold sanity — global ±0.15 clamp)
    """
    api_key = "bearer-thresh-clamp"
    _seed_opt_in(fresh_history, api_key)

    resp = client.post(
        "/v1/api-keys/me/validator-thresholds",
        json={"pass": 0.85, "retake": 0.40},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp.status_code == 422, resp.text

    resp2 = client.post(
        "/v1/api-keys/me/validator-thresholds",
        json={"pass": 0.70, "retake": 0.20},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp2.status_code == 422, resp2.text


def test_threshold_inversion_rejected(fresh_history):
    """pass <= retake → 422 INVALID_THRESHOLD_INVERSION.

    # safety-critical: invariant-14 (threshold sanity — pass > retake)
    """
    api_key = "bearer-thresh-inv"
    _seed_opt_in(fresh_history, api_key)

    resp = client.post(
        "/v1/api-keys/me/validator-thresholds",
        json={"pass": 0.50, "retake": 0.60},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp.status_code == 422, resp.text
    body = resp.json()
    assert body.get("error_code") == "INVALID_THRESHOLD_INVERSION"

    # Also: equal → still 422 (strict greater-than).
    resp_eq = client.post(
        "/v1/api-keys/me/validator-thresholds",
        json={"pass": 0.55, "retake": 0.55},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp_eq.status_code == 422


def test_threshold_inversion_via_single_knob_update(fresh_history):
    """Two-step inversion: store retake=0.50 first, then POST {pass: 0.40} —
    must be caught even though only one knob is in this body.

    # safety-critical: invariant-14 (threshold sanity — two-step inversion)
    """
    api_key = "bearer-thresh-2step"
    _seed_opt_in(fresh_history, api_key)

    r1 = client.post(
        "/v1/api-keys/me/validator-thresholds",
        json={"retake": 0.50},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert r1.status_code == 200, r1.text

    r2 = client.post(
        "/v1/api-keys/me/validator-thresholds",
        json={"pass": 0.55},  # 0.55 > 0.50 (stored retake) → 422 because 0.55 - 0.50 < strict
        headers={"Authorization": f"Bearer {api_key}"},
    )
    # 0.55 > 0.50, so this one IS valid; verify validation behaves correctly.
    assert r2.status_code == 200, r2.text

    r3 = client.post(
        "/v1/api-keys/me/validator-thresholds",
        json={"pass": 0.50},  # equal to stored retake → invalid (strict)
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert r3.status_code == 422
    assert r3.json().get("error_code") == "INVALID_THRESHOLD_INVERSION"


def test_composite_honors_per_bearer_pass_threshold():
    """Borderline clip at composite=0.68: default → pass; with pass=0.70 → warn."""
    from validator import composite

    # Choose tiers so composite = 0.4*1.0 + 0.2*1.0 + 0.4*0.2 = 0.68
    tier1 = {"dynamic_degree": 5.0, "flow_windows": [1, 1, 1, 1], "motion_smoothness": 0.9}
    tier2 = {"tier2_skipped": True}  # contributes 1.0
    tier3 = {"verdict": "pass", "score": 0.2, "judge_score": 0.2}

    out_default = composite(tier1, tier2, tier3)
    assert out_default["composite_score"] == pytest.approx(0.68, abs=1e-3)
    assert out_default["recommendation"] == "pass"

    out_strict = composite(
        tier1, tier2, tier3,
        pass_threshold=0.70, retake_threshold=0.40,
    )
    assert out_strict["composite_score"] == pytest.approx(0.68, abs=1e-3)
    assert out_strict["recommendation"] == "warn"


def test_composite_threshold_kwargs_default_when_none():
    """pass_threshold=None / retake_threshold=None must fall back to globals."""
    from validator import composite, GLOBAL_PASS_THRESHOLD, GLOBAL_RETAKE_THRESHOLD

    tier1 = {"dynamic_degree": 5.0, "flow_windows": [1, 1, 1, 1], "motion_smoothness": 0.9}
    tier2 = {"tier2_skipped": True}
    tier3 = {"verdict": "pass", "score": 0.5, "judge_score": 0.5}

    out_a = composite(tier1, tier2, tier3)
    out_b = composite(tier1, tier2, tier3, pass_threshold=None, retake_threshold=None)
    assert out_a == out_b
    # Global semantics intact.
    assert GLOBAL_PASS_THRESHOLD == 0.65
    assert GLOBAL_RETAKE_THRESHOLD == 0.45


# ---------------------------------------------------------------------------
# recommend-loras session_boosts
# ---------------------------------------------------------------------------


def _seed_two_loras_for_recommend(history: HistoryStore, api_key: str) -> None:
    """Seed clips so per-LoRA aggregation is deterministic."""
    _seed_opt_in(history, api_key)
    # 5 clips with lora-A at validator_score=0.80
    for i in range(5):
        _insert_clip(
            history,
            clip_id=f"a-{i}",
            api_key=api_key,
            prompt="boosting test prompt",
            embedding=_unit_embed(i + 1000),
            validator_score=0.80,
            lora_applied_id="lora-A",
            lora_applied_strength=0.5,
        )
    # 5 clips with lora-B at validator_score=0.78 (slightly lower)
    for i in range(5):
        _insert_clip(
            history,
            clip_id=f"b-{i}",
            api_key=api_key,
            prompt="boosting test prompt",
            embedding=_unit_embed(i + 2000),
            validator_score=0.78,
            lora_applied_id="lora-B",
            lora_applied_strength=0.5,
        )


def test_recommend_loras_session_boosts_clamped(fresh_history, patch_embed):
    """session_boosts={lora-B: 0.20} → clamped to +0.10 server-side."""
    api_key = "bearer-boost-clamp"
    _seed_two_loras_for_recommend(fresh_history, api_key)

    resp = client.post(
        "/v2/embeddings/recommend-loras",
        json={
            "prompt": "boosting test prompt",
            "k": 5,
            "session_boosts": {"lora-B": 0.20, "lora-A": -0.50},
        },
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp.status_code == 200, resp.text
    recs = {r["lora_id"]: r for r in resp.json()["recommendations"]}
    assert recs["lora-B"]["session_boost"] == pytest.approx(0.10)
    assert recs["lora-A"]["session_boost"] == pytest.approx(-0.10)


def test_recommend_loras_no_session_boosts_backward_compat(fresh_history, patch_embed):
    """Missing session_boosts → ranking unchanged from pre-rc1 formula
    (0.7·mean + 0.3·max(0, expected_boost)). session_boost field present
    but 0.0 across the board."""
    api_key = "bearer-boost-compat"
    _seed_two_loras_for_recommend(fresh_history, api_key)

    resp = client.post(
        "/v2/embeddings/recommend-loras",
        json={"prompt": "boosting test prompt", "k": 5},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp.status_code == 200, resp.text
    for r in resp.json()["recommendations"]:
        assert r["session_boost"] == 0.0


def test_recommend_loras_session_boost_changes_order(fresh_history, patch_embed):
    """Without boosts, lora-A (mean 0.80) ranks above lora-B (mean 0.78);
    a +0.10 boost on lora-B flips the order."""
    api_key = "bearer-boost-flip"
    _seed_two_loras_for_recommend(fresh_history, api_key)

    base = client.post(
        "/v2/embeddings/recommend-loras",
        json={"prompt": "boosting test prompt", "k": 5},
        headers={"Authorization": f"Bearer {api_key}"},
    ).json()["recommendations"]
    assert base[0]["lora_id"] == "lora-A"
    assert base[1]["lora_id"] == "lora-B"

    boosted = client.post(
        "/v2/embeddings/recommend-loras",
        json={
            "prompt": "boosting test prompt",
            "k": 5,
            "session_boosts": {"lora-B": 0.10},
        },
        headers={"Authorization": f"Bearer {api_key}"},
    ).json()["recommendations"]
    assert boosted[0]["lora_id"] == "lora-B"


def test_recommend_loras_null_session_boosts(fresh_history, patch_embed):
    """Explicit null is accepted (Pydantic Optional) and is byte-equivalent
    to omitting the field."""
    api_key = "bearer-boost-null"
    _seed_two_loras_for_recommend(fresh_history, api_key)

    resp = client.post(
        "/v2/embeddings/recommend-loras",
        json={"prompt": "boosting test prompt", "k": 5, "session_boosts": None},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp.status_code == 200, resp.text
    for r in resp.json()["recommendations"]:
        assert r["session_boost"] == 0.0


# ---------------------------------------------------------------------------
# Phase 1: rating endpoints (5 endpoints, 12 tests)
#
# Plan ref: melodic-sniffing-beacon.md §"Endpoints",
# §"preference_pairs Writer Logic" (Path C′),
# §"Rating submission contract", §"Rating mutation contracts",
# §"Active-learning queue contract".
# ---------------------------------------------------------------------------


_R_KEY_A = "rating-bearer-alpha"
_R_KEY_B = "rating-bearer-bravo"
_R_KEY_ADMIN = "rating-bearer-admin"
_R_HASH_A = _hash_key(_R_KEY_A)
_R_HASH_B = _hash_key(_R_KEY_B)


@pytest.fixture
def rating_db(fresh_history, monkeypatch):
    """Seed both bearers as opted-in for training, no cross-bearer consent.

    Anti-fatigue caps (``_RATING_MIN_INTERVAL_S`` and
    ``_RATING_SESSION_CAP``) are relaxed by default so existing tests can
    rapid-fire POSTs to verify idempotency / supersedence. Tests that
    specifically exercise the gate set their own values.
    """
    monkeypatch.setattr(server_mod, "_RATING_MIN_INTERVAL_S", 0.0)
    monkeypatch.setattr(server_mod, "_RATING_SESSION_CAP", 100_000)
    monkeypatch.setattr(config, "API_KEYS",
                        {_R_KEY_A, _R_KEY_B, _R_KEY_ADMIN})
    monkeypatch.setattr(config, "ADMIN_KEYS", {_R_KEY_ADMIN})
    now = time.time()
    fresh_history._conn.executemany(
        """INSERT OR REPLACE INTO api_key_metadata
           (api_key_hash, training_opt_in, tier, notes, created_at, updated_at)
           VALUES (?, 1, 'pro', NULL, ?, ?)""",
        [(_R_HASH_A, now, now), (_R_HASH_B, now, now)],
    )
    fresh_history._conn.commit()
    yield fresh_history


def _r_insert_clip(
    h: HistoryStore,
    *,
    clip_id: str,
    bearer_hash: str,
    validator_score: float | None,
    validator_version: str | None = "1.19.0-rc1",
    shot_config_key: str | None = None,
) -> None:
    h._conn.execute(
        """INSERT INTO generations
           (id, api_key_hash, job_type, prompt, model, width, height, turbo,
            status, result_uri, created_at, validator_score,
            validator_version, shot_config_key)
           VALUES (?, ?, 'audio-to-video', 'p', 'ltx-2-3-fast',
                   512, 512, 0, 'completed', ?, ?, ?, ?, ?)""",
        (
            clip_id, bearer_hash, f"storage://{clip_id}", time.time(),
            validator_score, validator_version, shot_config_key,
        ),
    )
    h._conn.commit()


def _r_auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def _r_pair_payload(partner: str = "partner-1", lr_seed: int = 42) -> dict:
    return {
        "validator_visible_at_rating": False,
        "pair_partner_clip_id": partner,
        "lr_seed": lr_seed,
    }


def test_rating_post_pair_writes_human_ratings_and_preference_pairs(rating_db):
    """1. POST pair_chose_a inserts both human_ratings + preference_pairs
    with signal_strength=0.75, signal_source='human_rating',
    pending_construction_until ~= now+86400 (Path C′ 24h quarantine).

    # safety-critical: P0-4 (rater_hash captured), P0-5 (24h quarantine),
    #                  invariant-8 (signal_strength=0.75), invariant-9 (lr_seed audit logged),
    #                  invariant-11 (Path C′ pending_construction_until)
    """
    _r_insert_clip(rating_db, clip_id="rc1", bearer_hash=_R_HASH_A,
                   validator_score=0.55)
    _r_insert_clip(rating_db, clip_id="rc2", bearer_hash=_R_HASH_A,
                   validator_score=0.5)

    before = time.time()
    resp = client.post(
        "/v1/clips/rc1/rating",
        json={"kind": "pair_chose_a", "value": 1.0,
              "payload": _r_pair_payload("rc2", lr_seed=12345)},
        headers=_r_auth(_R_KEY_A),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["kind"] == "pair_chose_a"
    assert body["pair_id"] is not None
    assert body["validator_composite_at_rating"] == 0.55
    assert body["validator_version"] == "1.19.0-rc1"

    pair = rating_db._conn.execute(
        "SELECT chosen_clip_id, rejected_clip_id, signal_source, "
        "signal_strength, validator_version, pending_construction_until "
        "FROM preference_pairs WHERE pair_id = ?",
        (body["pair_id"],),
    ).fetchone()
    assert pair["chosen_clip_id"] == "rc1"
    assert pair["rejected_clip_id"] == "rc2"
    assert pair["signal_source"] == "human_rating"
    assert pair["signal_strength"] == 0.75
    assert pair["validator_version"] == "1.19.0-rc1"
    assert pair["pending_construction_until"] is not None
    delta = pair["pending_construction_until"] - before
    assert 86399 <= delta <= 86402, f"24h quarantine off: delta={delta}"

    # lr_seed audit logged in payload
    rating_row = rating_db._conn.execute(
        "SELECT rating_payload_json FROM human_ratings WHERE rating_id = ?",
        (body["rating_id"],),
    ).fetchone()
    import json as _json
    pl = _json.loads(rating_row["rating_payload_json"])
    assert pl["lr_seed"] == 12345


def test_rating_post_audit_only_kinds_skip_preference_pairs(rating_db):
    """2. pair_tie / tag / warn are audit-only — no preference_pairs row."""
    _r_insert_clip(rating_db, clip_id="rc1", bearer_hash=_R_HASH_A,
                   validator_score=0.5)

    for kind in ("warn", "tag", "pair_tie"):
        payload = {"validator_visible_at_rating": False}
        if kind == "pair_tie":
            payload.update({"pair_partner_clip_id": "rc2", "lr_seed": 1})
        resp = client.post(
            "/v1/clips/rc1/rating",
            json={"kind": kind, "value": 0.0, "payload": payload},
            headers=_r_auth(_R_KEY_A),
        )
        assert resp.status_code == 200, f"kind={kind}: {resp.text}"
        assert resp.json()["pair_id"] is None

    pp_count = rating_db._conn.execute(
        "SELECT COUNT(*) AS c FROM preference_pairs WHERE signal_source='human_rating'"
    ).fetchone()["c"]
    assert pp_count == 0


def test_rating_post_idempotent_resubmit_supersedes(rating_db):
    """3. Re-rating same (rater, clip, kind) supersedes the prior row.

    Only ONE active row at a time per the partial UNIQUE index.

    # safety-critical: invariant-5 (idempotence + audit chain via partial UNIQUE)
    """
    _r_insert_clip(rating_db, clip_id="rc1", bearer_hash=_R_HASH_A,
                   validator_score=0.5)
    _r_insert_clip(rating_db, clip_id="rc2", bearer_hash=_R_HASH_A,
                   validator_score=0.5)

    r1 = client.post(
        "/v1/clips/rc1/rating",
        json={"kind": "pair_chose_a", "value": 1.0,
              "payload": _r_pair_payload("rc2")},
        headers=_r_auth(_R_KEY_A),
    )
    assert r1.status_code == 200
    rid_old = r1.json()["rating_id"]

    r2 = client.post(
        "/v1/clips/rc1/rating",
        json={"kind": "pair_chose_a", "value": 1.0,
              "payload": _r_pair_payload("rc2")},
        headers=_r_auth(_R_KEY_A),
    )
    assert r2.status_code == 200, r2.text
    rid_new = r2.json()["rating_id"]
    assert rid_new != rid_old
    assert r2.json()["superseded_rating_id"] == rid_old

    active = rating_db._conn.execute(
        "SELECT COUNT(*) AS c FROM human_ratings "
        "WHERE rater_api_key_hash = ? AND clip_id = ? AND rating_kind = ? "
        "  AND retracted_at IS NULL AND superseded_by IS NULL",
        (_R_HASH_A, "rc1", "pair_chose_a"),
    ).fetchone()["c"]
    assert active == 1, "partial UNIQUE index must yield exactly one active row"

    superseded = rating_db._conn.execute(
        "SELECT superseded_by FROM human_ratings WHERE rating_id = ?",
        (rid_old,),
    ).fetchone()["superseded_by"]
    assert superseded == rid_new


def test_rating_post_privacy_gate_blocks_cross_bearer(rating_db):
    """4. Cross-bearer rating without consent → 409 PRIVACY_GATE_BLOCKED.
    Granting consent unblocks the same caller.

    # safety-critical: invariant-3 (cross-bearer rating rejected without consent),
    #                  invariant-10 (cross_bearer_rating_consent_at gate),
    #                  attack-pattern (cross-bearer without consent → 409)
    """
    _r_insert_clip(rating_db, clip_id="rb1", bearer_hash=_R_HASH_B,
                   validator_score=0.5)
    _r_insert_clip(rating_db, clip_id="rb2", bearer_hash=_R_HASH_B,
                   validator_score=0.5)

    resp = client.post(
        "/v1/clips/rb1/rating",
        json={"kind": "pair_chose_a", "value": 1.0,
              "payload": _r_pair_payload("rb2")},
        headers=_r_auth(_R_KEY_A),
    )
    assert resp.status_code == 409
    body = resp.json()
    assert body["error_code"] == "PRIVACY_GATE_BLOCKED"
    assert body["error"] == "privacy_gate_blocked"

    rating_db._conn.execute(
        "UPDATE api_key_metadata SET cross_bearer_rating_consent_at = ? "
        "WHERE api_key_hash = ?",
        (time.time(), _R_HASH_B),
    )
    rating_db._conn.commit()
    resp2 = client.post(
        "/v1/clips/rb1/rating",
        json={"kind": "pair_chose_a", "value": 1.0,
              "payload": _r_pair_payload("rb2")},
        headers=_r_auth(_R_KEY_A),
    )
    assert resp2.status_code == 200, resp2.text


def test_rating_post_validator_composite_null_rejected(rating_db):
    """5. Clip with NULL validator_score → 409 VALIDATOR_COMPOSITE_NULL.

    # safety-critical: invariant-2 (validator composite NOT NULL pre-pair-write),
    #                  attack-pattern (NULL composite → 409)
    """
    _r_insert_clip(rating_db, clip_id="rc1", bearer_hash=_R_HASH_A,
                   validator_score=None)
    resp = client.post(
        "/v1/clips/rc1/rating",
        json={"kind": "warn", "value": -1.0,
              "payload": {"validator_visible_at_rating": False}},
        headers=_r_auth(_R_KEY_A),
    )
    assert resp.status_code == 409
    assert resp.json()["error_code"] == "VALIDATOR_COMPOSITE_NULL"


def test_rating_post_missing_validator_visible_flag_422(rating_db):
    """6. Missing payload.validator_visible_at_rating → 422.

    # safety-critical: P0-3 (validator_visible_at_rating REQUIRED — no default by design),
    #                  attack-pattern (omit validator_visible → 422)
    """
    _r_insert_clip(rating_db, clip_id="rc1", bearer_hash=_R_HASH_A,
                   validator_score=0.5)
    resp = client.post(
        "/v1/clips/rc1/rating",
        json={"kind": "warn", "value": -1.0, "payload": {}},
        headers=_r_auth(_R_KEY_A),
    )
    assert resp.status_code == 422
    assert resp.json()["error_code"] == "MISSING_VALIDATOR_VISIBLE_FLAG"


def test_rating_post_invalid_kind_422(rating_db):
    """7. kind not in enum → 422 INVALID_KIND. Also: signal_strength bound
    is hard-coded at 0.75 via _RATING_SIGNAL_STRENGTH (defense-in-depth check).

    # safety-critical: P0-1 (single-dim Likert rejected — only the 5-kind enum allowed),
    #                  invariant-8 (signal_strength locked at 0.75)
    """
    _r_insert_clip(rating_db, clip_id="rc1", bearer_hash=_R_HASH_A,
                   validator_score=0.5)
    resp = client.post(
        "/v1/clips/rc1/rating",
        json={"kind": "good", "value": 1.0,
              "payload": {"validator_visible_at_rating": False}},
        headers=_r_auth(_R_KEY_A),
    )
    assert resp.status_code == 422
    assert resp.json()["error_code"] == "INVALID_KIND"
    # Plan §safety #8: 0.7 < 0.75 < 0.9 — locked at module level.
    assert server_mod._RATING_SIGNAL_STRENGTH == 0.75


def test_rating_delete_retracts_and_deletes_unconsumed_pair(rating_db):
    """8. DELETE soft-deletes human_ratings + DELETEs unconsumed pair.
    Wrong bearer → 403 NOT_OWNER.

    # safety-critical: invariant-5 (audit chain via retracted_at)
    """
    _r_insert_clip(rating_db, clip_id="rc1", bearer_hash=_R_HASH_A,
                   validator_score=0.5)
    _r_insert_clip(rating_db, clip_id="rc2", bearer_hash=_R_HASH_A,
                   validator_score=0.5)
    r1 = client.post(
        "/v1/clips/rc1/rating",
        json={"kind": "pair_chose_a", "value": 1.0,
              "payload": _r_pair_payload("rc2")},
        headers=_r_auth(_R_KEY_A),
    )
    rid = r1.json()["rating_id"]
    pid = r1.json()["pair_id"]

    resp_403 = client.delete(
        f"/v1/clips/rc1/rating/{rid}", headers=_r_auth(_R_KEY_B),
    )
    assert resp_403.status_code == 403
    assert resp_403.json()["error_code"] == "NOT_OWNER"

    resp = client.delete(
        f"/v1/clips/rc1/rating/{rid}", headers=_r_auth(_R_KEY_A),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["preference_pair_deleted"] is True
    assert resp.json()["pair_id"] == pid

    pp_row = rating_db._conn.execute(
        "SELECT 1 FROM preference_pairs WHERE pair_id = ?", (pid,),
    ).fetchone()
    assert pp_row is None
    rt = rating_db._conn.execute(
        "SELECT retracted_at FROM human_ratings WHERE rating_id = ?", (rid,),
    ).fetchone()["retracted_at"]
    assert rt is not None


def test_rating_delete_consumed_pair_is_immutable(rating_db):
    """9. Pair already consumed by training run is preserved on retraction.
    Retraction soft-deletes the rating but doesn't unwind the trained-against
    artifact (plan §"Rating mutation contracts").
    """
    _r_insert_clip(rating_db, clip_id="rc1", bearer_hash=_R_HASH_A,
                   validator_score=0.5)
    _r_insert_clip(rating_db, clip_id="rc2", bearer_hash=_R_HASH_A,
                   validator_score=0.5)
    r1 = client.post(
        "/v1/clips/rc1/rating",
        json={"kind": "pair_chose_a", "value": 1.0,
              "payload": _r_pair_payload("rc2")},
        headers=_r_auth(_R_KEY_A),
    )
    rid = r1.json()["rating_id"]
    pid = r1.json()["pair_id"]
    rating_db._conn.execute(
        "UPDATE preference_pairs SET used_in_training_run_id = 'run_xyz' "
        "WHERE pair_id = ?",
        (pid,),
    )
    rating_db._conn.commit()

    resp = client.delete(
        f"/v1/clips/rc1/rating/{rid}", headers=_r_auth(_R_KEY_A),
    )
    assert resp.status_code == 200
    assert resp.json()["preference_pair_deleted"] is False
    pp = rating_db._conn.execute(
        "SELECT used_in_training_run_id FROM preference_pairs WHERE pair_id = ?",
        (pid,),
    ).fetchone()
    assert pp is not None
    assert pp["used_in_training_run_id"] == "run_xyz"


def test_rating_get_truncates_hash_for_non_admin(rating_db):
    """10. GET /v1/clips/{id}/ratings — non-admin caller sees rater_api_key_hash
    truncated to 8 chars; admin sees full hash; unrelated bearer → 403.
    """
    _r_insert_clip(rating_db, clip_id="rc1", bearer_hash=_R_HASH_A,
                   validator_score=0.5)
    client.post(
        "/v1/clips/rc1/rating",
        json={"kind": "warn", "value": -1.0,
              "payload": {"validator_visible_at_rating": False}},
        headers=_r_auth(_R_KEY_A),
    )

    r_owner = client.get("/v1/clips/rc1/ratings", headers=_r_auth(_R_KEY_A))
    assert r_owner.status_code == 200
    rh_owner = r_owner.json()["ratings"][0]["rater_api_key_hash"]
    assert len(rh_owner) == 8
    assert rh_owner == _R_HASH_A[:8]

    r_admin = client.get("/v1/clips/rc1/ratings",
                         headers=_r_auth(_R_KEY_ADMIN))
    assert r_admin.status_code == 200
    assert r_admin.json()["ratings"][0]["rater_api_key_hash"] == _R_HASH_A

    r_other = client.get("/v1/clips/rc1/ratings", headers=_r_auth(_R_KEY_B))
    assert r_other.status_code == 403
    assert r_other.json()["error_code"] == "NOT_AUTHORIZED"


def test_rating_queue_returns_borderline_clips_ordered_by_distance(rating_db):
    """11. /v1/ratings/queue returns clips with composite ∈ [0.45, 0.65]
    not yet rated by caller, ordered by ABS(composite - 0.55) ASC.
    Out-of-band and NULL-composite clips are filtered. Rating a clip
    drops it from the queue.
    """
    _r_insert_clip(rating_db, clip_id="rb_far_low", bearer_hash=_R_HASH_A,
                   validator_score=0.46)
    _r_insert_clip(rating_db, clip_id="ra_center", bearer_hash=_R_HASH_A,
                   validator_score=0.55)
    _r_insert_clip(rating_db, clip_id="rc_far_high", bearer_hash=_R_HASH_A,
                   validator_score=0.63)
    _r_insert_clip(rating_db, clip_id="rd_high", bearer_hash=_R_HASH_A,
                   validator_score=0.90)
    _r_insert_clip(rating_db, clip_id="re_low", bearer_hash=_R_HASH_A,
                   validator_score=0.10)
    _r_insert_clip(rating_db, clip_id="rf_null", bearer_hash=_R_HASH_A,
                   validator_score=None)

    resp = client.get("/v1/ratings/queue?limit=20", headers=_r_auth(_R_KEY_A))
    assert resp.status_code == 200
    ids = [it["clip_id"] for it in resp.json()["items"]]
    # 0.55 → 0.0, 0.63 → 0.08, 0.46 → 0.09 (closest band-center first).
    assert ids == ["ra_center", "rc_far_high", "rb_far_low"]
    assert resp.json()["total_borderline"] == 3

    client.post(
        "/v1/clips/ra_center/rating",
        json={"kind": "warn", "value": -1.0,
              "payload": {"validator_visible_at_rating": False}},
        headers=_r_auth(_R_KEY_A),
    )
    resp2 = client.get("/v1/ratings/queue?limit=20", headers=_r_auth(_R_KEY_A))
    ids2 = [it["clip_id"] for it in resp2.json()["items"]]
    assert "ra_center" not in ids2
    assert resp2.json()["total_borderline"] == 2


def _r_insert_rating(
    h: HistoryStore,
    *,
    clip_id: str,
    rater_hash: str,
    created_at: float,
) -> None:
    """Helper: insert an active human_ratings row used for cohort-novelty
    bookkeeping in the diversity tests. ``warn`` kind keeps the test out
    of the pair-construction codepath.
    """
    h._conn.execute(
        """INSERT INTO human_ratings
           (clip_id, rater_api_key_hash, rating_kind, rating_value,
            rating_payload_json, validator_version_at_rating,
            validator_composite_at_rating, validator_visible_at_rating,
            created_at)
           VALUES (?, ?, 'warn', -1.0, '{}', ?, 0.55, 0, ?)""",
        (clip_id, rater_hash, "1.19.0-rc1", created_at),
    )
    h._conn.commit()


def test_rating_queue_diversity_quota_when_one_cohort_overrepresented(rating_db):
    """13a. Safety Gap-E (P0-2c). Rater has 50 active ratings against
    cohort X; the 20 unrated borderline clips are split 10 cohort-X /
    10 split across cohorts Y/Z/W. Queue must return ≥30% from
    cohorts other than X.
    """
    now = time.time()
    # 50 already-rated clips in cohort X — one rating row per clip is
    # enough; the query joins on the rated clips' shot_config_key.
    for i in range(50):
        cid = f"rated_{i}"
        _r_insert_clip(rating_db, clip_id=cid, bearer_hash=_R_HASH_A,
                       validator_score=0.55, shot_config_key="cohort-X")
        _r_insert_rating(rating_db, clip_id=cid, rater_hash=_R_HASH_A,
                         created_at=now - (50 - i))
    # 10 unrated borderline clips in cohort X.
    for i in range(10):
        _r_insert_clip(rating_db, clip_id=f"unrated_x_{i}",
                       bearer_hash=_R_HASH_A,
                       validator_score=0.55, shot_config_key="cohort-X")
    # 10 unrated borderline clips spread across novel cohorts Y/Z/W.
    novel_cohorts = ["cohort-Y", "cohort-Z", "cohort-W"]
    for i in range(10):
        _r_insert_clip(rating_db, clip_id=f"unrated_n_{i}",
                       bearer_hash=_R_HASH_A,
                       validator_score=0.55,
                       shot_config_key=novel_cohorts[i % 3])

    resp = client.get("/v1/ratings/queue?limit=20", headers=_r_auth(_R_KEY_A))
    assert resp.status_code == 200, resp.text
    items = resp.json()["items"]
    assert len(items) == 20
    novel = [it for it in items if it["shot_config_key"] != "cohort-X"]
    # ≥30% of returned clips have a novel cohort.
    import math
    assert len(novel) >= math.ceil(0.3 * len(items)), (
        f"diversity quota not met: {len(novel)}/{len(items)} novel"
    )


def test_rating_queue_fresh_rater_orders_by_distance(rating_db):
    """13b. Back-compat: a rater with no prior ratings sees pure
    distance ordering (every clip is novel-cohort by definition, the
    diversity quota is satisfied for free).
    """
    _r_insert_clip(rating_db, clip_id="rb_far_low", bearer_hash=_R_HASH_A,
                   validator_score=0.46, shot_config_key="cohort-A")
    _r_insert_clip(rating_db, clip_id="ra_center", bearer_hash=_R_HASH_A,
                   validator_score=0.55, shot_config_key="cohort-B")
    _r_insert_clip(rating_db, clip_id="rc_far_high", bearer_hash=_R_HASH_A,
                   validator_score=0.63, shot_config_key="cohort-C")

    resp = client.get("/v1/ratings/queue?limit=20", headers=_r_auth(_R_KEY_A))
    assert resp.status_code == 200
    ids = [it["clip_id"] for it in resp.json()["items"]]
    # Distance ordering: 0.55 → 0.0, 0.63 → 0.08, 0.46 → 0.09.
    assert ids == ["ra_center", "rc_far_high", "rb_far_low"]


def test_rating_queue_does_not_zero_out_when_corpus_is_cohort_thin(rating_db):
    """13c. Edge case: rater has rated all other cohorts; only cohort-X
    has unrated borderline clips left. Queue still serves them
    (degraded diversity, never zero results).
    """
    now = time.time()
    # Rater has rated representatives of cohorts Y and Z previously.
    for cohort, idx in [("cohort-Y", 0), ("cohort-Z", 1)]:
        cid = f"rated_{cohort}"
        _r_insert_clip(rating_db, clip_id=cid, bearer_hash=_R_HASH_A,
                       validator_score=0.55, shot_config_key=cohort)
        _r_insert_rating(rating_db, clip_id=cid, rater_hash=_R_HASH_A,
                         created_at=now - idx)
    # ALL unrated borderline clips happen to be cohort-X (also a cohort
    # the rater already touched).
    rated_x = "rated_cohort-X"
    _r_insert_clip(rating_db, clip_id=rated_x, bearer_hash=_R_HASH_A,
                   validator_score=0.55, shot_config_key="cohort-X")
    _r_insert_rating(rating_db, clip_id=rated_x, rater_hash=_R_HASH_A,
                     created_at=now)
    for i in range(5):
        _r_insert_clip(rating_db, clip_id=f"only_x_{i}",
                       bearer_hash=_R_HASH_A,
                       validator_score=0.55, shot_config_key="cohort-X")

    resp = client.get("/v1/ratings/queue?limit=20", headers=_r_auth(_R_KEY_A))
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert len(items) == 5  # all five unrated cohort-X clips returned
    assert all(it["shot_config_key"] == "cohort-X" for it in items)


def test_rating_queue_diversity_env_var_configurable(rating_db, monkeypatch):
    """13d. ``RATING_QUEUE_MIN_COHORT_DIVERSITY=0.5`` raises the floor
    so ≥50% of returned clips have a novel cohort.
    """
    monkeypatch.setattr(server_mod, "_RATING_QUEUE_MIN_COHORT_DIVERSITY", 0.5)
    now = time.time()
    # 10 ratings in cohort-X to mark it as "recently rated".
    for i in range(10):
        cid = f"rated_{i}"
        _r_insert_clip(rating_db, clip_id=cid, bearer_hash=_R_HASH_A,
                       validator_score=0.55, shot_config_key="cohort-X")
        _r_insert_rating(rating_db, clip_id=cid, rater_hash=_R_HASH_A,
                         created_at=now - (10 - i))
    # 8 unrated cohort-X clips and 8 unrated novel-cohort clips.
    for i in range(8):
        _r_insert_clip(rating_db, clip_id=f"unrated_x_{i}",
                       bearer_hash=_R_HASH_A,
                       validator_score=0.55, shot_config_key="cohort-X")
    novel_cohorts = ["cohort-Y", "cohort-Z"]
    for i in range(8):
        _r_insert_clip(rating_db, clip_id=f"unrated_n_{i}",
                       bearer_hash=_R_HASH_A,
                       validator_score=0.55,
                       shot_config_key=novel_cohorts[i % 2])

    resp = client.get("/v1/ratings/queue?limit=10", headers=_r_auth(_R_KEY_A))
    assert resp.status_code == 200
    items = resp.json()["items"]
    novel = [it for it in items if it["shot_config_key"] != "cohort-X"]
    import math
    assert len(novel) >= math.ceil(0.5 * len(items)), (
        f"50% quota not met: {len(novel)}/{len(items)} novel"
    )


def test_rating_rate_limit_extends_to_clips_paths(rating_db, monkeypatch):
    """12. /v1/clips/{id}/rating + /v1/ratings/queue are covered by the
    middleware's _RATE_LIMITED_PATH_PREFIXES; bursting beyond capacity
    returns 429.

    # safety-critical: invariant-4 (rate-limit anti-fatigue),
    #                  attack-pattern (rate-limit → 429)
    """
    _r_insert_clip(rating_db, clip_id="rc1", bearer_hash=_R_HASH_A,
                   validator_score=0.5)
    monkeypatch.setattr(server_mod, "_EMBEDDINGS_RATE_LIMIT_BURST", 2)
    monkeypatch.setattr(server_mod, "_EMBEDDINGS_RATE_LIMIT_PER_SEC", 1)
    server_mod._embeddings_buckets.clear()

    assert "/v1/clips/" in server_mod._RATE_LIMITED_PATH_PREFIXES
    assert "/v1/ratings/queue" in server_mod._RATE_LIMITED_PATH_PREFIXES

    body = {"kind": "warn", "value": -1.0,
            "payload": {"validator_visible_at_rating": False}}
    statuses = []
    for _ in range(5):
        r = client.post(
            "/v1/clips/rc1/rating", json=body, headers=_r_auth(_R_KEY_A),
        )
        statuses.append(r.status_code)
    assert 429 in statuses, f"rate-limit never tripped: {statuses}"


# ---------------------------------------------------------------------------
# Phase 1 safety gap-fill tests — explicit attestation of safety claims
# whose enforcement was implicit in the surface above.
# ---------------------------------------------------------------------------


def test_signal_strength_ordering_invariant():
    """invariant-8: 0.7 (validator_pass) < 0.75 (human_rating) < 0.9
    (user_retake). Human-in-the-loop signal must be stronger than the
    validator-derived signals but weaker than an explicit user retake.
    The construction script's source weights are the source of truth.

    # safety-critical: invariant-8 (signal_strength bounds preserved)
    """
    import importlib.util
    import pathlib
    spec = importlib.util.spec_from_file_location(
        "construct_pp",
        pathlib.Path(__file__).resolve().parent.parent
        / "scripts" / "construct_preference_pairs.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    weights = mod.SIGNAL_STRENGTH
    assert weights["validator_pass"] == 0.7
    assert weights["validator_fail"] == 0.3
    assert weights["composition_kept"] == 0.5
    assert weights["user_retake"] == 0.9
    rating_strength = server_mod._RATING_SIGNAL_STRENGTH
    assert rating_strength == 0.75
    assert weights["validator_pass"] < rating_strength < weights["user_retake"], (
        "human_rating signal must satisfy 0.7 < 0.75 < 0.9"
    )


def test_rater_api_key_hash_not_null_constraint(fresh_history):
    """P0-4: ``human_ratings.rater_api_key_hash`` is NOT NULL at the
    schema level. A NULL insert raises IntegrityError — the column is
    load-bearing for the right-to-delete cascade and audit chain.

    # safety-critical: P0-4 (rater_hash NOT NULL — backfill-impossible without it)
    """
    import sqlite3
    with pytest.raises(sqlite3.IntegrityError):
        fresh_history._conn.execute(
            """INSERT INTO human_ratings
               (clip_id, rater_api_key_hash, rating_kind, rating_value,
                validator_visible_at_rating, created_at)
               VALUES (?, NULL, ?, ?, ?, ?)""",
            ("c_x", "warn", -1.0, 0, time.time()),
        )
        fresh_history._conn.commit()
    fresh_history._conn.rollback()


def test_rating_post_opted_out_subject_blocks_cross_bearer(rating_db):
    """invariant-1 + privacy: cross-bearer rating blocked when subject
    bearer is opted-out of training, even with consent flag set —
    training_opt_in=0 always wins over consent.

    # safety-critical: invariant-1 (bearer opt-in re-checked),
    #                  attack-pattern (opted-out cross-bearer → 409 PRIVACY_GATE_BLOCKED)
    """
    _r_insert_clip(rating_db, clip_id="rb1", bearer_hash=_R_HASH_B,
                   validator_score=0.5)
    _r_insert_clip(rating_db, clip_id="rb2", bearer_hash=_R_HASH_B,
                   validator_score=0.5)
    # Bearer B has consent set BUT is opted-out of training.
    rating_db._conn.execute(
        "UPDATE api_key_metadata SET cross_bearer_rating_consent_at = ?, "
        "training_opt_in = 0 WHERE api_key_hash = ?",
        (time.time(), _R_HASH_B),
    )
    rating_db._conn.commit()
    resp = client.post(
        "/v1/clips/rb1/rating",
        json={"kind": "pair_chose_a", "value": 1.0,
              "payload": _r_pair_payload("rb2")},
        headers=_r_auth(_R_KEY_A),
    )
    assert resp.status_code == 409
    assert resp.json()["error_code"] == "PRIVACY_GATE_BLOCKED"


def test_rating_post_persists_bearer_hash_and_lr_seed_audit(rating_db):
    """P0-2 + P0-4: rater_api_key_hash captured on every rating, and
    L/R seed audit-logged into ``rating_payload_json`` so post-hoc
    transitivity / order-bias analysis can recover decision context.

    # safety-critical: P0-2 (L/R audit logged for order-bias analysis),
    #                  P0-4 (rater_hash + bearer_hash captured per row)
    """
    import json as _json
    _r_insert_clip(rating_db, clip_id="rc1", bearer_hash=_R_HASH_A,
                   validator_score=0.5)
    _r_insert_clip(rating_db, clip_id="rc2", bearer_hash=_R_HASH_A,
                   validator_score=0.5)
    resp = client.post(
        "/v1/clips/rc1/rating",
        json={"kind": "pair_chose_a", "value": 1.0,
              "payload": _r_pair_payload("rc2", lr_seed=99887)},
        headers=_r_auth(_R_KEY_A),
    )
    assert resp.status_code == 200, resp.text
    rid = resp.json()["rating_id"]
    row = rating_db._conn.execute(
        "SELECT rater_api_key_hash, rating_payload_json FROM human_ratings "
        "WHERE rating_id = ?", (rid,),
    ).fetchone()
    assert row["rater_api_key_hash"] == _R_HASH_A
    assert row["rater_api_key_hash"] is not None
    assert len(row["rater_api_key_hash"]) == 64  # sha256 hex
    pl = _json.loads(row["rating_payload_json"])
    assert pl["lr_seed"] == 99887
    # Subject bearer recoverable via JOIN against generations.api_key_hash
    # — confirms the lineage backbone for post-hoc audit / right-to-delete.
    sub = rating_db._conn.execute(
        "SELECT g.api_key_hash AS bearer_hash FROM generations g "
        "WHERE g.id = (SELECT clip_id FROM human_ratings WHERE rating_id = ?)",
        (rid,),
    ).fetchone()
    assert sub["bearer_hash"] == _R_HASH_A


def test_threshold_endpoint_admin_only_for_other_bearers(rating_db):
    """invariant-14 + privacy: GET /v1/api-keys/me/validator-thresholds
    only ever reflects the caller's own thresholds. There is no
    /v1/api-keys/{other}/validator-thresholds in the v1 surface — bearer
    A cannot read or write bearer B's thresholds. Verifies separation
    via two distinct bearers writing different overrides.

    # safety-critical: invariant-14 (threshold sanity — per-bearer scoping),
    #                  privacy (bearer cannot read/write another's thresholds)
    """
    # Both opted in via rating_db fixture; explicit overrides here.
    client.post(
        "/v1/api-keys/me/validator-thresholds",
        json={"pass": 0.70, "retake": 0.40},
        headers=_r_auth(_R_KEY_A),
    )
    client.post(
        "/v1/api-keys/me/validator-thresholds",
        json={"pass": 0.60, "retake": 0.50},
        headers=_r_auth(_R_KEY_B),
    )
    a_get = client.get(
        "/v1/api-keys/me/validator-thresholds", headers=_r_auth(_R_KEY_A),
    ).json()
    b_get = client.get(
        "/v1/api-keys/me/validator-thresholds", headers=_r_auth(_R_KEY_B),
    ).json()
    assert a_get["pass"] == pytest.approx(0.70)
    assert a_get["retake"] == pytest.approx(0.40)
    assert b_get["pass"] == pytest.approx(0.60)
    assert b_get["retake"] == pytest.approx(0.50)


def test_consumed_pair_immutable_attack_pattern(rating_db):
    """invariant-5 + plan §"What this plan deliberately does NOT do":
    once a preference_pair is in a training run (used_in_training_run_id
    NOT NULL), retraction CANNOT unwind it — the trained-against
    artifact is immutable. The rating row is soft-deleted; the pair
    survives with its training_run_id intact.

    # safety-critical: invariant-5 (audit chain immutability for consumed pairs),
    #                  L2.5-artifact-tampering (retraction cannot unwind shipped LoRAs)
    """
    _r_insert_clip(rating_db, clip_id="rc1", bearer_hash=_R_HASH_A,
                   validator_score=0.5)
    _r_insert_clip(rating_db, clip_id="rc2", bearer_hash=_R_HASH_A,
                   validator_score=0.5)
    r1 = client.post(
        "/v1/clips/rc1/rating",
        json={"kind": "pair_chose_a", "value": 1.0,
              "payload": _r_pair_payload("rc2")},
        headers=_r_auth(_R_KEY_A),
    )
    rid = r1.json()["rating_id"]
    pid = r1.json()["pair_id"]
    rating_db._conn.execute(
        "UPDATE preference_pairs SET used_in_training_run_id = 'shipped_run' "
        "WHERE pair_id = ?", (pid,),
    )
    rating_db._conn.commit()

    drop = client.delete(
        f"/v1/clips/rc1/rating/{rid}", headers=_r_auth(_R_KEY_A),
    )
    assert drop.status_code == 200
    assert drop.json()["preference_pair_deleted"] is False
    # Pair survives with training_run_id intact.
    pp = rating_db._conn.execute(
        "SELECT used_in_training_run_id FROM preference_pairs WHERE pair_id = ?",
        (pid,),
    ).fetchone()
    assert pp is not None
    assert pp["used_in_training_run_id"] == "shipped_run"


# ---------------------------------------------------------------------------
# Phase 1 polish — PATCH privacy gate + per-rater anti-fatigue caps.
# ---------------------------------------------------------------------------


def test_rating_patch_rechecks_privacy_gate(rating_db):
    """PATCH must re-run ``_privacy_gate_allows`` against the clip — if
    the clip's bearer flips ``training_opt_in`` to 0 between POST and
    PATCH, the patch is rejected with 409 PRIVACY_GATE_BLOCKED.

    # safety-critical: invariant-1 (bearer opt-in re-checked on every write),
    #                  attack-pattern (post → flip-opt-out → patch → must 409)
    """
    _r_insert_clip(rating_db, clip_id="rb1", bearer_hash=_R_HASH_B,
                   validator_score=0.5)
    _r_insert_clip(rating_db, clip_id="rb2", bearer_hash=_R_HASH_B,
                   validator_score=0.5)
    rating_db._conn.execute(
        "UPDATE api_key_metadata SET cross_bearer_rating_consent_at = ? "
        "WHERE api_key_hash = ?",
        (time.time(), _R_HASH_B),
    )
    rating_db._conn.commit()

    r1 = client.post(
        "/v1/clips/rb1/rating",
        json={"kind": "pair_chose_a", "value": 1.0,
              "payload": _r_pair_payload("rb2")},
        headers=_r_auth(_R_KEY_A),
    )
    assert r1.status_code == 200, r1.text
    rid = r1.json()["rating_id"]

    rating_db._conn.execute(
        "UPDATE api_key_metadata SET training_opt_in = 0 "
        "WHERE api_key_hash = ?",
        (_R_HASH_B,),
    )
    rating_db._conn.commit()

    patch = client.patch(
        f"/v1/clips/rb1/rating/{rid}",
        json={"value": 0.5,
              "payload": {"validator_visible_at_rating": False,
                          "pair_partner_clip_id": "rb2",
                          "lr_seed": 9}},
        headers=_r_auth(_R_KEY_A),
    )
    assert patch.status_code == 409, patch.text
    assert patch.json()["error_code"] == "PRIVACY_GATE_BLOCKED"


def test_rating_anti_fatigue_two_second_gate(rating_db, monkeypatch):
    """Plan invariant #4: the second POST within 2s of the first
    returns 409 RATE_LIMIT_TOO_FAST, regardless of clip / kind.

    # safety-critical: invariant-4 (anti-fatigue 2s gate)
    """
    monkeypatch.setattr(server_mod, "_RATING_MIN_INTERVAL_S", 2.0)
    _r_insert_clip(rating_db, clip_id="rc1", bearer_hash=_R_HASH_A,
                   validator_score=0.5)
    _r_insert_clip(rating_db, clip_id="rc2", bearer_hash=_R_HASH_A,
                   validator_score=0.5)
    body = {"kind": "warn", "value": -1.0,
            "payload": {"validator_visible_at_rating": False}}

    r1 = client.post("/v1/clips/rc1/rating", json=body,
                     headers=_r_auth(_R_KEY_A))
    assert r1.status_code == 200, r1.text
    r2 = client.post("/v1/clips/rc2/rating", json=body,
                     headers=_r_auth(_R_KEY_A))
    assert r2.status_code == 409
    assert r2.json()["error_code"] == "RATE_LIMIT_TOO_FAST"


def test_rating_session_cap_exceeded(rating_db, monkeypatch):
    """Plan invariant #4: hitting ``_RATING_SESSION_CAP`` ratings inside
    the 90-minute rolling window returns 409 SESSION_CAP_EXCEEDED.

    Uses a low cap (3) and disables the 2s gate so we can verify the
    cap independently of the rate-limit gate.

    # safety-critical: invariant-4 (anti-fatigue session cap)
    """
    monkeypatch.setattr(server_mod, "_RATING_MIN_INTERVAL_S", 0.0)
    monkeypatch.setattr(server_mod, "_RATING_SESSION_CAP", 3)
    _r_insert_clip(rating_db, clip_id="rc1", bearer_hash=_R_HASH_A,
                   validator_score=0.5)
    body = {"kind": "warn", "value": -1.0,
            "payload": {"validator_visible_at_rating": False}}
    statuses = []
    for _ in range(4):
        r = client.post("/v1/clips/rc1/rating", json=body,
                        headers=_r_auth(_R_KEY_A))
        statuses.append((r.status_code, r.json().get("error_code")))
    assert statuses[0][0] == 200
    # Subsequent supersedes are 200 until the cap kicks in (the cap counts
    # all human_ratings rows, including superseded). With cap=3, the 4th
    # write is rejected.
    assert statuses[-1][0] == 409
    assert statuses[-1][1] == "SESSION_CAP_EXCEEDED"
