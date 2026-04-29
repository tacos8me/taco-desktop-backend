# noodlefinger-mcp — LLM-driven workflows over the taco-backend API

**v0.6.0** · catalog: [github.com/tacos8me/noodle-portal](https://github.com/tacos8me/noodle-portal) (`mcp/` subdir) · live-validated end-to-end (one MCP `cut_music_video` call → real H.264+AAC MP4).

A single Python MCP server that wraps `api.noodlefinger.io` for LLM clients (Claude Code, Cursor, Continue, Codex CLI, OpenCode). Two tiers:

- **Tier-0** (anonymous, 6 tools + 3 resources): docs lookup. Always available.
- **Tier-1** (authenticated, 23 tools): submit jobs, upload files, run the `cut_music_video` macro, browse + apply LTX/Flux LoRAs, beat-grid analysis, shot-list authoring helpers. Registers iff `NOODLEFINGER_API_KEY` is set.

Notable shipped behavior since v0.2:

- **v0.4.4** — DAG-aware parallel clip dispatch in `cut_music_video` (auto-scales to live worker count via `GET /v1/system/workers`).
- **v0.4.6** — chain-mode default flipped from `seamless-segment` (hard-pin, motion-suppressing) to `keyframes` (soft-guide, motion-friendly). The MV motion fix.
- **v0.5.0** — opt-in beat-aligned audio slicing for `cut_music_video` via `slice_strategy="beats"|"downbeats"` + `beat_analyzer="auto"|"madmom"|"librosa"`. Wraps `POST /v1/music/analyze`.
- **v0.6.0** — LoRA browse + apply: `list_loras` / `get_lora` / `search_loras` / `rescan_flux_loras` tier-1 tools, plus top-level + per-shot `lora: {id, strength}` field on `cut_music_video`. Both LTX and Flux registries.

See also: [docs/API.md](API.md) (canonical HTTP contract) · [docs/QUICKSTART.md](QUICKSTART.md) · [noodle-portal repo](https://github.com/tacos8me/noodle-portal).

---

## Quick start (Claude Code)

One-liner with tier-1 enabled:

```bash
claude mcp add --scope user \
  -e NOODLEFINGER_API_KEY=nf_live_sk_... \
  noodlefinger uvx -- \
    --from "git+https://github.com/tacos8me/noodle-portal#subdirectory=mcp" \
    noodlefinger-mcp
```

Restart Claude Code, then:

1. **Verify** — `/mcp` shows `noodlefinger ✓ Connected`.
2. **Smoke tier-0** — ask: *"List the noodlefinger endpoint groups."* → 13 groups via `mcp__noodlefinger__list_groups`.
3. **Smoke tier-1** — ask: *"Cut me a 6-second draft music video — vinyl spinning on a turntable, lo-fi beat."* → one `cut_music_video` call, ~80 s, returns a `final_storage_uri` you can `download_job_result` to disk.

That's it. Drop the `-e NOODLEFINGER_API_KEY=...` line for the docs-only tier-0 install (no key required, nothing leaves your machine).

For non-Claude-Code clients, advanced env vars, or pinning to a specific commit SHA, see [§2 Connection](#2-connection). For security model, see [§3 Security](#3-security).

---

## Table of contents

1. [What it is](#1-what-it-is)
2. [Connection](#2-connection)
3. [Security](#3-security)
4. [Usage](#4-usage)
5. [Architecture](#5-architecture)
6. [Tools reference](#6-tools-reference)
7. [Roadmap](#7-roadmap)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. What it is

A single `noodlefinger-mcp` binary, distributed as a Python wheel via `uvx --from git+...`, speaking the [Model Context Protocol](https://modelcontextprotocol.io) over stdio. It bundles:

- **Endpoint catalog** — 77+ endpoints across 13 groups. Same data that powers `portal.noodlefinger.io/docs`.
- **Flow walkthroughs** — 17+ step-by-step recipes (`text-to-video`, `audio-to-video`, `video-hdr`, `cut-music-video`, `lora-browse-apply`, …).
- **CHANGELOG** — for cross-referencing endpoint `since_version`.

| Surface | Tier-0 (anonymous) | Tier-1 (`NOODLEFINGER_API_KEY` required) |
|---|---|---|
| **Tools (29)** | 6: `search_endpoints`, `get_endpoint`, `list_groups`, `list_flows`, `get_flow`, `get_changelog` | 23: `submit_job`, `get_job`, `wait_for_job`, `cancel_job`, `download_job_result`, `download_storage_uri`, `upload_file`, `extract_segment`, `create_composition`, `export_composition`, `get_changelog_for_endpoint`, `cut_music_video`, `resume_music_video`, `list_sessions`, `get_beat_grid`, `plan_shot_list`, `match_cut_prompt_pair`, `weave_inserts`, `apply_section_palette`, `list_loras`, `get_lora`, `search_loras`, `rescan_flux_loras` |
| **Resources (3)** | `noodlefinger://endpoints`, `noodlefinger://flows`, `noodlefinger://changelog` | — |

When tier-1 is enabled, the server submits jobs against `api.noodlefinger.io`, polls them to completion, downloads results, and runs the `cut_music_video` macro: music → N a2v clips with seamless chain conditioning → composition → export, all in one tool call with session-resume on failure.

---

## 2. Connection

The [Quick start](#quick-start-claude-code) above covers the common path. The rest of this section covers advanced installs.

### Pinning a specific commit (production)

```bash
claude mcp add --scope user noodlefinger uvx -- \
  --from "git+https://github.com/tacos8me/noodle-portal@<sha>#subdirectory=mcp" \
  noodlefinger-mcp
```

Pinning a SHA defends against accidental or malicious upstream changes between installs. See [§3 Security § Supply chain](#supply-chain-pin-a-commit-sha).

### Direct `settings.json` (Cursor, Continue, Codex CLI, OpenCode)

```jsonc
{
  "mcpServers": {
    "noodlefinger": {
      "command": "uvx",
      "args": [
        "--from", "git+https://github.com/tacos8me/noodle-portal#subdirectory=mcp",
        "noodlefinger-mcp"
      ],
      "env": {
        "NOODLEFINGER_API_KEY": "nf_live_sk_...",
        "NOODLEFINGER_API_BASE": "https://api.noodlefinger.io"
      }
    }
  }
}
```

- **Cursor**: `Cmd-Shift-P → Cursor: Open MCP Settings`.
- **Continue / Codex CLI / OpenCode**: their docs cover the same `mcpServers` shape.

### Environment variables

| Var | Default | Purpose |
|---|---|---|
| `NOODLEFINGER_API_KEY` | unset | Bearer token for tier-1 calls. **When unset, tier-1 tools are not registered** — anonymous installs see only the 6 docs tools. |
| `NOODLEFINGER_API_BASE` | `https://api.noodlefinger.io` | Override for self-hosting / local dev (e.g. `http://localhost:8090`). Trailing slash is stripped. |
| `NOODLEFINGER_SESSION_DIR` | `~/.cache/noodlefinger-mcp/sessions/` | Where `cut_music_video` checkpoints live. Created mode `0o700`; files mode `0o600`. |
| `NOODLEFINGER_MCP_CATALOG` | bundled `_data/endpoints.json` | Override the catalog path (e.g. point at a private fork). |
| `NOODLEFINGER_MCP_FLOWS` | bundled `_data/flows.json` | Override the flows path. |
| `NOODLEFINGER_MCP_CHANGELOG` | `/mnt/nvme-1/servers/taco-backend/CHANGELOG.md` | Override the changelog path. Default points at the dev box; on fresh installs either point it at a checked-out copy or expect `get_changelog` to soft-fail. |

---

## 3. Security

Distilled from the [v0.2 security audit](https://github.com/tacos8me/noodle-portal/blob/main/mcp/_phase2/04-security-audit.md): 12 findings (0 outstanding, 3 build-time blockers all fixed before merge, 3 recommended, 5 accepted-as-documented).

### Trust boundary

Three layers, two boundaries:

```
┌─────────────────┐  stdio JSON-RPC   ┌─────────────────────┐  HTTPS+Bearer  ┌────────────────────┐
│   LLM client    │ ────────────────► │  noodlefinger-mcp   │ ─────────────► │ api.noodlefinger.io│
│ (semi-trusted)  │                   │  (your local proc)  │                │ (outer boundary)   │
│ Claude Code,    │                   │  holds API key      │                │ per-key quotas,    │
│ Cursor, etc.    │                   │  catalog allow-list │                │ admin-bearer guard │
└─────────────────┘                   └─────────────────────┘                └────────────────────┘
```

The LLM client is **semi-trusted** — it can submit arbitrary tool arguments, but it is constrained by the catalog-driven endpoint allow-list and the input schemas. The noodlefinger.io API is the **outer trust boundary** — it enforces per-key quotas and admin-bearer guards on system endpoints.

**Worst-case for a compromised IDE plugin**: it can use your bearer to consume per-key generation quota and produce content under your account. It cannot toggle turbo, unload models, or scale the remote pool through the MCP. **If you suspect compromise, revoke the key in the portal admin UI** — all in-flight orchestrator sessions die on the next HTTP call.

**Recommended hygiene**: rotate keys periodically; consider a secondary low-quota key dedicated to MCP use so a leak doesn't compromise your primary key's quota.

### API key handling

`NOODLEFINGER_API_KEY` is read once at startup into a module-level constant, used to build `Authorization: Bearer <key>` headers, and **never** written to disk, stdout, or session checkpoints.

Structured stderr logs include only `{tool, method, path, status, elapsed_s}` — no headers, no bodies, no prompt text. The logger is a closed-form helper (`_log_call(tool, method, path, status, elapsed_s, *, error=None)`) with named parameters only — no `**kwargs`, so a future maintainer can't accidentally add `headers=...` or `body=...` and silently leak secrets. `tests/test_no_secret_leak.py` exercises every tier-1 tool and asserts the test API key, the literal "Authorization", and prompt text never appear in captured stderr.

### Supply chain — pin a commit SHA

`uvx --from "git+...#subdirectory=mcp"` resolves the default branch HEAD at install time. Without an `@<ref>` qualifier, every fresh `uvx` invocation could pull a different commit if upstream moves. **A compromised upstream repo** — push access leaked, account compromised, malicious PR merged — could ship an MCP server that exfiltrates your API key on next launch. Blast radius: every developer who has the MCP registered.

```bash
# Production: pin to a tagged release commit SHA
claude mcp add --scope user noodlefinger uvx -- \
  --from "git+https://github.com/tacos8me/noodle-portal@<sha>#subdirectory=mcp" \
  noodlefinger-mcp

# Development only: auto-tracks main, accepts upstream changes silently
claude mcp add --scope user noodlefinger uvx -- \
  --from "git+https://github.com/tacos8me/noodle-portal#subdirectory=mcp" \
  noodlefinger-mcp
```

v0.3 ships PyPI distribution under `noodlefinger-mcp`, after which signed-by-PyPI is available. The git-SHA pattern remains the v0.2 best practice.

### Session checkpoint file permissions

Session files at `$NOODLEFINGER_SESSION_DIR/<session_id>.json` (default `~/.cache/noodlefinger-mcp/sessions/`) contain `prompt`, `music_lyrics`, `subject_anchor_uri`, `final_uri`, and intermediate `storage://` URIs. The server enforces:

- `os.makedirs(session_dir, mode=0o700, exist_ok=True)` on init (idempotent repair if the dir exists with looser perms from a prior version).
- `os.chmod(file, 0o600)` immediately after every atomic-write — umask interacts with `O_CREAT` mode bits, so setting mode at `open()` time is insufficient.
- `fcntl.flock(LOCK_EX)` held continuously across the `os.replace()` rename, preventing concurrent-resume races.

**Treat `storage://` URIs as capabilities** — even with `0o600` on the checkpoint, the bytes referenced are still fetchable by anyone with both the URI and a valid bearer for your key. Windows portability for the chmod calls is out of scope for v0.2 (POSIX only).

### `submit_job` allow-list

Tier-1 `submit_job(endpoint_id, params)` is a generic POST passthrough, so it must defend against the LLM trying to call `/v1/system/turbo` or `/v1/ltx/unload`. Two assertions before any HTTP call:

```python
if ep.get("method") != "POST":
    return _err(f"endpoint {endpoint_id} is not a POST endpoint")
if ep.get("response", {}).get("schema") != "SubmissionEnvelope":
    return _err(f"endpoint {endpoint_id} does not return a job (not a submission endpoint)")
```

System-control endpoints return `{"ok": true}` or telemetry dicts — not `SubmissionEnvelope` — so they're rejected by the schema gate. The URL is built from the catalog `path` (data) + `_base_url()` (env var), with no string-interpolation of user input and no shell-out anywhere. **Defense in depth**: the backend's admin-bearer guard rejects those endpoints with 403 even if the gate were bypassed (user keys aren't in `.admin_keys`).

`tests/test_tools_tier1.py::test_submit_job_rejects_admin_endpoint` exercises this guard.

### `upload_file` magic-byte bypass

`upload_file` defaults `Content-Type: application/octet-stream`. The taco-backend treats octet-stream as opt-out from magic-byte verification — by design, to support exotic media that legitimately fails sniffing.

A malicious caller could upload non-media bytes (executable, archive, exfiltrated text) and reference the resulting `storage://` URI in a generation request. The backend's PyAV decode fails safely on non-media bytes; there is no path where octet-stream uploads are executed or trusted as having a particular structure. The bytes do consume your per-key `PER_KEY_UPLOAD_BYTES_PER_DAY` quota — that's the abuse cap.

If you have the correct MIME type, pass it explicitly via the `mime` argument and the backend will validate the bytes match.

### Retry budget — per-call, not orchestrator-wide

Three attempts per HTTP call, backoff `[1s, 3s]` jittered, retryable on `{429, 502, 503, 504, ConnectError, ReadTimeout}`, with `Retry-After` honored on 429. **Non-retryable 4xx codes (400, 401, 403, 404, 409, 413, 422) fail immediately.**

The orchestrator's per-step retry budgets are scoped per-call, not shared across the orchestrator — a flaky backend triggering 3 retries on every HTTP call cannot snowball a 12-call pipeline into 12 × N exponential explosion. Worst case for `cut_music_video(num_clips=5)`: `1 (music) + 5 (a2v) + 4 (segment) + 1 (composition) + 1 (export) = 12 calls × 3 = 36 max HTTP requests`. The 1800 s default `timeout_s` on `cut_music_video` is the global ceiling.

### Local cache hygiene — manual cleanup for v0.2

`download_job_result` writes to `~/.cache/noodlefinger-mcp/results/<job_id>.<ext>`; `download_storage_uri` writes to `~/.cache/noodlefinger-mcp/uploads/<uuid>.<ext>`. Files are written `0o600` (matching session-file hygiene; they may contain proprietary generated content), but neither has eviction in v0.2. Over months of daily `cut_music_video` use a developer can accumulate tens of GB.

**v0.2 recommendation**: `rm -rf ~/.cache/noodlefinger-mcp/results/` periodically. v0.3 will add a `--max-cache-size` env var with LRU eviction.

### What the MCP does NOT do

- **No system-control endpoints.** Tier-1 deliberately excludes `POST /v1/system/{turbo,pause,resume}`, `POST /v1/{flux,ltx}/{unload,reload}`, and `POST /v1/system/pool/remote-workers`. An LLM cannot take the backend offline, toggle turbo, scale the remote pool, or reload models. Operational controls stay on the dashboard.
- **No cookies, no session middleware, no HTTP server.** MCP is stdio-only — the process speaks JSON-RPC over stdin/stdout to one parent (the LLM client), not to the network.
- **No telemetry.** No usage counters, no error reporting back to a server, no analytics. The server only ever talks to `NOODLEFINGER_API_BASE`.
- **No PyPI publish (yet).** v0.2 is git-distribution only. v0.3 ships PyPI for signed-by-PyPI distribution.

---

## 4. Usage

Each transcript below is a real (lightly abbreviated) MCP session. JSON outputs are trimmed for readability — actual responses include all fields documented in [§6 Tools reference](#6-tools-reference).

### 4.1 Tier-0: discovering an endpoint without an API key

LLM client booted without `NOODLEFINGER_API_KEY`. Only the 6 docs tools visible. Goal: find the right endpoint to drive video from a song.

```text
USER: How do I drive video from an audio file with this API?

ASSISTANT (tool: mcp__noodlefinger__list_groups)
  → 13 groups, including "video-generation"

ASSISTANT (tool: mcp__noodlefinger__list_flows)
  → 16 flows: upload-flow, text-to-video, image-to-video, audio-to-video, …

ASSISTANT (tool: mcp__noodlefinger__get_flow, args: {flow_id: "audio-to-video"})
  → 4-step walkthrough, references endpoints v1-upload-slot, v2-audio-to-video,
    v2-extract-segment, v2-jobs-stream

ASSISTANT (tool: mcp__noodlefinger__get_endpoint, args: {endpoint_id: "v2-audio-to-video"})
  → full request_body schema:
    { audio_uri: string (required), prompt: string (required),
      model: "ltx-2-3-fast" | "ltx-2-3-pro" | "ltx-2-3-hq",
      resolution: enum, duration: number, fps: number,
      image_uri | keyframes | segment_uri (3-way mutex), … }

ASSISTANT: Use POST /v2/audio-to-video. Here's the curl:
  curl -X POST https://api.noodlefinger.io/v2/audio-to-video \
       -H "Authorization: Bearer $KEY" \
       -d '{"audio_uri": "...", "prompt": "...", "model": "ltx-2-3-fast", ...}'
```

This tier is fully usable as a documentation surface — no key required, nothing leaves your machine.

### 4.2 Tier-1: single-job submit + wait + download

Goal: take a local `song.mp3`, generate a 6-second clip driven by it, save the MP4 locally.

```text
USER: Make a 6s music video from my song.mp3 with prompt "neon-lit dancer".

ASSISTANT (tool: mcp__noodlefinger__upload_file,
           args: {local_path: "/home/me/song.mp3", mime: "audio/mpeg"})
  → {"storage_uri": "storage://abcdef0123…", "size_bytes": 4128000}

ASSISTANT (tool: mcp__noodlefinger__submit_job,
           args: {endpoint_id: "v2-audio-to-video",
                  params: {audio_uri: "storage://abcdef…",
                           prompt: "neon-lit dancer in slow motion",
                           model: "ltx-2-3-fast",
                           resolution: "1080x1920",
                           duration: 6.0, fps: 24}})
  → {"job_id": "job_xyz789", "status": "queued"}

ASSISTANT (tool: mcp__noodlefinger__wait_for_job,
           args: {job_id: "job_xyz789", timeout_s: 600})
  → polls every 2 s; after ~45 s returns:
     {"status": "completed", "result_storage_uri": "storage://fedcba…"}

ASSISTANT (tool: mcp__noodlefinger__download_job_result,
           args: {job_id: "job_xyz789", out_path: "/home/me/clip.mp4"})
  → {"path": "/home/me/clip.mp4", "size_bytes": 12_400_000}

ASSISTANT: Saved to /home/me/clip.mp4 (12.4 MB).
```

Three tool calls (upload, submit + wait, download). `wait_for_job` polls internally — the LLM doesn't see the per-poll requests.

### 4.3 Tier-1 macro: `cut_music_video` end-to-end

Goal: a complete 30-second music video with 5 chained clips, exported as a single MP4. **One tool call.**

```text
USER: Cut me a 30-second music video — five clips of a masked drummer in a smoky
      room, dark cinematic percussion soundtrack. Preview quality is fine.

ASSISTANT (tool: mcp__noodlefinger__cut_music_video,
           args: {prompt: "A masked drummer plays in a smoky red-lit room",
                  music_prompt: "dark cinematic percussion, slow build",
                  duration_s: 30, num_clips: 5, quality: "preview"})

  [internal pipeline — all hidden from the LLM, all checkpointed:]
  ├─ POST /v2/music                        → audio storage URI (~30 s)
  ├─ for clip in 0..4:
  │    POST /v2/audio-to-video             (~50 s wait)
  │    [if clip > 0: segment_uri from prior tail]
  │    POST /v2/video/extract-segment      (~2 s, only if clip < 4)
  │    [historyId = job_id — no /v2/history lookup needed]
  │    [checkpoint to ~/.cache/noodlefinger-mcp/sessions/<id>.json]
  ├─ POST /v2/compositions                 (timeline blob)
  └─ POST /v2/compositions/{id}/export     → final MP4 storage URI

  → {"session_id": "mv_1745712000_deadbeef",
     "status": "completed",
     "music_job_id": "job_music123",
     "clip_jobs": [
       {"clip_idx": 0, "job_id": "job_c0", "video_uri": "storage://v0", "history_id": "gen_0"},
       …5 total…
     ],
     "composition_id": "comp_abc",
     "export_job_id": "job_exp456",
     "final_storage_uri": "storage://final_uri",
     "elapsed_s": 412}

ASSISTANT (tool: mcp__noodlefinger__download_job_result,
           args: {job_id: "job_exp456", out_path: "/home/me/mv.mp4"})
  → {"path": "/home/me/mv.mp4", "size_bytes": 48_200_000}

ASSISTANT: Done — 5-clip music video saved to /home/me/mv.mp4 (48 MB).
          Session id mv_1745712000_deadbeef saved if you want to re-export.
```

Wall-clock budgets on a warm backend:

| Quality | Model | Resolution / stage 2 | 5-clip MV |
|---|---|---|---|
| `draft` | `ltx-2-3-fast` | 540p effective (`skip_stage_2=true`) | ~3 min |
| `preview` | `ltx-2-3-fast` | 1920×1080 full | ~7 min |
| `final` | `ltx-2-3-pro` | 1920×1080 full | ~12 min |

### 4.4 Resuming a failed run

Failures happen — Modal cold-starts, transient OOM on a single clip, network blips. The orchestrator checkpoints after each step. Recovering is one call:

```text
USER: My MV crashed on clip 3 of 5. Pick it up.

ASSISTANT (tool: mcp__noodlefinger__list_sessions, args: {})
  → [{"session_id": "mv_1745712000_deadbeef",
      "current_phase": "clip_3_submitted",
      "prompt_preview": "A masked drummer plays in a smoky…",
      "age_seconds": 240,
      "final_uri": null}]

ASSISTANT (tool: mcp__noodlefinger__resume_music_video,
           args: {session_id: "mv_1745712000_deadbeef"})

  [internal — orchestrator reads the checkpoint:]
  ├─ Music job already completed → reuse cached audio URI.
  ├─ Clips 0, 1, 2 already completed → reuse video_uri + segment_uri.
  ├─ Clip 3 was in "submitted" state → wait_for_job on the existing job_id.
  │   [if it actually completed in the background while we weren't looking,
  │    we just pick up its result; if it failed, we re-submit with the
  │    cached prior segment]
  ├─ Submit clip 4, extract segments, …
  └─ Composition + export.

  → Same shape as cut_music_video success.
```

Capturing the `session_id` early (it's in the failure response too) lets you retry without losing the 4-5 minutes of completed clip work. Sessions are pruned after 7 days.

### 4.5 Editing-grammar walkthrough (v0.3)

`cut_music_video` v0.2 produces N uniform clips chained by segment-uri — competent assembly, **not editing**. v0.3 adds an editing-grammar layer: an audio-analysis primitive (`get_beat_grid`), a shot-list planner that consumes the beat grid (`plan_shot_list`), and a per-shot composition language with J/L cuts, palette swaps, transitions, speed ramps, B-roll inserts, and synthetic flash frames. The full theory + tool surface is in [`docs/MV_EDITING.md`](MV_EDITING.md); this section is the worked Claude Code transcript.

The contract is locked across taco-backend v1.15.0 (PR 1, this repo — `/v1/music/analyze` + `clip.speed` + `transition.audioLeadFrames`) and noodlefinger-mcp v0.3.0 (PR 2 — `get_beat_grid` / `plan_shot_list` tools).

```text
USER: I have a 32-second hair-metal demo at storage://b3a1f0c4d8e7… —
      cut a music video against it, hard cuts on the snare, smash into
      the chorus, two flash inserts on the chorus bar 1.

ASSISTANT (tool: mcp__noodlefinger__get_beat_grid,
           args: {audio_uri: "storage://b3a1f0c4d8e7…"})
  [calls POST /v1/music/analyze on taco-backend]
  → {bpm: 124.5,
     beats: [0.482, 0.964, 1.446, 1.928, …],   // 67 beats total
     downbeats: [0.482, 2.410, 4.338, …],      // every 4th beat (4/4)
     onsets: [0.482, 0.840, 0.964, 1.213, …],
     rms_envelope: [[0.0, -28.4], [0.512, -22.1], …],
     duration_s: 32.16,
     confidence: 0.91}

  [The high confidence (>0.7) means the beat track is solid — bar 1
   downbeats are reliable. Lower confidence (<0.3) on swung / rubato
   tracks means the LLM should fall back to manual shot durations.]

ASSISTANT (tool: mcp__noodlefinger__plan_shot_list,
           args: {audio_summary: <beat_grid above>,
                  prompt: "lead singer with long black hair, white tank,
                           silver cross, leather pants, smoke-filled stage",
                  genre: "hair-metal",
                  num_beats_per_shot: 8,
                  sections: [{label: "intro", t_start_s: 0.0,  t_end_s: 8.0},
                             {label: "verse", t_start_s: 8.0,  t_end_s: 16.0},
                             {label: "chorus",t_start_s: 16.0, t_end_s: 32.0}]})
  [pure helper — no HTTP. Runs the §4 cut-placement algorithm in
   docs/MV_EDITING.md against the genre prior for hair-metal:
   verse cpb=1.5, chorus cpb=2.5, match-cut budget 5%]
  → {treatment: {vibe: "1987 Sunset Strip, magenta neon, dry-ice smoke",
                 anchor_ref: "Mötley Crüe 'Wild Side' — hard cuts on snare",
                 subject_lock: "Lead — long black hair, white tank, silver
                                cross, leather pants",
                 cadence: "8 shots × ~4s mean. Chorus = 1-beat cuts + flash."},
     shots: [
       {shot_n: 1, duration_s: 4.0, scale: "WS",  framing: "group",
        camera: "locked-off", subject: "band on smoke-filled stage",
        section: "intro", palette: "low-key + neon backlit",
        transition_to_next: "cut",
        prompt: "wide stage shot, four-piece band silhouetted in magenta haze"},
       {shot_n: 2, duration_s: 4.0, scale: "ECU", framing: "insert",
        camera: "locked-off", subject: "fingers on fretboard mid-bend",
        section: "intro", palette: "hard top",
        transition_to_next: "cut",
        prompt: "extreme close-up of guitar fretboard, pick mid-strum"},
       {shot_n: 3, duration_s: 4.0, scale: "MCU", framing: "single",
        camera: "slow push", subject: "Lead [LOCK] mouths verse lyric",
        section: "verse", palette: "rim + smoky",
        transition_to_next: "cut",
        prompt: "medium close-up of lead singer at mic, slow push"},
       {shot_n: 4, duration_s: 4.0, scale: "MWS", framing: "two-shot",
        camera: "handheld drift", subject: "Lead + bassist back-to-back",
        section: "verse", palette: "neon + smoky",
        transition_to_next: "smash",          // hits chorus on the 1
        audio_lead_frames: 6,                  // J-cut into chorus
        prompt: "lead and bassist back-to-back on stage, handheld drift"},
       {shot_n: 5, duration_s: 0.083, scale: "flash", framing: null,
        camera: null, subject: null, section: "chorus",
        flash_color: "#FFFFFF",                // synthetic, no LTX
        transition_to_next: "cut"},
       {shot_n: 6, duration_s: 2.0, scale: "CU", framing: "single",
        camera: "whip pan L", subject: "Lead [LOCK] at mic",
        section: "chorus", palette: "low-key",
        transition_to_next: "cut",
        prompt: "close-up of lead singer screaming chorus, whip pan left"},
       {shot_n: 7, duration_s: 0.083, scale: "flash", framing: null,
        camera: null, subject: null, section: "chorus",
        flash_color: "#FFFFFF",
        transition_to_next: "cut"},
       {shot_n: 8, duration_s: 8.0, scale: "WS", framing: "group",
        camera: "crane up", subject: "band on stage, lighter wave",
        section: "chorus", palette: "warm amber",
        transition_to_next: "cut",
        prompt: "wide crane shot, full band performing, audience cheering"}
     ]}

  [User reviews the shot list. They notice shot 4 has audio_lead_frames=6
   which J-cuts the chorus. They like it — the chorus vocal pre-laps
   into the last 0.25s of the verse. They tweak shot 6 to add speed
   ramp slow-mo.]

USER: Looks good — slow shot 6 down to 0.5x for that classic Mulcahy
      slow-mo on the whip-pan. Ship it.

ASSISTANT [edits shot_list[5].speed = 0.5 in the tool call args]
ASSISTANT (tool: mcp__noodlefinger__cut_music_video,
           args: {prompt: "...",
                  duration_s: 32.0,
                  num_clips: 6,                // flashes don't generate
                  audio_uri: "storage://b3a1f0c4d8e7…",
                  shot_list: <shot_list above with shot 6 speed=0.5>})

  [internal — orchestrator threads each shot into a clip body:
  ├─ Shots with flash_color → mint synthetic 9-frame solid-color MP4
  │   via ffmpeg lavfi, register a synthetic-flash history row, no LTX.
  ├─ Shots with prompt → submit to LTX as normal segment-chained clips.
  ├─ shot.speed → composition.clips[i].speed (B2 backend cascade).
  ├─ shot.audio_lead_frames → composition.transitions[i].audioLeadFrames
  │   (B1 backend two-pass slice refactor).
  ├─ shot.transition_to_next + transition_duration_s → composition.transitions[i].
  └─ Export composition; backend's export_handler.py applies J/L cut
     audio offsets, speed cascades through effective_durations, force-
     keyframe seams land on post-speed positions.]

  → {music_uri: "storage://…", composition_id: "comp_…",
     final_uri: "storage://7c2…",
     shots: [{shot_n, history_id, segment_uri, …}, …],
     beat_grid_used: {bpm: 124.5, downbeats: [...], …}}
```

**What just happened**:

- The LLM (Claude Code) authored the *creative* layer — treatment, subject lock, palette per section, transition types, speed ramp on shot 6.
- The algorithm authored the *rhythmic* layer — cut placement on the metric grid, accent-snap onto the chorus downbeat, half-time ramp into chorus.
- The backend rendered the result deterministically — flashes are synthetic (no model rolls), J-cut audio offset is exact (B1 two-pass math), speed ramp re-times via setpts (B2 cascade), xfade and beat-synced seams land on post-speed timestamps.

The shot list is data the LLM can edit before submission. The user reviewed, tweaked, and shipped. That's the v0.3 editor-tier UX.

For the full theory (Murch's Rule of Six, Goodwin's synaesthesia, the atomic-vs-content-bound split), the genre priors, the per-section density curves, the cut-placement pseudocode, and worked examples for power ballad / hair metal / hip-hop, see [`docs/MV_EDITING.md`](MV_EDITING.md).

---

## 5. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  LLM client (Claude Code, Cursor, Continue, Codex CLI, …)          │
│                                                                     │
│      tool calls  ↑↓  text/json over stdio (MCP framing)            │
└─────────────────┬───────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  noodlefinger-mcp (Python, uvx-launched subprocess)                 │
│                                                                     │
│  src/noodlefinger_mcp/                                              │
│    ├─ server.py        list_tools / call_tool / read_resource       │
│    ├─ actions.py       tier-1 tool implementations                  │
│    ├─ orchestrator.py  cut_music_video / resume_music_video         │
│    ├─ http.py          shared httpx.AsyncClient + retry policy      │
│    ├─ _session.py      checkpoint read/write (atomic, fcntl-locked) │
│    └─ _data/                                                        │
│         ├─ endpoints.json   (bundled — 77 endpoints)                │
│         └─ flows.json       (bundled — 16 walkthroughs)             │
│                                                                     │
│  Tier-0 path: pure local JSON reads, no network.                    │
│  Tier-1 path: httpx.AsyncClient → api.noodlefinger.io.              │
└─────────────────┬───────────────────────────────────────────────────┘
                  │ HTTPS, Bearer auth, 3-attempt retry,
                  │ Retry-After honored, follow_redirects=True
                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  api.noodlefinger.io (taco-backend, FastAPI on uvicorn:8090)        │
│  Cloudflare-tunneled. Same surface documented in docs/API.md.       │
└─────────────────────────────────────────────────────────────────────┘
```

**Why a single binary, not two servers.** Tier-0 stays useful (and harmless) without credentials. Tier-1 layers on top with no separate install. Conditional registration (`NOODLEFINGER_API_KEY` set → 14 extra tools, unset → tools hidden) means an LLM client never sees an action tool it can't actually call.

**Why polling, not SSE.** `wait_for_job` polls `GET /v2/jobs/{id}` every 2 s, switching to 5 s after 2 minutes. SSE would be more efficient but `httpx.stream()` complicates the client substantially and the overhead at 2 s/poll is negligible — a 16-min final-quality job is ~480 polls. The backend's `GET /v2/jobs/{id}/stream` endpoint exists and is documented for browser clients (flow `live-progress-via-sse`), but the MCP tool stays on the polling path.

**File layout when installed.** `uvx` builds a wheel from the noodle-portal git checkout's `mcp/` subdirectory. The wheel includes `src/noodlefinger_mcp/_data/endpoints.json` + `flows.json` so the server is fully self-contained — no clone of the portal repo needed. CHANGELOG defaults to the dev box path; on a fresh install, set `NOODLEFINGER_MCP_CHANGELOG` or expect `get_changelog` to fail soft.

---

## 6. Tools reference

### Tier-0 (always available)

| Tool | Inputs | Description |
|---|---|---|
| `search_endpoints` | `query: str`, `limit?: int`, `method?: str`, `group?: str` | Lexical search over the catalog. Returns top-N endpoint summaries. |
| `get_endpoint` | `endpoint_id: str` | Full endpoint spec — body schema, response, errors, code in 3 langs. |
| `list_groups` | — | All 13 endpoint groups. |
| `list_flows` | — | All 16 step-by-step generation walkthroughs. |
| `get_flow` | `flow_id: str` | Full walkthrough with curl/python/node code per step + gotchas. |
| `get_changelog` | `version?: str` | Latest CHANGELOG entry, or a specific `vX.Y.Z` section. |

### Tier-1 (require `NOODLEFINGER_API_KEY`)

| Tool | Inputs | Description |
|---|---|---|
| `submit_job` | `endpoint_id: str`, `params: object` | Generic POST passthrough to any v2 generation endpoint. Returns `{job_id, status, poll_url, stream_url}`. |
| `get_job` | `job_id: str` | Single status snapshot. Returns `{status, progress, phase, result_storage_uri, error}`. |
| `wait_for_job` | `job_id: str`, `timeout_s?: number=600`, `poll_interval_s?: number=2.0` | Polls until terminal status. Returns the final job dict. |
| `cancel_job` | `job_id: str` | DELETE `/v2/jobs/{id}`. Best-effort for processing jobs (~1-3 s latency). |
| `download_job_result` | `job_id: str`, `out_path?: str` | Download final media. Default path: `~/.cache/noodlefinger-mcp/results/<job_id>.<ext>`. |
| `download_storage_uri` | `storage_uri: str`, `out_path?: str` | Download any `storage://<uuid>` (uploads, results, segments) to disk. |
| `upload_file` | `local_path: str`, `mime?: str` | Two-step slot+PUT upload. Default mime `application/octet-stream` (bypasses magic-byte check). |
| `extract_segment` | `video_uri: str`, `start_frame: int`, `num_frames?: 9\|17\|25\|33=9` | Synchronous tail-segment extract for chain conditioning. |
| `create_composition` | `name: str`, `clips?: array`, `transitions?: array`, `audio_uri?: str`, `chain_mode?: "hardcut"\|"seamless"\|"seamless-segment"` | POST `/v2/compositions`. Returns `{composition_id, name, clip_count}`. |
| `export_composition` | `comp_id: str`, `audio_uri?: str` | Enqueue export → ffmpeg concat + audio overlay → final H.264+AAC MP4. |
| `get_changelog_for_endpoint` | `endpoint_id: str` | Cross-reference an endpoint's `since_version` with the CHANGELOG. |
| `cut_music_video` | `prompt`, `music_prompt`, `duration_s`, `num_clips?=5`, `quality?="preview"`, `subject_anchor_uri?`, `shot_list?`, `music_lyrics?`, `music_style?`, `music_bpm?`, `chain_mode?="keyframes"`, `parallel_clips?="auto"`, `slice_strategy?="uniform"`, `beat_analyzer?="auto"`, `min_clip_duration_s?=0.5`, `lora?={id,strength}`, `enhance_prompt?=false`, `session_id?` | The orchestrator. Music → N a2v clips with chained keyframes → composition → export. v0.4.4: parallel clip dispatch. v0.4.6 default chain mode flipped to `"keyframes"` (soft pin via `VideoConditionByKeyframeIndex`). v0.5: opt-in beat-aligned slicing via `slice_strategy="downbeats"`. v0.6: top-level + per-shot `lora` (per-shot wins). Blocking; checkpointed; resume-safe. |
| `resume_music_video` | `session_id: str` | Resume a `cut_music_video` session that failed or was interrupted. Skips completed steps. |
| `list_sessions` | — | List locally cached MV sessions. Auto-prunes entries >7 days old. |
| `get_beat_grid` | `audio_uri: str`, `analyzer?: "librosa"\|"madmom"="librosa"` | Wraps `POST /v1/music/analyze`. Returns `{bpm, beats[], downbeats[], onsets[], rms_envelope[][2], duration_s, confidence, analyzer_used?}`. Used by `cut_music_video` v0.5 beat-align internally; also user-callable for hand-authoring beat-aligned shot lists. |
| `plan_shot_list` | `audio_summary: dict`, `prompt: str`, `genre?: str="modern-pop"`, `num_beats_per_shot?: int=8`, `sections?: array` | Pure helper: density-driven, section-aware shot-list authoring with genre presets. Returns shots with `prompt`, `duration_s`, `audioStart_s`, `section`, `palette`, `scale`, `camera`, `transition_to_next` per item. Pair with `get_beat_grid` for beat-aware authoring. |
| `match_cut_prompt_pair` | `subject: str`, `motif: str` | Pure helper: returns prompt pair `{a, b}` for a beat-aligned match-cut (J-cut / L-cut continuity through a visual motif). |
| `weave_inserts` | `primary_shots: array`, `inserts: array`, `cadence?: int=4` | Pure helper: interleaves insert / b-roll shots into a primary shot list at the given cadence. |
| `apply_section_palette` | `shots: array`, `palette: dict` | Pure helper: applies per-section color palette + grade hints to shots. |
| `list_loras` | `family?: "ltx"\|"flux"\|"all"="all"`, `strategy?: str` | v0.6: wraps `GET /v1/loras` and/or `GET /v1/flux-loras`. Returns `{ltx: [...], flux: [...], count: {...}}` with full registry metadata (id, name, strategy, trigger_word, description). |
| `get_lora` | `lora_id: str`, `family?: "ltx"\|"flux"="ltx"` | v0.6: single-LoRA lookup with full metadata. Use to inspect `trigger_word` before authoring prompts that need it. |
| `search_loras` | `query: str`, `family?: "ltx"\|"flux"\|"all"="all"`, `limit?: int=20` | v0.6: substring search over `name`, `description`, `trigger_word`. Sort: name > trigger_word > description. |
| `rescan_flux_loras` | — | v0.6: wraps `POST /v1/flux-loras/rescan`. Tells the backend to reindex `flux_loras/` after dropping a new `.safetensors`. |

### Tool → endpoint mapping

| Tool | Endpoint(s) |
|---|---|
| `submit_job` | any v2 generation POST returning `SubmissionEnvelope` |
| `get_job`, `wait_for_job` | `GET /v2/jobs/{id}` |
| `cancel_job` | `DELETE /v2/jobs/{id}` |
| `download_job_result` | `GET /v2/jobs/{id}/result` |
| `download_storage_uri` | `GET /uploads/get/{upload_id}` |
| `upload_file` | `POST /v1/upload` + `PUT /uploads/put/{upload_id}` |
| `extract_segment` | `POST /v2/video/extract-segment` |
| `create_composition` | `POST /v2/compositions` |
| `export_composition` | `POST /v2/compositions/{id}/export` |
| `cut_music_video` | composes `/v2/music` → `/v2/audio-to-video` × N → `/v2/video/extract-segment` × (N-1) → `/v2/compositions` → `/v2/compositions/{id}/export` |

Each tool's full inputSchema and result projection live in [`mcp/_phase2/01-tools-trace.md`](https://github.com/tacos8me/noodle-portal/blob/main/mcp/_phase2/01-tools-trace.md). The orchestrator's state machine and quality-tier mapping live in [`mcp/_phase2/02-orchestrator-design.md`](https://github.com/tacos8me/noodle-portal/blob/main/mcp/_phase2/02-orchestrator-design.md).

---

## 7. Roadmap

### Shipped between v0.2 and v0.6

| # | Feature | Status |
|---|---|---|
| **B1** | `get_beat_grid(audio_uri)` — BPM + downbeat + onset extraction. | **Shipped v0.4.0** as `get_beat_grid` (wraps `POST /v1/music/analyze`). v0.5.0 wires it into `cut_music_video` automatically when `slice_strategy="downbeats"` is passed. |
| — | Parallel clip dispatch in `cut_music_video`. | **Shipped v0.4.4** — DAG-aware executor auto-scales to live worker count via `GET /v1/system/workers`. |
| — | Chain-mode soft-pin default (motion fix). | **Shipped v0.4.6** — flipped from `seamless-segment` (hard-pin propagation suppressed motion in first ~3s of every follower clip) to `keyframes` (soft-guide via `VideoConditionByKeyframeIndex`). |
| — | LoRA browse + apply in `cut_music_video`. | **Shipped v0.6.0** — `list_loras` / `get_lora` / `search_loras` / `rescan_flux_loras` tier-1 tools + top-level + per-shot `lora` schema fields. |

### Open candidates

| # | Feature | Blocker |
|---|---|---|
| **B4** | Lyric timestamps: `lyrics_timestamps: [{line, start_s, end_s}]` on music job results so a shot can pin to lyric line N. | Backend: ACE Step already knows where vocals land — needs to surface them in the music job result payload. |
| **B7** | `share_to_gallery(comp_id, label)` — one-click "publish a draft to the portal gallery" so the user can review without leaving the LLM session. | MCP only: thin wrapper over `POST /v1/approved-images`. Trivial; deferred to keep v0.6 surface tight. |
| — | Auto-`trigger_word` injection — when `lora` is set, automatically prepend `trigger_word` to the per-clip prompt if it's not already present. | MCP only: ~30 LOC opt-in flag. Deferred to v0.6.1. |
| — | SSE `wait_for_job` variant — instead of polling, open `/v2/jobs/{id}/stream` and consume server-sent events. ~10× lower poll volume on long jobs. | MCP only: `httpx.stream()` complexity. Not user-facing important. |
| — | Phrase-boundary-aware slicing (beyond beats / downbeats). | Backend: needs an `allin1`-style sidecar or extension to madmom; current `/v1/music/analyze` doesn't return phrase markers. |
| — | LoRA upload tool (`POST /v1/loras` multipart from MCP). | MCP only: multipart-from-tool-call is operationally awkward. v0.6 deferred to ops via curl/dashboard. |
| — | PyPI publish under `noodlefinger-mcp` for signed-by-PyPI distribution + simpler `uvx noodlefinger-mcp` invocation (no git URL). | None — release engineering only. |

If you have a use case for one of these, file an issue at [github.com/tacos8me/noodle-portal](https://github.com/tacos8me/noodle-portal).

---

## 8. Troubleshooting

**`claude mcp list` shows "Failed to connect" for `noodlefinger`.**
Check that `uvx` is in your PATH. `which uvx` should print a path. If not, install [astral-sh/uv](https://docs.astral.sh/uv/getting-started/installation/) — the MCP launches via `uvx` from a git URL on every start. The first launch downloads the wheel (~5 s); subsequent launches are cached and start in <1 s.

**Tier-1 tools missing — `mcp__noodlefinger__submit_job` returns "tool not found".**
Set `NOODLEFINGER_API_KEY` in the env config of your client's MCP entry and restart the client. The 23 tier-1 tools are conditionally registered at `list_tools()` time; if the env var is unset at server start, they never appear in the tool list. Run `mcp__noodlefinger__list_groups` (tier-0) to confirm the server is otherwise healthy.

**`wait_for_job` returns `timeout: job did not complete in 600s`.**
The backend may be in turbo mode (music + image gen blocked, video only) or a sidecar may be down. Check `https://api.noodlefinger.io/health` first. For long jobs (final-quality 4K outpaint, 30 s retake) bump `timeout_s` to 1800. If the job actually completed but the timeout fired first, re-run `get_job(job_id)` — the result is still cached server-side for 30 days.

**`cut_music_video` failed mid-pipeline; `resume_music_video` returns `not_found: session <id>`.**
The session file isn't where the orchestrator expects it. Check `$NOODLEFINGER_SESSION_DIR` — if you set it in shell but the MCP was launched via `claude mcp add` without forwarding the env var, the orchestrator wrote to the default `~/.cache/noodlefinger-mcp/sessions/` instead. Add `-e NOODLEFINGER_SESSION_DIR=/absolute/path` to the `claude mcp add` invocation.

**`upload_file` returns `validation_error: content_type_mismatch`.**
The backend's magic-byte check rejects MIME headers that don't match the file's actual bytes (since v1.8.2). Either pass the correct `mime` argument, or omit it entirely — the default `application/octet-stream` bypasses the check by design.

**`cut_music_video` runs forever on `quality="final"`.**
Final quality is `ltx-2-3-pro` at 1920×1080 with full stage 2. Five clips × ~80 s/clip + composition + export ≈ 8-12 min on a warm backend, longer when cold (Modal startup, local LTX cold-load). Use `quality="preview"` (`ltx-2-3-fast`, ~3-5 min total) for iteration; switch to final only when you're happy.

**`get_changelog` returns `not found: changelog file missing`.**
The default `NOODLEFINGER_MCP_CHANGELOG` path points at the dev box. On a fresh install, either point it at a checked-out copy of `taco-backend/CHANGELOG.md` or expect `get_changelog` to soft-fail. The other 5 tier-0 tools work without it.

**Server logs at stderr are spamming my terminal.**
The server logs structured JSON to stderr — that's where MCP servers are expected to log (stdout is reserved for protocol framing). Your client should be redirecting stderr to its own log file. If you're running the server directly for debugging, redirect with `2> /tmp/mcp.log`.
