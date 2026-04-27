# noodlefinger-mcp — LLM-driven workflows over the taco-backend API

**Server version:** v0.2.1 (2026-04-27) · **Catalog source:** [github.com/tacos8me/noodle-portal](https://github.com/tacos8me/noodle-portal) (`mcp/` subdir) · **Status:** end-to-end live-validated against `localhost:8090` (single MCP `cut_music_video` call → real 1.6 MB H.264+AAC MP4 in 82 s at draft quality)

> **What this is:** an [MCP](https://modelcontextprotocol.io) server that wraps the entire `api.noodlefinger.io` surface so an LLM client (Claude Code, Cursor, Continue, Codex CLI, OpenCode, …) can discover endpoints, submit jobs, and orchestrate multi-step pipelines without re-reading `docs/API.md`. Two tiers: **tier-0** (anonymous, docs-only, 6 tools) is always available; **tier-1** (authenticated, 14 action tools) registers iff `NOODLEFINGER_API_KEY` is set.
>
> See also: [docs/API.md](API.md) (canonical HTTP contract) · [docs/QUICKSTART.md](QUICKSTART.md) · [the noodle-portal repo](https://github.com/tacos8me/noodle-portal).

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

A single `noodlefinger-mcp` binary, distributed as a Python wheel via `uvx --from git+...`, that speaks the [Model Context Protocol](https://modelcontextprotocol.io) over stdio. It bundles:

- The **endpoint catalog** (`endpoints.json`, 77 endpoints across 13 groups) — same data that powers `portal.noodlefinger.io/docs`.
- The **flow walkthroughs** (`flows.json`, 16 step-by-step recipes) — `text-to-video`, `audio-to-video`, `video-hdr`, `cut-music-video`, …
- The **CHANGELOG** for cross-referencing endpoint `since_version`.

When tier-1 is enabled (env var `NOODLEFINGER_API_KEY`), it additionally:

- Submits jobs against `api.noodlefinger.io` and waits for them with a single tool call.
- Uploads files, downloads results, extracts segments, builds compositions, exports MP4s.
- Runs the `cut_music_video` macro: music → N a2v clips with seamless chain conditioning → composition → export, all in one tool call with session-resume on failure.

### Tool / resource counts

| Surface | Count | Tier | Notes |
|---|---|---|---|
| Tools — discovery | 6 | tier-0 | `search_endpoints`, `get_endpoint`, `list_groups`, `list_flows`, `get_flow`, `get_changelog` |
| Tools — actions | 14 | tier-1 | `submit_job`, `get_job`, `wait_for_job`, `cancel_job`, `download_job_result`, `download_storage_uri`, `upload_file`, `extract_segment`, `create_composition`, `export_composition`, `get_changelog_for_endpoint`, `cut_music_video`, `resume_music_video`, `list_sessions` |
| Tools — total | **20** | — | 6 + 14 |
| Resources | 3 | tier-0 | `noodlefinger://endpoints`, `noodlefinger://flows`, `noodlefinger://changelog` |

---

## 2. Connection

### Recommended: Claude Code, scoped to your user

```bash
# Tier-0 only (anonymous docs)
claude mcp add --scope user noodlefinger uvx -- \
  --from "git+https://github.com/tacos8me/noodle-portal#subdirectory=mcp" \
  noodlefinger-mcp

# Tier-0 + tier-1 (authenticated actions)
claude mcp add --scope user \
  -e NOODLEFINGER_API_KEY=nf_live_sk_... \
  noodlefinger uvx -- \
  --from "git+https://github.com/tacos8me/noodle-portal#subdirectory=mcp" \
  noodlefinger-mcp
```

`claude mcp add` writes to `~/.claude.json` under `mcpServers.noodlefinger`. Restart Claude Code, then run `/mcp` — you should see `noodlefinger` listed and connected.

### Pinning a specific commit (recommended for repeatable installs)

```bash
claude mcp add --scope user noodlefinger uvx -- \
  --from "git+https://github.com/tacos8me/noodle-portal@<sha>#subdirectory=mcp" \
  noodlefinger-mcp
```

Pinning a SHA defends against an accidental or malicious upstream change between installs. See [§3 Security](#3-security).

### Direct settings.json (Cursor, Continue, Codex CLI, OpenCode)

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

Cursor: `Cmd-Shift-P → Cursor: Open MCP Settings`. Continue / Codex CLI: their docs cover the same `mcpServers` shape.

### Environment variables

| Var | Default | Purpose |
|---|---|---|
| `NOODLEFINGER_API_KEY` | unset | Bearer token for tier-1 calls. **Tier-1 tools are not registered when this is unset** — anonymous installs see only the 6 docs tools. |
| `NOODLEFINGER_API_BASE` | `https://api.noodlefinger.io` | Override for self-hosting / local dev (e.g. `http://localhost:8090`). Trailing slash is stripped. |
| `NOODLEFINGER_SESSION_DIR` | `~/.cache/noodlefinger-mcp/sessions/` | Where `cut_music_video` checkpoints live. See [§3 Security](#3-security) for permissions. |
| `NOODLEFINGER_MCP_CATALOG` | bundled `_data/endpoints.json` | Override the catalog path (point at a private fork). |
| `NOODLEFINGER_MCP_FLOWS` | bundled `_data/flows.json` | Override the flows path. |
| `NOODLEFINGER_MCP_CHANGELOG` | `/mnt/nvme-1/servers/taco-backend/CHANGELOG.md` | Override the changelog path (default is the dev box; for a clean install, point at a checked-out copy or remove the env var to disable changelog reads). |

### First-run sanity check

After registering, restart your client and:

1. Run `/mcp` (Claude Code) — `noodlefinger` should be `connected`.
2. Call `mcp__noodlefinger__list_groups` — should return all 13 groups (`video-generation`, `image-generation`, `music-generation`, `batch`, `jobs-lifecycle`, `history`, `uploads`, `compositions`, `approved-images`, `loras`, `chat`, `system`, `dashboard-meta`).
3. With tier-1 enabled, call `mcp__noodlefinger__list_sessions` — returns `[]` on first run (no sessions yet).

If `list_groups` returns "tool not found", check the server is connected — your client may have failed to launch `uvx`.

---

## 3. Security

<!-- mcp-security agent fills this in -->

*Filled in 2026-04-27 from the v0.2 security audit ([`mcp/_phase2/04-security-audit.md`](https://github.com/tacos8me/noodle-portal/blob/main/mcp/_phase2/04-security-audit.md), 12 findings, 0 ACCEPTABLE blockers, 3 RECOMMENDED, 5 ACCEPTABLE — plus 3 build-time BLOCKERs already filed against the implementation team).*

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

The LLM client is **semi-trusted** — it can submit arbitrary tool arguments, but it is constrained by the catalog-driven endpoint allow-list and the input schemas. The noodlefinger.io API is the **outer trust boundary** and enforces per-key quotas and admin-bearer guards on system endpoints. Worst-case for a compromised IDE plugin: it can use your bearer to consume per-key generation quota and produce content under your account — it cannot toggle turbo, unload models, or scale the remote pool through the MCP. **If you suspect compromise, revoke the key in the portal admin UI;** all in-flight orchestrator sessions die on the next HTTP call.

Recommended hygiene: rotate keys periodically, and consider a secondary low-quota key dedicated to MCP use so a leak doesn't compromise your primary key's quota.

### API key handling (audit § 1, § 11)

`NOODLEFINGER_API_KEY` is read once at startup into a module-level constant, used to build `Authorization: Bearer <key>` headers, and **never** written to disk, stdout, or session checkpoints. Structured stderr logs include only `{tool, method, path, status, elapsed_s}` — no headers, no bodies, no prompt text. The logger is a closed-form helper (`_log_call(tool, method, path, status, elapsed_s, slow=False)`) with named parameters only, no `**kwargs`, so a future maintainer can't accidentally add `headers=...` or `body=...` and leak secrets. `tests/test_no_secret_leak.py` asserts the test key never appears in captured stderr across every tier-1 tool.

### Supply chain (audit § 3) — pin a commit SHA

The `uvx --from "git+...#subdirectory=mcp"` install line resolves the default branch HEAD at install time. With no `@<ref>` qualifier, every fresh `uvx` invocation could pull a different commit if upstream moves. **A compromised upstream repo** — push access leaked, account compromised, malicious PR merged — could ship an MCP server that exfiltrates your `NOODLEFINGER_API_KEY` to an attacker host on next launch. Blast radius: every developer who has the MCP registered.

```bash
# Recommended (production): pin to a tagged release commit SHA
claude mcp add --scope user noodlefinger uvx -- \
  --from "git+https://github.com/tacos8me/noodle-portal@<sha>#subdirectory=mcp" \
  noodlefinger-mcp

# Development only — auto-tracks main, accepts upstream changes silently
claude mcp add --scope user noodlefinger uvx -- \
  --from "git+https://github.com/tacos8me/noodle-portal#subdirectory=mcp" \
  noodlefinger-mcp
```

v0.3 publishes to PyPI under `noodlefinger-mcp`, after which signed-by-PyPI distribution is available; the git-SHA pattern remains the v0.2 best practice.

### Session checkpoint file permissions (audit § 2)

Session files at `$NOODLEFINGER_SESSION_DIR/<session_id>.json` (default `~/.cache/noodlefinger-mcp/sessions/`) contain `prompt`, `music_lyrics`, `subject_anchor_uri`, `final_uri`, and intermediate `storage://` URIs. On a multi-user box these would otherwise be world-readable at default umask `0o644`. The server enforces:

- `os.makedirs(session_dir, mode=0o700, exist_ok=True)` + `os.chmod(session_dir, 0o700)` on init (idempotent repair if the dir exists with looser perms from a prior version).
- `os.chmod(final_path, 0o600)` immediately after every atomic-write, because umask interacts with `O_CREAT` mode bits and setting mode at `open()` time is insufficient.

`fcntl.flock(LOCK_EX)` is held continuously across the `os.replace()` rename and released only when the fd is closed, preventing concurrent-resume races.

Note: even with mode `0o600` on the checkpoint, the bytes referenced by `storage://` URIs are still fetchable by **anyone with both the URI and a valid bearer for your key**. Treat URIs as capabilities. Windows portability for the chmod calls is out-of-scope for v0.2 (POSIX only).

### `submit_job` allow-list (audit § 4) — admin endpoints excluded

Tier-1 `submit_job(endpoint_id, params)` is a generic POST passthrough, so it must defend against the LLM trying to call `/v1/system/turbo` or `/v1/ltx/unload`. The implementation gates with **two assertions** before any HTTP call:

```python
if ep.get("method") != "POST":
    return _err(f"endpoint {endpoint_id} is not a POST endpoint")
if ep.get("response", {}).get("schema") != "SubmissionEnvelope":
    return _err(f"endpoint {endpoint_id} does not return a job (not a submission endpoint)")
```

System-control endpoints return `{"ok": true}` or telemetry dicts — not `SubmissionEnvelope` — so they're rejected by the schema gate. The URL is constructed from the catalog `path` (data) + `_base_url()` (env var) — no string interpolation of user input, no shell-out anywhere in the codepath. Defense in depth: the backend's admin-bearer guard would reject those endpoints with 403 even if the gate were bypassed, since user keys aren't in `.admin_keys`.

`tests/test_tools_tier1.py::test_submit_job_rejects_admin_endpoint` calls `submit_job("system-turbo", {"enable": true})` and asserts the response contains "not a submission endpoint".

### `upload_file` magic-byte bypass (audit § 5) — known and documented

`upload_file` defaults `Content-Type: application/octet-stream`. The taco-backend treats octet-stream as opt-out from magic-byte verification — by design, to support exotic media that legitimately fails sniffing. A malicious caller could upload non-media bytes (executable, archive, exfiltrated text) and reference the resulting `storage://` URI in a generation request. The backend's PyAV decode fails safely on non-media bytes; there is no path where octet-stream uploads are executed or trusted as having a particular structure. The bytes do consume your per-key `PER_KEY_UPLOAD_BYTES_PER_DAY` quota — that's the abuse cap.

If you have the correct MIME type, pass it explicitly via the `mime` argument and the backend will validate the bytes match.

### Retry budget (audit § 6) — per-call, not orchestrator-wide

Three attempts per HTTP call, backoff `[1s, 3s]` jittered, retryable on `{429, 502, 503, 504, ConnectError, ReadTimeout}`, with `Retry-After` honored on 429. **Non-retryable 4xx codes (400, 401, 403, 404, 409, 413, 422) fail immediately.** The orchestrator's per-step retry budgets are scoped per-call — not shared across the orchestrator — so a flaky backend triggering 3 retries on every HTTP call cannot snowball a 12-call pipeline into 12 × N exponential explosion. Worst case for `cut_music_video(num_clips=5)`: `1 (music) + 5 (a2v) + 4 (segment) + 1 (composition) + 1 (export) = 12 calls × 3 = 36 max HTTP requests`.

The 1800 s default `timeout_s` on `cut_music_video` is the global ceiling.

### Local cache hygiene (audit § 10) — manual cleanup for v0.2

`download_job_result` writes to `~/.cache/noodlefinger-mcp/results/<job_id>.<ext>`; `download_storage_uri` writes to `~/.cache/noodlefinger-mcp/uploads/<uuid>.<ext>`. Neither has eviction in v0.2 — over months of daily `cut_music_video` use a developer can accumulate tens of GB. Result files are written `0o600` (matching session-file hygiene; they may contain proprietary generated content), but disk size grows unbounded.

**v0.2 recommendation:** `rm -rf ~/.cache/noodlefinger-mcp/results/` periodically. v0.3 will add a `--max-cache-size` env var with LRU eviction (oldest mtime first) or auto-evict files older than 30 days on each download.

### What the MCP does NOT do

- **No system-control endpoints.** Tier-1 deliberately excludes `POST /v1/system/{turbo,pause,resume}`, `POST /v1/{flux,ltx}/{unload,reload}`, and `POST /v1/system/pool/remote-workers`. An LLM cannot take the backend offline, toggle turbo, scale the remote pool, or reload models. Operational controls stay on the dashboard.
- **No cookies, no session middleware, no HTTP server.** MCP is stdio-only — the process speaks JSON-RPC over stdin/stdout to one parent (the LLM client), not to the network.
- **No telemetry.** No usage counters, no error reporting back to a server, no analytics. The server only ever talks to `NOODLEFINGER_API_BASE`.
- **No PyPI publish (yet).** v0.2 is git-distribution only via `uvx --from`. v0.3 ships PyPI for signed-by-PyPI distribution.

<!-- end-security -->

---

## 4. Usage

Each transcript below is a real (lightly abbreviated) MCP session. JSON outputs are trimmed for readability — actual responses include all fields documented in [§6 Tools reference](#6-tools-reference).

### 4.1 Tier-0: discovering an endpoint without an API key

The setup: an LLM client booted without `NOODLEFINGER_API_KEY`. Only the 6 docs tools are visible. Goal: find the right endpoint to drive video from a song.

```text
USER: How do I drive video from an audio file with this API?

ASSISTANT (tool: mcp__noodlefinger__list_groups)
  → returns 12 groups, including "v2-generation"

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

Setup: `NOODLEFINGER_API_KEY` set. Goal: take a local `song.mp3`, generate a 6-second clip driven by it, save the MP4 locally.

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
  → {"job_id": "job_xyz789", "status": "queued",
     "stream_url": "/v2/jobs/job_xyz789/stream"}

ASSISTANT (tool: mcp__noodlefinger__wait_for_job,
           args: {job_id: "job_xyz789", timeout_s: 600})
  → polls every 2s; after ~45s returns:
     {"status": "completed", "result_storage_uri": "storage://fedcba…",
      "result_media_type": "video/mp4"}

ASSISTANT (tool: mcp__noodlefinger__download_job_result,
           args: {job_id: "job_xyz789",
                  out_path: "/home/me/clip.mp4"})
  → {"path": "/home/me/clip.mp4", "size_bytes": 12_400_000,
     "media_type": "video/mp4"}

ASSISTANT: Saved to /home/me/clip.mp4 (12.4 MB).
```

Three tool calls (upload, submit+wait, download). The `wait_for_job` polls internally — the LLM does not see the per-poll requests. Compared to the bash-loop-with-curl approach this avoids the most common bug (forgetting to escape a `?token=` query param when the SSE handshake fails).

### 4.3 Tier-1 macro: `cut_music_video` end-to-end

Setup: `NOODLEFINGER_API_KEY` set. Goal: a complete 30-second music video with 5 chained clips, exported as a single MP4. One tool call.

```text
USER: Cut me a 30-second music video — five clips of a masked drummer in a smoky
      room, dark cinematic percussion soundtrack. Preview quality is fine.

ASSISTANT (tool: mcp__noodlefinger__cut_music_video,
           args: {prompt: "A masked drummer plays in a smoky red-lit room",
                  music_prompt: "dark cinematic percussion, slow build",
                  duration_s: 30, num_clips: 5, quality: "preview"})

  [internally — all hidden from the LLM, all checkpointed:]
  ├─ POST /v2/music                        (~30s)
  ├─ wait_for_job (music)                  → audio storage URI
  ├─ for clip in 0..4:
  │    POST /v2/audio-to-video             (clip submit, ~50s wait)
  │    [if clip > 0: segment_uri from prior tail]
  │    POST /v2/video/extract-segment      (~2s, only if clip < 4)
  │    [historyId = job_id — no /v2/history lookup needed]
  │    [checkpoint to ~/.cache/noodlefinger-mcp/sessions/<id>.json]
  ├─ POST /v2/compositions                 (timeline blob)
  └─ POST /v2/compositions/{id}/export
       wait_for_job (export)               → final MP4 storage URI

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
           args: {job_id: "job_exp456",
                  out_path: "/home/me/mv.mp4"})
  → {"path": "/home/me/mv.mp4", "size_bytes": 48_200_000}

ASSISTANT: Done — 5-clip music video saved to /home/me/mv.mp4 (48 MB).
          Session id mv_1745712000_deadbeef saved if you want to re-export
          with different audio later.
```

Total wall-clock: ~7 minutes for `quality="preview"` on a warm backend. `quality="draft"` cuts that to ~3 minutes (fast model + `skip_stage_2=true`); `quality="final"` runs ~12 minutes (`ltx-2-3-pro`).

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

  [internally, the orchestrator reads the checkpoint:]
  ├─ Music job already completed → reuse cached audio URI.
  ├─ Clips 0, 1, 2 already completed → reuse their video_uri + segment_uri.
  ├─ Clip 3 was in "submitted" state → wait_for_job on the existing job_id.
  │   [if it actually completed in the background while we weren't looking,
  │    we just pick up its result; if it failed, we re-submit with the
  │    cached prior segment]
  ├─ Submit clip 4, extract segments, …
  ├─ Composition + export.

  → Same shape as cut_music_video success.
```

Capturing the `session_id` early (it's in the failure response too) lets you retry without losing the 4-5 minutes of completed clip work. Sessions are pruned after 7 days.

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

**Why polling, not SSE.** `wait_for_job` polls `GET /v2/jobs/{id}` every 2 s, switching to 5 s after 2 minutes. SSE would be more efficient but `httpx.stream()` complicates the client substantially and the overhead at 2 s/poll is negligible — a 16-min final-quality job is ~480 polls. The backend's `GET /v2/jobs/{id}/stream` endpoint exists and is documented for browser clients ([flow](https://github.com/tacos8me/noodle-portal/tree/main/data/flows.json) `live-progress-via-sse`), but the MCP tool stays on the polling path.

**File layout when installed.** `uvx` builds a wheel from the noodle-portal git checkout's `mcp/` subdirectory. The wheel includes `src/noodlefinger_mcp/_data/endpoints.json` + `flows.json` so the server is fully self-contained — no clone of the portal repo needed. CHANGELOG defaults to the dev box path; on a fresh install, set `NOODLEFINGER_MCP_CHANGELOG` or expect `get_changelog` to fail soft.

---

## 6. Tools reference

### Tier-0 (always available)

| Tool | Tier | Inputs | Description |
|---|---|---|---|
| `search_endpoints` | 0 | `query: str`, `limit?: int`, `method?: str`, `group?: str` | Lexical search over the catalog. Returns top-N endpoint summaries. |
| `get_endpoint` | 0 | `endpoint_id: str` | Full endpoint spec — body schema, response, errors, code in 3 langs. |
| `list_groups` | 0 | — | All 13 endpoint groups (sidebar structure). |
| `list_flows` | 0 | — | 16 step-by-step generation walkthroughs. |
| `get_flow` | 0 | `flow_id: str` | Full walkthrough with curl/python/node code per step + gotchas. |
| `get_changelog` | 0 | `version?: str` | Latest CHANGELOG entry, or a specific `vX.Y.Z` section. |

### Tier-1 (require `NOODLEFINGER_API_KEY`)

| Tool | Tier | Inputs | Description |
|---|---|---|---|
| `submit_job` | 1 | `endpoint_id: str`, `params: object` | Generic POST passthrough to any v2 generation endpoint. Returns `{job_id, status, poll_url, stream_url}`. |
| `get_job` | 1 | `job_id: str` | Single status snapshot. Returns `{status, progress, phase, result_storage_uri, error}`. |
| `wait_for_job` | 1 | `job_id: str`, `timeout_s?: number=600`, `poll_interval_s?: number=2.0` | Polls until terminal status. Returns the final job dict. |
| `cancel_job` | 1 | `job_id: str` | DELETE `/v2/jobs/{id}`. Best-effort for processing jobs (~1-3s latency). |
| `download_job_result` | 1 | `job_id: str`, `out_path?: str` | Download final media to local path. Default: `~/.cache/noodlefinger-mcp/results/<job_id>.<ext>`. |
| `download_storage_uri` | 1 | `storage_uri: str`, `out_path?: str` | Download any `storage://<uuid>` (uploads, results, segments) to disk. |
| `upload_file` | 1 | `local_path: str`, `mime?: str` | Two-step slot+PUT upload, returns `{storage_uri, size_bytes}`. Default mime is `application/octet-stream` (bypasses magic-byte check). |
| `extract_segment` | 1 | `video_uri: str`, `start_frame: int`, `num_frames?: 9\|17\|25\|33=9` | Synchronous: extract a tail segment for chain conditioning. Returns `{segment_uri, width, height, num_frames, fps}`. |
| `create_composition` | 1 | `name: str`, `clips?: array`, `transitions?: array`, `audio_uri?: str`, `chain_mode?: "hardcut"\|"seamless"\|"seamless-segment"` | POST `/v2/compositions` with the agreed timeline shape. Returns `{composition_id, name, clip_count}`. |
| `export_composition` | 1 | `comp_id: str`, `audio_uri?: str` | Enqueue export → ffmpeg concat + audio overlay → final H.264+AAC MP4. Returns the export `{job_id, …}`. |
| `get_changelog_for_endpoint` | 1 | `endpoint_id: str` | Cross-reference an endpoint's `since_version` with the CHANGELOG. |
| `cut_music_video` | 1 | `prompt`, `music_prompt`, `duration_s`, `num_clips?=5`, `quality?="preview"`, `subject_anchor_uri?`, `shot_list?`, `music_lyrics?`, `music_style?`, `chain_mode?="seamless-segment"`, `session_id?` | The orchestrator. Music → N a2v clips with chained segments → composition → export. Blocking; checkpointed. |
| `resume_music_video` | 1 | `session_id: str` | Resume a `cut_music_video` session that failed or was interrupted. Skips completed steps. |
| `list_sessions` | 1 | — | List locally cached MV sessions. Auto-prunes entries >7 days old. |

Each tool maps 1:1 to one or more endpoints in [docs/API.md](API.md):

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
| `cut_music_video` | composes `/v2/music` → `/v2/audio-to-video` × N → `/v2/video/extract-segment` × (N-1) → `/v2/history` → `/v2/compositions` → `/v2/compositions/{id}/export` |

Each tool's full inputSchema and result projection live in [`mcp/_phase2/01-tools-trace.md`](https://github.com/tacos8me/noodle-portal/blob/main/mcp/_phase2/01-tools-trace.md). The orchestrator's state machine and quality-tier mapping live in [`mcp/_phase2/02-orchestrator-design.md`](https://github.com/tacos8me/noodle-portal/blob/main/mcp/_phase2/02-orchestrator-design.md).

---

## 7. Roadmap

v0.3 candidates — each one is gated on something either upstream (taco-backend) or in the MCP itself:

| # | Feature | Blocker |
|---|---|---|
| **B1** | `get_audio_grid(audio_uri)` — BPM + downbeat + onset extraction so `cut_music_video` aligns clip cuts to the beat instead of uniform `duration_s/num_clips`. | Backend: needs a new `POST /v2/audio/analyze` returning `{bpm, downbeats, peaks}`. ~100 LOC of librosa. |
| **B4** | Lyric timestamps: `lyrics_timestamps: [{line, start_s, end_s}]` on music job results so a shot can pin to lyric line N. | Backend: ACE Step already knows where vocals land — needs to surface them in the music job result payload. |
| **B7** | `share_to_gallery(comp_id, label)` — one-click "publish a draft to the portal gallery" so the user can review without leaving the LLM session. | MCP only: thin wrapper over `POST /v1/approved-images`. Trivial; deferred to keep v0.2 surface tight. |
| — | SSE `wait_for_job` variant — instead of polling, open `/v2/jobs/{id}/stream` and consume server-sent events. ~10× lower poll volume on long jobs. | MCP only: `httpx.stream()` complexity. Not user-facing important. |
| — | Parallel-independent clip mode — when `chain_mode="independent"`, `cut_music_video` could submit all clips at once via `asyncio.gather` and finish in `max(clip_t)` not `sum(clip_t)`. | Backend: needs the server's queue + remote pool to handle 5-20 concurrent video jobs. v1.13 gave us 14 concurrent workers, so this is unblocked — just engineering. |

If you have a use case for a v0.3 feature, file an issue at github.com/tacos8me/noodle-portal.

---

## 8. Troubleshooting

**`claude mcp list` shows "Failed to connect" for `noodlefinger`.**
Check that `uvx` is in your PATH. `which uvx` should print a path. If not, install [astral-sh/uv](https://docs.astral.sh/uv/getting-started/installation/) — the MCP launches via `uvx` from a git URL on every start. The first launch downloads the wheel (~5 s); subsequent launches are cached and start in <1 s.

**Tier-1 tools missing — `mcp__noodlefinger__submit_job` returns "tool not found".**
Set `NOODLEFINGER_API_KEY` in the env config of your client's MCP entry and restart the client. The 14 tier-1 tools are conditionally registered at `list_tools()` time; if the env var is unset at server start, they never appear in the tool list. Run `mcp__noodlefinger__list_groups` (tier-0) to confirm the server is otherwise healthy.

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
