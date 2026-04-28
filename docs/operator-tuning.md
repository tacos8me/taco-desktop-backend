# Operator tuning — rate limits, concurrency, fd ceilings, export quality

> Server version: v1.16.2 (2026-04-28).
> Audience: backend operator / on-call. Frontend / SDK callers should look at [API.md](./API.md) instead.

This file documents the runtime-tunable knobs introduced (or rebalanced) in **v1.16.1** after a real-world MV submission pattern (28 sequential a2v jobs at 1.5 s pacing) tripped `per_key_queue_full` on 24/28 first-pass submissions and surfaced `Connection reset by peer` on concurrent client polls.

## What's tuned in v1.16.1

Three layers were touched:

1. **Application layer** — per-bearer + global queue caps in `config.py`.
2. **HTTP layer** — uvicorn `--limit-concurrency` + `--backlog` flags in `run.sh`.
3. **System layer** — `LimitNOFILE` in the systemd user unit.

All three are environment-overridable; nothing is baked into a binary. To roll back to v1.15.x behavior, set the env vars in `.env` to the pre-v1.16.1 numbers and restart the unit.

## Per-key job queue caps (application layer)

`config.py` exposes four ceilings. Each accepts an env-var override of the same name; missing or unparsable values fall back to the default.

| Knob | v1.15.x default | v1.16.1 default | What it caps |
|---|---|---|---|
| `PER_KEY_QUEUE_CAP` | 3 | **15** | In-flight v2 video jobs per bearer (text-to-video, image-to-video, audio-to-video, retake, video-outpaint, video-hdr). Returns `429 per_key_queue_full` + `Retry-After: 30` when exceeded. |
| `PER_KEY_MUSIC_CAP` | 2 | **5** | In-flight music submissions per bearer (`/v2/music`). |
| `PER_KEY_BATCH_CAP` | 2 | **5** | In-flight batch submissions per bearer (`/v2/batch`). |
| `MAX_QUEUE_DEPTH` | 10 | **30** | Global queue ceiling across all bearers. Returns `429 queue_full`. |

`PER_KEY_QUEUE_CAP` was the proximate cause of the v1.16.1 regression — 28 sequential video submissions from one bearer overflowed a 3-slot per-key cap before the queue worker could drain even one. The new ceiling fits a 28-job batch + retry headroom while still leaving the global cap (30) at 2× the per-key cap, so a single bearer can't fully starve another.

### Override pattern

In `.env` (or any env source loaded by `run.sh`):

```
PER_KEY_QUEUE_CAP=8
PER_KEY_MUSIC_CAP=3
PER_KEY_BATCH_CAP=3
MAX_QUEUE_DEPTH=20
```

Then `systemctl --user restart taco-backend`. No code edit required.

## HTTP layer (uvicorn)

`run.sh` invokes uvicorn with explicit flags as of v1.16.1:

```
exec uv run --no-sync uvicorn server:app --host 0.0.0.0 --port 8090 --no-access-log \
    --limit-concurrency 200 --backlog 4096
```

| Flag | Why |
|---|---|
| `--limit-concurrency 200` | Caps in-flight ASGI request handlers. Default uvicorn uses `limit_concurrency=None` (unbounded) — under 28+ concurrent client polls per video job, the unbounded path lets the kernel SYN backlog overflow before ASGI ever sees the request, and clients get `Connection reset by peer` instead of a clean `503`. With the cap, requests beyond 200 in-flight return clean 503s with the standard envelope. |
| `--backlog 4096` | Doubles the kernel TCP listen queue from the uvicorn default of 2048. Shipping with the higher value gives more headroom for short bursts of poll-heavy clients (the same 28-job MV submission with 240 GET polls per job is ~6,720 polls in flight at peak). |

Tunable: bump `--limit-concurrency` higher if your operator workload includes many slow concurrent SSE streams (each holds a handler open). Lower it if you want more aggressive backpressure.

## System layer (systemd)

The taco-backend user unit at `~/.config/systemd/user/taco-backend.service` carries:

```
[Service]
LimitNOFILE=16384
```

Default systemd inherits the user-default `nofile=1024`, which was *exactly* the pre-v1.16.1 ceiling. A single in-flight MV submission opens, at peak:

- ~30 concurrent client SSE streams + result-poll connections,
- ~10 outbound httpx connections to the LTX local + Modal + RunPod sidecars,
- ~3 outbound httpx connections to ACE / JoyAI or ERNIE / madmom sidecars,
- 1 SQLite-WAL writer + N reader fds for `history.db`,
- ~50 transient fds for upload-store reads, thumbnail extraction, and ffmpeg pipes.

Crossed at ~150 fds in steady state during a busy 14-worker turbo session. 16,384 leaves plenty of headroom for accidental fd leaks; the system-wide `nofile-max` is ~1M on a stock Ubuntu 24.04 box so this is not load-bearing on system limits.

## Reversibility

To run more conservatively in a constrained environment (a small VPS, a noisy multi-tenant box):

1. Lower the per-key caps in `.env`:
   ```
   PER_KEY_QUEUE_CAP=4
   PER_KEY_MUSIC_CAP=2
   PER_KEY_BATCH_CAP=2
   MAX_QUEUE_DEPTH=12
   ```
2. Optionally lower the uvicorn limits in `run.sh` (edit `--limit-concurrency` and `--backlog`).
3. Optionally lower `LimitNOFILE` in the systemd unit and `systemctl --user daemon-reload && systemctl --user restart taco-backend`.

Caps are read at process start; restart is required after any of the above.

## Validation — confirm the caps are live

After a config change + restart, verify the live values from outside the running process:

```bash
# (1) Per-key + global caps are loaded into config:
python3 -c "import config; print(config.PER_KEY_QUEUE_CAP, config.MAX_QUEUE_DEPTH)"
# Expect: 15 30   (or your override)

# (2) NOFILE limit is in effect on the live process:
cat /proc/$(systemctl --user show -p MainPID --value taco-backend)/limits | grep "Max open files"
# Expect: Max open files            16384                16384                files
```

If `(1)` reports the v1.15.x defaults `3 10`, the unit was likely started without your `.env` — confirm the file path and that systemd is sourcing it (`EnvironmentFile=` in the `[Service]` block).

If `(2)` reports `1024`, the unit didn't pick up `LimitNOFILE` — `systemctl --user daemon-reload` was probably skipped after the unit edit.

## Cross-references

- Application-layer 429 envelopes are documented in `docs/API.md` → "Queue / rate limit" table.
- The v1.16.0 `/v1/music/analyze` endpoint (which `madmom_client` proxies) is documented in `docs/API.md`.
- The `cut_music_video` MCP orchestrator (which is the most common driver of the 28-job submission pattern) lives in noodlefinger-mcp v0.4.2+; per-shot audio slicing is described in its `flows.json`.

---

## Export quality (v1.16.2)

`POST /v2/compositions/{comp_id}/export` ships a libx264-based default encoder stack as of v1.16.2. Pre-v1.16.2 the export pipeline hardcoded `-c:v libopenh264` with NO CRF / bitrate flags — libopenh264 defaults to ~4-8 Mbps which produced visible blocking on 1080p+ output.

The fix switches the default to:

| Knob | v1.16.2 default |
|---|---|
| Video encoder | `libx264` |
| CRF | `18` (libx264) / `22` (libx265) |
| Preset | `medium` |
| Profile | `high` |
| Pixel format | `yuv420p` (max compat) |
| Audio bitrate | `256k` (was 192k) |

CRF 18 + high profile is the standard "visually transparent" operating point for H.264 — perceptually lossless at typical viewing distances. `yuv420p` maximizes player compatibility (mobile / iOS QuickTime / Chrome all decode it natively).

### Per-export overrides

There are NO env vars for export quality — every knob is per-request. Full schema in `docs/API.md` → `POST /v2/compositions/{comp_id}/export`:

- `output_encoder` ∈ `{libx264, libx265, libopenh264}`
- `output_crf` ∈ `[0, 51]` (lower = higher quality)
- `output_preset` ∈ x264 preset names (`ultrafast` … `placebo`)
- `output_profile` ∈ `{baseline, main, high, high10, high422, high444}`
- `output_video_bitrate` (string `\d+[kMG]?`, e.g. `"12M"`)
- `output_audio_bitrate` (string `\d+[kMG]?`, e.g. `"320k"`)

Setting `output_video_bitrate` on libx264/libx265 switches the encoder from CRF mode to 1-pass ABR with `-maxrate` and `-bufsize 24M` — useful when you have a hard size budget and don't care about per-clip quality variance.

Malformed values return `422` BEFORE the job is enqueued.

### `TACO_FFMPEG_BIN` env var (only operator knob)

This is the ONLY operator-side knob for export quality. Sets the ffmpeg binary used for export. Defaults to autodetect.

The autodetect order is:

1. `$TACO_FFMPEG_BIN` if set.
2. `ffmpeg` from `$PATH` (whatever venv / conda is active).
3. `/usr/bin/ffmpeg` (Ubuntu system ffmpeg).

The exporter probes each candidate with `-encoders` and picks the first one that supports the requested encoder. **Note**: conda-installed ffmpeg in this box's miniconda env was built `--disable-gpl` and ships WITHOUT libx264/libx265 — only libopenh264. The Ubuntu system ffmpeg at `/usr/bin/ffmpeg` has libx264 from the `libx264-*` package, so autodetect picks it for the libx264 path even though it's not first in `$PATH`.

If neither has libx264/libx265, the exporter logs a WARN and falls back to libopenh264 (lower quality at default bitrate). To install:

```bash
sudo apt install libx264-164  # or libx265-199
# Both are dependencies of Ubuntu's `ffmpeg` meta-package.
```

### Validation

After deployment, sanity-check the live encoder via a small export:

```bash
COMP_ID=...
KEY=$(grep -v '^#' .api_keys | grep -v '^$' | head -1)

# Default export.
JOB=$(curl -sS -X POST "http://localhost:8090/v2/compositions/$COMP_ID/export" \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d '{}' | jq -r .id)

# Poll until done, then download:
curl -sS -H "Authorization: Bearer $KEY" \
  "http://localhost:8090/v2/jobs/$JOB/result" -o /tmp/result.mp4

# Confirm libx264 was used and the bitrate is higher than v1.16.1:
ffprobe -v error -show_entries stream=codec_name,bit_rate /tmp/result.mp4
```
