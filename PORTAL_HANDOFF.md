# Handoff — what's real vs. what's a shell

**Status as of 2026-04-26 / commit `0d23f69`:** docs site, MCP server, auth gate, and hosting are real and live. **Every page beyond `/docs` is currently a visual shell over mock JSON in `src/data-real.jsx`.** This file is the punch list to actually finish it.

## Live + working ✓

| Surface | Status |
|---|---|
| `https://portal.noodlefinger.io/docs` (77 endpoints, 13 flows) | real, accurate, regenerated from `data/*.json` |
| `https://docs.noodlefinger.io` (alias → portal/docs) | real |
| `https://mcp.noodlefinger.io` (install landing) | real |
| `noodlefinger-mcp` server (`uvx --from git+...`) | real, bundles catalog |
| WorkOS AuthKit login → cookie session | real, gating works |
| Caddy + auth-proxy on Vultr `64.176.217.133` ($6/mo) | real |
| Visual design system (paper grain, JetBrains Mono, light/dark, tweaks panel) | real, matches reference verbatim |

## Shell pages backed by mock data ✗

Every page below renders correctly but the data is static JSON in `src/data-real.jsx` (auto-generated from the build script). To make these real, build a thin BFF (FastAPI alongside `auth-proxy.py` on the VPS) that hits the right backing system per page.

### Dashboard (`home`)
- "Generations today", queue depth, GPU util, recent jobs.
- **Backing data**: `taco-backend` `/v1/system/gpu`, `/v2/history?limit=10`, `/v1/system/workers`. Already exposed; just need a fetch + render.
- **BFF endpoint to add**: `GET /api/dashboard/summary` → fanout, return aggregated.
- ETA: half a day.

### API Keys (`keys`)
- List/rotate/revoke real bearer keys. Currently shows 4 mock entries.
- **Critical gap**: `taco-backend` has NO key-management API today (`.api_keys` is a flat file). Need to either:
  - Add `/v1/keys` CRUD to `taco-backend` (writes the flat file atomically + reloads in-memory set), OR
  - Move to a proper auth model: portal owns the key store (DB-backed), taco-backend reads via shared secret / JWT.
- **Recommendation**: option B — portal SQLite or Postgres holds keys + per-key metadata (label, env, perms, last_used, created_by); taco-backend's `_extract_api_key` middleware queries the portal's `/internal/keys/verify` endpoint with a tiny LRU cache.
- ETA: 2-3 days.

### Usage (`usage`)
- Spend, generations, token costs.
- **Backing data**: `taco-backend/history.db` (per-key SQLite, already keyed by `api_key_hash`). Needs a "billing layer" mapping job_type → cost.
- **BFF endpoints**: `GET /api/usage/summary?period=...`, `GET /api/usage/by_model?...`, `GET /api/usage/breakdown_by_key?...`.
- **Optional**: Stripe integration for real billing. Otherwise show a usage-tracking-only display.
- ETA: 2-3 days for usage-only; +1 week with Stripe.

### Team (`team`)
- WorkOS handles user identity. Team membership currently mock.
- **Backing data**: WorkOS Organizations + Memberships APIs. Map WorkOS users → portal team table.
- **BFF endpoints**: `GET /api/team/members`, `POST /api/team/invite`, `PATCH /api/team/members/{id}/role`.
- ETA: 1-2 days.

### Audit log (`audit`)
- Currently 9 mock rows.
- **Backing data**: needs an audit table. Every write op (key rotate, member invite, billing change, webhook create) writes a row.
- **Decision**: SQLite at `/var/portal/audit.db` is fine for v0. WAL mode. Append-only, no edit.
- ETA: 1 day to plumb into existing actions; can be done lazily as new write-ops land.

### Webhooks (`webhooks`)
- Doesn't exist on `taco-backend` yet. The portal's mock UI shows what's intended.
- **Required backend changes**: subscription store, signing secret per webhook, retry queue (Redis or SQLite-based), `requests.post` from a worker, signature verification helper for clients.
- **Reference shape**: Stripe-style. Events: `job.queued`, `job.processing`, `job.completed`, `job.failed`, `job.cancelled`, `batch.completed`, `key.rotated`.
- ETA: 1 week. Largest single gap.

### Models (`models`)
- Model list is real (8 entries from `taco-backend/docs/models.md`); per-model usage and toggle states are mock.
- **Quick win**: wire each model card to real `taco-backend` health/load via `/v1/system/gpu` + `/v1/loras` membership. Static for now is fine.
- ETA: half a day.

### Account (`account`)
- Tweaks panel works. Profile/email/avatar/password change all mock.
- **Backing data**: WorkOS User APIs. Most flows are "redirect to WorkOS hosted page" — minimal portal code needed.
- ETA: half a day.

### Support (`support`)
- Static articles. KB content needs writing. Optional Linear/Zendesk ticket integration.
- ETA: half a day for static. KB writing is its own thing.

## Required infrastructure additions

1. **BFF service**: `/var/portal/bff/` — FastAPI app on `localhost:8002`, mounted in Caddy under `/api/*`. Talks to: `taco-backend`, WorkOS, the portal's own SQLite/Postgres, Stripe (optional).
2. **Portal DB**: SQLite at `/var/portal/portal.db` (WAL mode). Tables: `keys`, `audit`, `webhooks`, `webhook_deliveries`, `team_invites`. Migrate later if Postgres-needed.
3. **Worker**: minimal `dramatiq` or APScheduler for webhook delivery + retry. Same VPS for now.
4. **Cron**: nightly `vacuum` + WAL checkpoint on `portal.db`; nightly snapshot to `/var/portal/backups/` (rotate 7 days).
5. **Monitoring**: Caddy access logs already JSON. Pipe to Loki or just `journalctl` for v0.

## Suggested order (smallest-risk first)

1. **BFF skeleton** — FastAPI app, healthcheck, WorkOS session middleware, `localhost:8002`.
2. **Dashboard real metrics** — fanout to `taco-backend` health endpoints. Easiest, validates the BFF wiring.
3. **API keys CRUD** (option B above) — biggest user-visible win, unblocks every other page that needs real bearer auth.
4. **Audit log** — append-only as new write ops land, no big-bang.
5. **Usage** — read from `history.db`. No backend changes.
6. **Team** — WorkOS API integration.
7. **Account** — mostly redirects to WorkOS hosted pages.
8. **Models** — real load/usage from `taco-backend`.
9. **Webhooks** — full system.
10. **Stripe + billing** — last; blocks on key CRUD + usage.

## How to start the next session

```bash
cd /mnt/nvme-1/projects/noodlefinger-portal
claude
```

Tell the new session:
> Read HANDOFF.md. Start the BFF skeleton — new `bff/` directory, FastAPI on `localhost:8002`, WorkOS session middleware that reads the same `nf_session` cookie auth-proxy issues. Mount it in `deploy/Caddyfile` under `/api/*`. After the skeleton is up, do "Dashboard real metrics" first — that validates the wiring end-to-end.

Or: spawn another team via TeamCreate. Repo + handoff doc are the persistent context; nothing in this conversation's memory is required.

## Cost / billing reality

- Vultr instance: $6/mo (running now).
- WorkOS: free up to 1M MAU.
- Cloudflare: free.
- Stripe (when added): % per transaction.
- Total infra: $6/mo for v0; <$30/mo even with light traffic.

## Things deliberately NOT done

- **Test suite for portal pages** — visual app, no logic tests. JSDOM smoke is the bar.
- **CI/CD** — `bash deploy/sync.sh` from a checkout is the deploy. Add GitHub Actions when there's a second deploy target.
- **SDK packages** — no `noodlefinger` npm/PyPI package yet. Just curl/fetch examples in the docs. SDKs are weeks of work each.
- **End-to-end browser tests** — Playwright wasn't set up. Add when there's flow critical enough to break visibly.
