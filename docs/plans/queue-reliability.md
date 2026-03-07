# Async Job Queue -- Reliability & Edge Cases

Risk analysis for adding an in-memory async job queue to taco-backend.
Context: single-process FastAPI, 1-3 LAN users, two GPUs serialized via `asyncio.Lock`.

---

## 1. Server Restart Mid-Job

**Risk:** In-memory queue means all pending/running jobs are lost on crash or restart.

**Assessment:** Acceptable for this deployment.
- 1-3 LAN users, not a production SaaS. Users can resubmit.
- Persistence (Redis, SQLite) adds operational complexity disproportionate to the user count.
- GPU model reload takes ~60s anyway, so any in-flight job would fail regardless.

**Mitigation:**
- On startup, the job store should be empty. No stale "running" jobs to confuse clients.
- Client should treat connection failure during poll as "job lost, resubmit."
- Document this behavior in the API response (e.g., a note that jobs do not survive restarts).

**Verdict:** Accept the tradeoff. No persistence needed.

---

## 2. Result Cleanup (TTL & Purge)

**Risk:** Completed job results (MP4 video bytes, PNG image bytes) accumulate in memory. A single video can be 50-200MB. Without cleanup, memory grows unbounded.

**Assessment:** Critical. Must have a cleanup strategy.

**Recommended approach:**
- **TTL of 10 minutes** after job completion. Clients have a generous window to fetch results.
- **Eager delete on fetch:** Once a client retrieves the result via `GET /v1/jobs/{id}/result`, mark it for cleanup (or delete immediately if single-consumer is assumed).
- **Background purge task:** An `asyncio.Task` running on a 60-second interval that sweeps expired results. Simple `dict` iteration, no external deps.
- **Memory cap:** If total result memory exceeds a threshold (e.g., 2GB), purge oldest completed jobs first, even if TTL hasn't expired. This prevents OOM from rapid submissions.
- **Failed jobs:** Keep error message (small) for same TTL, then purge.

**Note:** Upload files (`upload_store.py`) also have no cleanup today. Consider adding TTL to uploads in a separate pass.

---

## 3. Client Disconnect

**Risk:** Client submits a job, then disconnects (closes browser, navigates away, network issue). Should the job continue?

**Assessment:** Yes, jobs should always run to completion.

**Rationale:**
- Inference is GPU-bound and non-interruptible mid-generation (no clean cancellation point in ltx-pipelines or diffusers).
- The GPU is occupied regardless. Aborting mid-inference doesn't free it faster -- the CUDA kernels are already dispatched.
- The user likely still wants the result and will reconnect.

**Design implication:**
- The POST endpoint returns a job ID immediately. The job runs independently of the HTTP connection.
- No WebSocket/SSE dependency for job execution. Polling is the primary interface.
- If the client never fetches the result, TTL cleanup handles it (Section 2).

---

## 4. Queue Depth Limits

**Risk:** Without limits, a client could flood the queue with hundreds of jobs, exhausting memory before any complete.

**Assessment:** Needed, but can be simple.

**Recommended approach:**
- **Max queue depth: 10 pending jobs** (across all clients). With inference times of 16-63s per job, 10 jobs = 3-10 minutes of backlog. Beyond that, the user is unlikely to wait.
- **Return 429 Too Many Requests** when the queue is full, with a `Retry-After` header estimating wait time.
- **Per-endpoint limits are unnecessary** at this scale. A single global counter suffices.
- **No priority queue needed.** FIFO is fair and simple for 1-3 users.

**Alternative considered:** Per-API-key limits. Adds complexity without clear benefit at 1-3 users. Revisit if user count grows.

---

## 5. Duplicate Submission (Idempotency)

**Risk:** Client submits the same request twice (double-click, retry on timeout). Two identical jobs run, wasting GPU time.

**Assessment:** Not worth implementing idempotency keys for this deployment.

**Rationale:**
- Idempotency requires either client-generated keys (adds client complexity) or server-side request hashing (unreliable with seeds/timestamps).
- With the queue, the client gets a job ID back quickly (~instant). Double-click protection is better handled client-side.
- Even if duplicates run, the cost is GPU time, not data corruption. Acceptable at 1-3 users.
- Queue depth limit (Section 4) naturally bounds the damage from accidental spam.

**Verdict:** Skip idempotency keys. Rely on client-side dedup and queue depth limits.

---

## 6. Job Scoping / Per-API-Key Isolation

**Risk:** One user could see or cancel another user's jobs.

**Assessment:** Low risk given the deployment, but worth basic isolation.

**Current auth model:** API keys loaded from `.api_keys` file. Keys are shared secrets, not user identities. Multiple users might share one key.

**Recommended approach:**
- **Associate jobs with the API key that created them.** A job listing endpoint (`GET /v1/jobs`) should only return jobs belonging to the caller's key.
- **Job result retrieval** should also check key ownership. Even though job IDs are unguessable (Section 10), defense in depth is cheap.
- **No admin endpoint needed** for now. If needed later, add a separate admin key with elevated access.

**Edge case:** If `API_KEYS` is empty (auth disabled), all jobs are "unscoped." This is fine -- no auth means no isolation expectation.

---

## 7. Cloudflare Timeout Specifics

**Risk:** The entire motivation for the queue. Need to understand exactly what CF enforces.

**Findings:**
- **Cloudflare Tunnel (cloudflared) proxy timeout: 100 seconds** by default for HTTP responses. This is the `proxy-connection-timeout` / origin response timeout.
- **This is not configurable** on the free/standard tier. Enterprise can adjust it.
- **SSE / chunked transfer:** Cloudflare does support streaming responses. Once the first byte is sent, the 100s timeout resets on each chunk. However:
  - SSE requires the server to send periodic keepalive data.
  - The inference pipeline doesn't produce intermediate output that maps cleanly to progress events.
  - Sending synthetic heartbeats (empty SSE comments) every ~30s would keep the connection alive, but adds complexity to every endpoint.
- **WebSocket:** CF tunnels support WebSocket upgrade. Same keepalive considerations apply.

**Recommended approach for the queue:**
- **Do not rely on SSE/chunked to work around the timeout.** The queue approach is cleaner and more robust.
- Submit returns job ID in <1s (well within timeout).
- Poll returns status in <1s (well within timeout).
- Result fetch returns binary data. For large videos (>100MB), this might itself take time over slow connections, but CF handles streaming responses fine once headers are sent.

**Fallback consideration:** If the queue is implemented, the synchronous endpoints could optionally remain for direct LAN access (bypassing CF). This avoids breaking existing clients during migration (Section 8).

---

## 8. Backward Compatibility & Migration

**Risk:** Existing taco-desktop client expects synchronous POST -> binary response. Switching to async queue breaks the client.

**Assessment:** Needs a migration period.

**Recommended approach:**
- **Phase 1: Add async endpoints alongside sync ones.** New endpoints:
  - `POST /v1/jobs/text-to-video` -> returns `{job_id, status}` (202 Accepted)
  - `GET /v1/jobs/{id}` -> returns job status
  - `GET /v1/jobs/{id}/result` -> returns binary result
  - Original `POST /v1/text-to-video` continues to work (sync, will timeout via CF).
- **Phase 2: Update taco-desktop** to use async endpoints.
- **Phase 3: Deprecate sync endpoints** (or keep them for direct LAN use where CF timeout doesn't apply).

**Alternative:** A single set of endpoints that detect the `Prefer: respond-async` header. If present, return 202 + job ID. If absent, behave synchronously. This is more elegant but harder to test and debug.

**Recommendation:** Separate endpoint paths (Phase 1 approach). Simpler, explicit, no header magic.

---

## 9. Monitoring / Health Endpoint

**Risk:** Without queue visibility, operators can't diagnose why a job is slow or stuck.

**Assessment:** Extend `/health` with queue stats. Low effort, high value.

**Recommended additions to `/health` response:**
```json
{
  "status": "ok",
  "ltx": "ready",
  "flux": "ready",
  "chat": "ready",
  "queue": {
    "pending": 3,
    "running": 1,
    "completed": 2,
    "failed": 0,
    "oldest_pending_age_s": 45.2
  }
}
```

**Additional considerations:**
- `oldest_pending_age_s` helps detect stalls. If this grows beyond expected inference time, something is stuck.
- **No separate metrics endpoint needed** (no Prometheus, no Grafana) for 1-3 users. `/health` JSON is sufficient.
- **Logging:** Log job lifecycle events at INFO level: created, started, completed/failed, result fetched, expired. Include job ID and elapsed time.

---

## 10. Security

### Unguessable Job IDs

- Use `secrets.token_urlsafe(16)` for job IDs (22 characters, 128 bits of entropy). This matches the security level of the upload IDs (uuid4 = 122 bits).
- Do NOT use sequential integers or timestamps. Job IDs are exposed in URLs and could be enumerated.
- Even with unguessable IDs, enforce API key ownership checks (Section 6) as defense in depth.

### Rate Limiting

- **Queue depth limit (Section 4) acts as a natural rate limiter.** 10 pending jobs max.
- **No per-second rate limiting needed** at 1-3 users. The inference lock already serializes everything.
- If abuse becomes a concern, add per-key rate limiting (e.g., max 5 submissions per minute per key). But this is premature for the current deployment.

### Result Access

- Results should only be fetchable by the API key that submitted the job.
- Results should not be cached by intermediate proxies. Set `Cache-Control: no-store` on result responses.
- After TTL expiration, results are gone. No forensic recovery.

### Input Validation

- Existing Pydantic models already validate inputs (prompt length, resolution, duration bounds).
- The queue layer should not bypass these validations. Validate at submission time, not at execution time. Fail fast with 422 rather than queuing an invalid job.

---

## Summary of Recommendations

| Concern | Decision | Complexity |
|---|---|---|
| Persistence | None (in-memory only) | None |
| Result TTL | 10 min, eager delete on fetch | Low |
| Client disconnect | Jobs run to completion | Built-in |
| Queue depth | Max 10 pending, 429 when full | Low |
| Idempotency | Skip | None |
| Job scoping | Per-API-key filtering | Low |
| CF timeout | Queue solves it; no SSE needed | N/A |
| Migration | Parallel async endpoints, then deprecate sync | Medium |
| Monitoring | Extend /health with queue stats | Low |
| Job IDs | secrets.token_urlsafe(16) | Low |
| Rate limiting | Queue depth limit is sufficient | None |

Overall complexity: Low-Medium. The queue itself is straightforward. The main work is the new endpoint set and client migration.
