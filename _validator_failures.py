"""In-memory ring buffer for recent validator dispatch failures.

Wired into ``server._dispatch_validator``'s exception path so the
operator console can surface a "last hour's failures" panel without
adding a SQLite table. Process-local; resets on restart (acceptable
per the plan — persistent storage is a P2 item).
"""
from __future__ import annotations

import threading
import time
from collections import deque

_MAXLEN = 1000
_TRACE_TRUNC = 500

_buffer: deque[dict] = deque(maxlen=_MAXLEN)
_lock = threading.Lock()


def record(job_id: str, tier: str, error: str) -> None:
    """Append a failure entry to the ring buffer.

    ``tier`` is one of ``"tier1"`` / ``"tier2"`` / ``"tier3"`` /
    ``"composite"`` / ``"dispatch"`` (catch-all). ``error`` is
    truncated to 500 chars to keep the buffer footprint bounded.
    """
    entry = {
        "job_id": job_id,
        "tier": tier,
        "error": (error or "")[:_TRACE_TRUNC],
        "ts": time.time(),
    }
    with _lock:
        _buffer.append(entry)


def recent(window_seconds: float) -> list[dict]:
    """Return failures within the last ``window_seconds``.

    Newest first. Returns a copy so callers can iterate without lock
    contention.
    """
    cutoff = time.time() - max(0.0, window_seconds)
    with _lock:
        snap = [e for e in _buffer if e["ts"] >= cutoff]
    snap.reverse()
    return snap


def clear() -> None:
    """Drop all entries — test-only convenience."""
    with _lock:
        _buffer.clear()
