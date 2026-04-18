"""taco-dashboard — LAN-only admin dashboard + API proxy.

Runs on a separate port (8099, LAN-bound) from the public taco-backend (8090).
Serves `dashboard.html` and transparently proxies every other path to the
backend on localhost, forwarding the caller's Authorization header.

Why split it out: the previous `/dashboard` endpoint and the
`/v1/system/gpu` no-auth whitelist exposed ops telemetry (GPU model, memory,
load, gen_config) + the management SPA to anyone who could reach the public
API host (`api.noodlefinger.io`) over the internet — see SEC P1-1 / P1-2.
Moving them to a LAN-only port (no Cloudflare Tunnel ingress rule → not
internet-reachable) removes that exposure without refactoring the dashboard's
fetch paths.

The proxy forwards the caller's Bearer token unchanged, so authed API access
still flows through the normal taco-backend auth middleware. The dashboard
HTML stays byte-identical (uses relative `/v1/...` URLs that land here and
get transparently forwarded to `http://127.0.0.1:8090/v1/...`).

Run via: `bash run-dashboard.sh` (or the matching systemd user unit).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DASHBOARD_HOST = os.environ.get("DASHBOARD_HOST", "192.168.1.80")
DASHBOARD_PORT = int(os.environ.get("DASHBOARD_PORT", "8099"))
BACKEND_URL = os.environ.get("TACO_BACKEND_URL", "http://127.0.0.1:8090").rstrip("/")

DASHBOARD_HTML = Path(__file__).parent / "dashboard.html"

app = FastAPI(
    title="taco-dashboard (admin)",
    description="LAN-only admin UI + API proxy for taco-backend",
    openapi_url=None,  # don't publish a spec; this is ops-only
    docs_url=None,
    redoc_url=None,
)

# Long timeout — the dashboard may trigger turbo toggles, pool scaling, and
# other ops that block for tens of seconds inside the backend. Share a single
# client to reuse connections.
_http = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=5.0))


@app.on_event("shutdown")
async def _close_http() -> None:
    await _http.aclose()


@app.get("/dashboard", include_in_schema=False)
@app.get("/", include_in_schema=False)
async def serve_dashboard() -> HTMLResponse:
    if not DASHBOARD_HTML.exists():
        return HTMLResponse(content="dashboard.html missing", status_code=500)
    return HTMLResponse(
        content=DASHBOARD_HTML.read_text(encoding="utf-8"),
        media_type="text/html",
    )


# Forward every other method+path to the backend, carrying headers + body.
# Header names that httpx manages itself (host, content-length) are stripped.
_HOP_BY_HOP = frozenset({
    "host", "content-length", "transfer-encoding", "connection",
    "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "upgrade",
})


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
    include_in_schema=False,
)
async def proxy(path: str, request: Request) -> Response:
    url = f"{BACKEND_URL}/{path}"
    query = request.url.query
    if query:
        url = f"{url}?{query}"

    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in _HOP_BY_HOP
    }

    body = await request.body()

    try:
        upstream = await _http.request(
            request.method, url, headers=headers, content=body,
        )
    except httpx.TimeoutException:
        return Response(status_code=504, content=b'{"error":"backend_timeout"}', media_type="application/json")
    except httpx.ConnectError:
        return Response(status_code=502, content=b'{"error":"backend_unreachable"}', media_type="application/json")
    except httpx.HTTPError as exc:
        logger.exception("proxy error")
        return Response(status_code=502, content=f'{{"error":"proxy_error: {exc}"}}'.encode(), media_type="application/json")

    resp_headers = {
        k: v for k, v in upstream.headers.items()
        if k.lower() not in _HOP_BY_HOP
    }

    # Stream SSE (text/event-stream) and large bodies; otherwise return buffered.
    ctype = upstream.headers.get("content-type", "")
    if "text/event-stream" in ctype.lower():
        async def _stream():
            async for chunk in upstream.aiter_bytes():
                yield chunk
        return StreamingResponse(_stream(), status_code=upstream.status_code, headers=resp_headers, media_type=ctype)

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=resp_headers,
        media_type=ctype or None,
    )


def main() -> None:
    logger.info("taco-dashboard serving %s → proxy %s", f"{DASHBOARD_HOST}:{DASHBOARD_PORT}", BACKEND_URL)
    uvicorn.run(
        "dashboard_server:app",
        host=DASHBOARD_HOST,
        port=DASHBOARD_PORT,
        log_level="info",
        access_log=False,  # reduce noise; backend still logs real requests
    )


if __name__ == "__main__":
    main()
