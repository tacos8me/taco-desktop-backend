# taco-backend

LTX-compatible inference server for taco-desktop.

## Structure
- `server.py` — FastAPI app, all HTTP endpoints
- `pipeline_manager.py` — Dual-GPU pipeline loading and request dispatch
- `config.py` — Paths, model mapping, resolution tables
- `upload_store.py` — UUID file storage for uploads

## Key commands
- Run: `uv run uvicorn server:app --host 0.0.0.0 --port 8080`
- Test: `uv run pytest tests/ -v`

## Conventions
- Use ltx-pipelines classes directly (DistilledPipeline, TI2VidTwoStagesPipeline, etc.)
- All generation runs under `@torch.inference_mode()`
- Return raw MP4 bytes with `Content-Type: video/mp4`
