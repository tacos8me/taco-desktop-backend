# Hybrid B: Dual-Denoiser with Shared Text Encoder

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable 2 concurrent video generation jobs by giving each GPU its own denoiser while sharing text encoding on GPU:0.

**Architecture:** GPU:0 holds text encoder + embeddings processor + audio encoder (shared) AND its own transformer + decoders (denoiser). GPU:1 holds only transformer + decoders (denoiser). Text encoding is serialized through GPU:0; denoising runs in parallel on both GPUs. Each GPU manages its own transformer swap state independently.

**Tech Stack:** Python 3.13, PyTorch (bf16, cu130), ltx-core, ltx-pipelines, FastAPI, asyncio

**VRAM Budget (validated):**
- GPU:0: ~75GB (encoder ~32GB + denoiser ~43GB, video_encoder shared not duplicated), ~21GB headroom
- GPU:1: ~48GB (denoiser only), 48GB headroom

**Concurrency model:** CUDA stream serialization on GPU:0 — text encoding (0.4s) may wait up to 1 denoising step (1.5s) when GPU:0 is also denoising. This is ~2-3% overhead, acceptable.

---

## Context for Implementer

### Current State
`split_model_manager.py` has `SplitModelManager` with:
- GPU:0 = encoder hub only (text encoder, VAE encoders) — **31GB, 0.9% utilization**
- GPU:1 = denoiser only (transformer, decoders) — **42GB, 90% utilization**
- Single `asyncio.Lock` — only 1 job at a time
- 4 generation flows: `_run_t2v`, `_run_i2v`, `_run_a2v`, `_run_retake`

### Target State
- GPU:0 = encoder hub + denoiser (both encode AND denoise)
- GPU:1 = denoiser only (receives encoded text from GPU:0 via PCIe)
- Per-worker locks — 2 concurrent jobs
- `_acquire_worker()` picks first free GPU
- Text encoding serialized via `_encode_lock` (GPU:0 text encoder is shared)

### Key Files
- `split_model_manager.py` — main refactor target
- `config.py` — no changes needed (already has `GPU_DEVICES`, `USE_SPLIT_GPU`)
- `server.py` — no changes needed (async API unchanged)

### Reference Code
- Pipeline helpers: `/mnt/nvme-1/repos/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py`
- Model ledger: `/mnt/nvme-1/repos/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/model_ledger.py`
- Types: `/mnt/nvme-1/repos/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/types.py`

### API Gotchas (from prior debugging)
- `denoise_video_only()` returns single `LatentState`, NOT tuple
- `decode_audio_from_file(path, device, max_duration=...)` — use keyword for max_duration
- `Audio` uses `sampling_rate` not `sample_rate`
- Audio VAE expects stereo (2 channels)
- `PipelineComponents(dtype, device)` must match denoiser device
- `torch.Generator(device=...)` must match denoiser device
- `TilingConfig.default()` required for VAE decode

---

## Task 1: Add DenoiserWorker Dataclass

**Files:**
- Modify: `split_model_manager.py:137-155` (replace SplitModelManager state with DenoiserWorker)

**Step 1: Add DenoiserWorker class after CachingModelLedger (around line 108)**

Add this dataclass that holds per-GPU denoiser state:

```python
from dataclasses import dataclass, field

@dataclass
class DenoiserWorker:
    """Per-GPU worker with its own transformer, decoders, and swap state."""
    device: torch.device
    ledger: CachingModelLedger
    components: PipelineComponents
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    transformer_state: str = ""
    # Direct refs for swap logic
    transformer: object = None
    cache: dict[str, object] = field(default_factory=dict)

    def ensure_transformer(self, state: str) -> None:
        """Swap transformer checkpoint on this worker's GPU."""
        if self.transformer_state == state:
            return

        logger.info("Worker %s: swapping transformer %s -> %s", self.device, self.transformer_state, state)
        if state == "dev":
            checkpoint, loras = config.DEV_CHECKPOINT, ()
        elif state == "distilled":
            checkpoint, loras = config.DISTILLED_CHECKPOINT, ()
        elif state == "dev_lora":
            distilled_lora = LoraPathStrengthAndSDOps(
                path=config.DISTILLED_LORA, strength=1.0,
                sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
            )
            checkpoint, loras = config.DEV_CHECKPOINT, (distilled_lora,)
        else:
            raise ValueError(f"Unknown transformer state: {state}")

        ledger = ModelLedger(
            dtype=torch.bfloat16, device=self.device,
            checkpoint_path=checkpoint, gemma_root_path=config.GEMMA_ROOT,
            spatial_upsampler_path=config.SPATIAL_UPSAMPLER, loras=loras,
        )
        new_transformer = ledger.transformer()

        old = self.transformer
        self.transformer = new_transformer
        self.cache["transformer"] = new_transformer
        self.transformer_state = state
        del old
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        logger.info("Worker %s: transformer now %s", self.device, state)
```

**Step 2: Verify import**

Run: `uv run --no-sync python -c "from split_model_manager import DenoiserWorker; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add split_model_manager.py
git commit -m "feat: add DenoiserWorker dataclass for per-GPU state"
```

---

## Task 2: Refactor load_all() for Dual Denoisers

**Files:**
- Modify: `split_model_manager.py` — `SplitModelManager.__init__` and `load_all()`

**Step 1: Rewrite __init__ and load_all**

Replace the current `__init__` and `load_all` with:

```python
class SplitModelManager:
    """Dual-denoiser manager: GPU:0 = shared encoder + denoiser, GPU:1 = denoiser.

    Text encoding serialized on GPU:0. Both GPUs denoise concurrently.
    """

    def __init__(self) -> None:
        self._workers: list[DenoiserWorker] = []
        self._encoder_device: torch.device | None = None
        self._encoder_ledger: CachingModelLedger | None = None
        # No explicit encode lock needed — CUDA serializes ops on the default
        # stream per-device. Text encoding (0.4s) may wait at most 1 denoising
        # step (~1.5s) when GPU:0 is concurrently denoising. Acceptable tradeoff.

    @property
    def is_ready(self) -> bool:
        return len(self._workers) > 0

    def load_all(self) -> None:
        devices = [torch.device(d) for d in config.GPU_DEVICES]

        # --- Shared encoder hub on GPU:0 ---
        self._encoder_device = devices[0]
        logger.info("Loading shared encoder hub on %s ...", devices[0])
        enc_ledger = ModelLedger(
            dtype=torch.bfloat16, device=devices[0],
            checkpoint_path=config.DEV_CHECKPOINT,
            gemma_root_path=config.GEMMA_ROOT,
            spatial_upsampler_path=config.SPATIAL_UPSAMPLER,
            loras=(),
        )
        enc_video_encoder = enc_ledger.video_encoder()  # kept for retake conditioning
        encoder_cache = {
            "text_encoder": enc_ledger.text_encoder(),
            "embeddings_processor": enc_ledger.gemma_embeddings_processor(),
            "video_encoder": enc_video_encoder,
            "audio_encoder": enc_ledger.audio_encoder(),
        }
        self._encoder_ledger = CachingModelLedger(devices[0], encoder_cache)

        # --- Denoiser worker on each GPU ---
        for device in devices:
            logger.info("Loading denoiser worker on %s ...", device)
            den_ledger = ModelLedger(
                dtype=torch.bfloat16, device=device,
                checkpoint_path=config.DEV_CHECKPOINT,
                gemma_root_path=config.GEMMA_ROOT,
                spatial_upsampler_path=config.SPATIAL_UPSAMPLER,
                loras=(),
            )
            transformer = den_ledger.transformer()
            # GPU:0 reuses encoder hub's video_encoder to avoid ~5GB duplication
            vid_enc = enc_video_encoder if device == devices[0] else den_ledger.video_encoder()
            cache = {
                "transformer": transformer,
                "video_encoder": vid_enc,
                "spatial_upsampler": den_ledger.spatial_upsampler(),
                "video_decoder": den_ledger.video_decoder(),
                "audio_decoder": den_ledger.audio_decoder(),
                "vocoder": den_ledger.vocoder(),
            }
            worker = DenoiserWorker(
                device=device,
                ledger=CachingModelLedger(device, cache),
                components=PipelineComponents(dtype=torch.bfloat16, device=device),
                transformer_state="dev",
                transformer=transformer,
                cache=cache,
            )
            self._workers.append(worker)
            logger.info("Denoiser worker ready on %s", device)

        logger.info("All models loaded: %d workers, encoder on %s", len(self._workers), self._encoder_device)
```

**Step 2: Verify import still works**

Run: `uv run --no-sync python -c "from split_model_manager import SplitModelManager; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add split_model_manager.py
git commit -m "feat: load dual denoiser workers + shared encoder hub"
```

---

## Task 3: Add Worker Acquisition and Context Transfer

**Files:**
- Modify: `split_model_manager.py` — add `_acquire_worker`, update `_contexts_to_denoiser`

**Step 1: Add worker acquisition method**

```python
    async def _acquire_worker(self) -> DenoiserWorker:
        """Wait for and return the first unlocked worker."""
        while True:
            for worker in self._workers:
                if not worker.lock.locked():
                    await worker.lock.acquire()
                    return worker
            await asyncio.sleep(0.05)
```

**Step 2: Update _contexts_to_denoiser to accept target device**

Change from:
```python
    def _contexts_to_denoiser(self, contexts):
        ...
        target = self._denoiser_device
```
To:
```python
    def _contexts_to_device(self, contexts, target: torch.device):
        """Move EmbeddingsProcessorOutput tensors to target device."""
        from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessorOutput
        return [
            EmbeddingsProcessorOutput(
                video_encoding=ctx.video_encoding.to(target),
                audio_encoding=ctx.audio_encoding.to(target) if ctx.audio_encoding is not None else None,
                attention_mask=ctx.attention_mask.to(target),
            )
            for ctx in contexts
        ]
```

**Step 3: Commit**

```bash
git add split_model_manager.py
git commit -m "feat: add worker acquisition and device-flexible context transfer"
```

---

## Task 4: Refactor Generation Flows to Use Workers

This is the largest task. Each `_run_*` method changes from using `self._denoiser_device` / `self._denoiser_ledger` / `self._ensure_transformer` to using a `worker` parameter.

**Files:**
- Modify: `split_model_manager.py` — all 4 `_run_*` methods

**Step 1: Refactor _run_t2v**

Change signature to accept worker:
```python
    @torch.inference_mode()
    def _run_t2v(
        self, worker: DenoiserWorker, prompt: str, model: str, width: int, height: int,
        num_frames: int, fps: float, seed: int, generate_audio: bool,
    ) -> bytes:
        device = worker.device
        dtype = torch.bfloat16
        is_fast = model == "ltx-2-3-fast"

        worker.ensure_transformer("distilled" if is_fast else "dev")

        # Text encoding on GPU:0 (shared encoder)
        if is_fast:
            (ctx_p,) = encode_prompts([prompt], self._encoder_ledger)
            (ctx_p,) = self._contexts_to_device([ctx_p], device)
            v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
            v_context_n, a_context_n = None, None
        else:
            ctx_p, ctx_n = encode_prompts([prompt, DEFAULT_NEGATIVE_PROMPT], self._encoder_ledger)
            ctx_p, ctx_n = self._contexts_to_device([ctx_p, ctx_n], device)
            v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
            v_context_n, a_context_n = ctx_n.video_encoding, ctx_n.audio_encoding

        generator = torch.Generator(device=device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()

        # Stage 1: half-resolution denoising
        stage_1_shape = VideoPixelShape(batch=1, frames=num_frames, width=width // 2, height=height // 2, fps=fps)
        video_encoder = worker.ledger.video_encoder()
        stage_1_cond = combined_image_conditionings(images=[], height=stage_1_shape.height, width=stage_1_shape.width, video_encoder=video_encoder, dtype=dtype, device=device)

        transformer = worker.ledger.transformer()

        if is_fast:
            sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(device)
            def denoising_loop(sigmas, video_state, audio_state, stepper):
                return euler_denoising_loop(sigmas=sigmas, video_state=video_state, audio_state=audio_state, stepper=stepper,
                    denoise_fn=simple_denoising_func(video_context=v_context_p, audio_context=a_context_p, transformer=transformer))
        else:
            params = detect_params(config.DEV_CHECKPOINT)
            sigmas = LTX2Scheduler().execute(steps=params.num_inference_steps).to(dtype=torch.float32, device=device)
            def denoising_loop(sigmas, video_state, audio_state, stepper):
                return euler_denoising_loop(sigmas=sigmas, video_state=video_state, audio_state=audio_state, stepper=stepper,
                    denoise_fn=multi_modal_guider_factory_denoising_func(
                        video_guider_factory=create_multimodal_guider_factory(params=params.video_guider_params, negative_context=v_context_n),
                        audio_guider_factory=create_multimodal_guider_factory(params=params.audio_guider_params, negative_context=a_context_n),
                        v_context=v_context_p, a_context=a_context_p, transformer=transformer))

        video_state, audio_state = denoise_audio_video(
            output_shape=stage_1_shape, conditionings=stage_1_cond, noiser=noiser,
            sigmas=sigmas, stepper=stepper, denoising_loop_fn=denoising_loop,
            components=worker.components, dtype=dtype, device=device,
        )

        # Stage 2: upsample + refine
        upscaled = upsample_video(latent=video_state.latent[:1], video_encoder=video_encoder, upsampler=worker.ledger.spatial_upsampler())
        stage_2_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=fps)
        stage_2_cond = combined_image_conditionings(images=[], height=stage_2_shape.height, width=stage_2_shape.width, video_encoder=video_encoder, dtype=dtype, device=device)

        if not is_fast:
            worker.ensure_transformer("dev_lora")
            transformer = worker.ledger.transformer()

        distilled_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(device)
        def stage2_loop(sigmas, video_state, audio_state, stepper):
            return euler_denoising_loop(sigmas=sigmas, video_state=video_state, audio_state=audio_state, stepper=stepper,
                denoise_fn=simple_denoising_func(video_context=v_context_p, audio_context=a_context_p, transformer=transformer))

        video_state, audio_state = denoise_audio_video(
            output_shape=stage_2_shape, conditionings=stage_2_cond, noiser=noiser,
            sigmas=distilled_sigmas, stepper=stepper, denoising_loop_fn=stage2_loop,
            components=worker.components, dtype=dtype, device=device,
            noise_scale=distilled_sigmas[0], initial_video_latent=upscaled,
            initial_audio_latent=audio_state.latent,
        )

        if not is_fast:
            worker.ensure_transformer("dev")

        decoded_video = vae_decode_video(video_state.latent, worker.ledger.video_decoder(), TilingConfig.default(), generator)
        decoded_audio = vae_decode_audio(audio_state.latent, worker.ledger.audio_decoder(), worker.ledger.vocoder())
        return _video_to_bytes(decoded_video, fps, decoded_audio, num_frames, include_audio=generate_audio)
```

**Step 2: Apply same pattern to _run_i2v**

Same changes: `self._denoiser_device` → `worker.device`, `self._denoiser_ledger` → `worker.ledger`, `self._ensure_transformer` → `worker.ensure_transformer`, `self._components` → `worker.components`.

**Step 3: Apply same pattern to _run_a2v**

Additional note: audio encoding still uses `self._encoder_ledger` on GPU:0. The encoded audio latent transfers to `worker.device`:
```python
        encoded_audio_latent = encoded_audio_latent[:, :, :audio_shape.frames].to(worker.device)
```

**Step 4: Apply same pattern to _run_retake**

Same mechanical changes. Video conditioning encoding uses `self._encoder_ledger.video_encoder()` on GPU:0, then transfers:
```python
        initial_video_latent = video_encoder_enc(video_conditioning.to(self._encoder_device, dtype=dtype)).to(worker.device)
```

**Step 5: Verify import**

Run: `uv run --no-sync python -c "from split_model_manager import SplitModelManager; print('OK')"`
Expected: `OK`

**Step 6: Commit**

```bash
git add split_model_manager.py
git commit -m "refactor: generation flows use per-worker device and transformer"
```

---

## Task 5: Refactor Async API for Concurrent Dispatch

**Files:**
- Modify: `split_model_manager.py` — async wrapper methods

**Step 1: Replace single-lock async methods with worker acquisition**

Delete the old `_lock`-based methods and replace with:

```python
    async def generate_text_to_video(
        self, prompt: str, model: str, width: int, height: int,
        num_frames: int, fps: float, seed: int, generate_audio: bool = True,
    ) -> bytes:
        worker = await self._acquire_worker()
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self._run_t2v, worker, prompt, model, width, height,
                num_frames, fps, seed, generate_audio,
            )
        finally:
            worker.lock.release()

    async def generate_image_to_video(
        self, prompt: str, image_path: str, model: str, width: int, height: int,
        num_frames: int, fps: float, seed: int, generate_audio: bool = True,
    ) -> bytes:
        worker = await self._acquire_worker()
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self._run_i2v, worker, prompt, image_path, model, width, height,
                num_frames, fps, seed, generate_audio,
            )
        finally:
            worker.lock.release()

    async def generate_audio_to_video(
        self, prompt: str, audio_path: str, image_path: str | None,
        model: str, width: int, height: int, num_frames: int, fps: float, seed: int,
    ) -> bytes:
        worker = await self._acquire_worker()
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self._run_a2v, worker, prompt, audio_path, image_path,
                width, height, num_frames, fps, seed,
            )
        finally:
            worker.lock.release()

    async def retake(
        self, video_path: str, start_time: float, duration: float,
        mode: str, prompt: str, seed: int,
    ) -> bytes:
        worker = await self._acquire_worker()
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self._run_retake, worker, video_path, start_time, duration,
                mode, prompt, seed,
            )
        finally:
            worker.lock.release()
```

**Step 2: Remove old methods**

Delete the old `_ensure_transformer`, `_contexts_to_denoiser`, and single `_lock` field — they've been replaced by `DenoiserWorker.ensure_transformer`, `_contexts_to_device`, and per-worker locks.

**Step 3: Verify import**

Run: `uv run --no-sync python -c "from split_model_manager import SplitModelManager; m = SplitModelManager(); print(f'ready={m.is_ready}, workers={len(m._workers)}')"`
Expected: `ready=False, workers=0`

**Step 4: Commit**

```bash
git add split_model_manager.py
git commit -m "feat: concurrent dispatch via per-worker locks and acquisition"
```

---

## Task 6: GPU Smoke Test

**Step 1: Kill existing server**

```bash
pkill -f "uvicorn server:app" 2>/dev/null; sleep 3
```

**Step 2: Start server**

```bash
bash run.sh > /tmp/taco-backend.log 2>&1 &
```

Wait ~30s for model loading. Check logs:
```bash
tail -20 /tmp/taco-backend.log
```

Expected: 2 "Denoiser worker ready on cuda:X" messages + "All models loaded: 2 workers"

**Step 3: Check GPU memory distribution**

```bash
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
```

Expected:
- GPU 0 (nvidia-smi): ~60GB used (encoder hub + denoiser)
- GPU 2 (nvidia-smi): ~48GB used (denoiser only)
- GPU 1 (nvidia-smi): ~0GB (unused 24GB card)

**Step 4: Single request smoke test**

```bash
curl -s -X POST http://localhost:8090/v1/text-to-video \
  -H "Content-Type: application/json" \
  -d '{"prompt":"A cat on a windowsill","model":"ltx-2-3-fast","resolution":"1920x1080","duration":2.0,"fps":24.0}' \
  -o /tmp/smoke-single.mp4 -w "%{http_code} %{size_download} bytes %{time_total}s\n"
```

Expected: `200` with MP4 output in ~15s

**Step 5: Concurrent request smoke test**

Fire 2 requests simultaneously:
```bash
curl -s -X POST http://localhost:8090/v1/text-to-video \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Ocean waves at sunset","model":"ltx-2-3-fast","resolution":"1920x1080","duration":2.0,"fps":24.0}' \
  -o /tmp/smoke-concurrent-1.mp4 -w "Job1: %{http_code} %{size_download}b %{time_total}s\n" &

curl -s -X POST http://localhost:8090/v1/text-to-video \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Mountain sunrise timelapse","model":"ltx-2-3-fast","resolution":"1920x1080","duration":2.0,"fps":24.0}' \
  -o /tmp/smoke-concurrent-2.mp4 -w "Job2: %{http_code} %{size_download}b %{time_total}s\n" &

wait
```

Expected: Both return `200`. Total wall time should be ~15-20s (NOT 30s), proving concurrent execution.

**Step 6: Commit (if not already)**

```bash
git add -A
git commit -m "test: verify dual-denoiser concurrent generation"
```

---

## Task 7: Frontend Concurrency Guide

**Files:**
- No code changes needed on backend

**The backend already supports concurrent requests.** The frontend (taco-desktop) just needs to stop serializing requests. Here's what to change:

### For taco-desktop (Electron app)

The client currently waits for each generation request to complete before sending the next. To enable concurrency:

1. **Fire-and-forget pattern**: Send generation requests without awaiting the previous one. The backend queues them internally via `_acquire_worker()`.

2. **Connection pooling**: Ensure the HTTP client allows multiple concurrent connections to the same host. Most HTTP clients default to 6+ concurrent connections per host.

3. **No backend changes needed**: The `/v1/text-to-video`, `/v1/image-to-video`, `/v1/audio-to-video`, and `/v1/retake` endpoints all use the same `_acquire_worker()` pattern. When both GPUs are busy, the third request blocks until one finishes.

4. **Capacity indicator** (optional): Add `GET /health` response field for available workers:
   ```json
   {"status": "ok", "available_workers": 1, "total_workers": 2}
   ```
   This lets the frontend show queue state to the user.

### For tacojourney worker (`packages/backend/app/worker.py`)

The tacojourney worker has a **single-worker Redis lock** (line 167: `tacoview:worker:lock`). To enable 2 concurrent workers:

1. Remove the `tacoview:worker:lock` single-instance guard
2. Change `dequeue_job(count=1)` to `count=1` per worker (already correct)
3. Run 2 worker instances: `--consumer worker-0` and `--consumer worker-1`
4. Each worker uses one GPU via `CUDA_VISIBLE_DEVICES`

---

## Summary

| Task | What | Risk |
|------|------|------|
| 1 | DenoiserWorker dataclass | Low — new code |
| 2 | Dual load_all() | Medium — VRAM fit on GPU:0 |
| 3 | Worker acquisition + context transfer | Low — pattern from pipeline_manager |
| 4 | Refactor 4 generation flows | Medium — mechanical but large |
| 5 | Concurrent async dispatch | Low — remove single lock |
| 6 | GPU smoke test | High — first real validation |
| 7 | Frontend guide | Low — documentation only |

---

## Validation Findings (addressed in v2)

1. **CRITICAL — video_encoder duplication**: Plan v1 removed video_encoder from encoder_cache (broke retake) and would have loaded it twice on GPU:0 (~5GB waste). **Fix:** Keep video_encoder in encoder_cache; GPU:0's denoiser worker reuses the encoder hub's instance.

2. **CRITICAL — VRAM budget wrong**: Plan v1 claimed 60GB on GPU:0 (36GB headroom). Actual with all models: ~75GB (21GB headroom). Still fits 96GB safely — working memory for 1920x1080 is ~13GB.

3. **MEDIUM — _encode_lock unused**: Defined but never acquired. CUDA stream serialization handles thread safety implicitly. Removed dead field; added explanatory comment.

4. **LOW — GPU:0 CUDA stream contention**: Text encoding may wait up to 1 denoising step (~1.5s) when GPU:0 is concurrently denoising. This is ~2-3% overhead. Documented, acceptable.
