# Prompt Enhancement Endpoint Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an OpenAI-compatible `/v1/chat/completions` endpoint to taco-backend for prompt enhancement, powered by Gemma 3 12B IT on cuda:2.

**Architecture:** New `ChatManager` class (following the `FluxManager` pattern) loads Gemma 3 12B IT QAT with int4 quantization via `TorchAoConfig` on cuda:2 (RTX PRO 4000, 24GB). The endpoint accepts OpenAI chat completions format and returns enhanced prompts. Multimodal support handles base64 image_url content parts for gap-fill suggestions.

**Tech Stack:** transformers 4.57.6, torchao (new dep for QAT quantization via TorchAoConfig), Gemma3ForConditionalGeneration + Gemma3Processor

---

## Context

### GPU Layout After Implementation
| GPU | Device | Model | VRAM |
|-----|--------|-------|------|
| RTX PRO 6000 96GB | cuda:0 | LTX-2 (SplitModelManager) | ~69GB |
| RTX PRO 6000 96GB | cuda:1 | Flux 2 Dev FP8 (FluxManager) | ~77GB |
| RTX PRO 4000 24GB | cuda:2 | Gemma 3 12B IT int4 (ChatManager) | ~8GB |

### Model Details
- **Model**: `google/gemma-3-12b-it-qat-q4_0-unquantized`
- **HF Cache**: `/mnt/nvme-1/huggingface/hub/models--google--gemma-3-12b-it-qat-q4_0-unquantized/`
- **Architecture**: `Gemma3ForConditionalGeneration` (multimodal: text + images)
- **Stored as**: BF16 (24GB on disk), needs quantization at load time → ~6-8GB VRAM
- **Quantization**: `TorchAoConfig(quant_type="int4_weight_only")` passed to `from_pretrained` — quantizes shard-by-shard during loading, avoiding full BF16 materialization in VRAM
- **Chat template**: Built into `Gemma3Processor.apply_chat_template()`

### Critical: apply_chat_template requires list-format content
`Gemma3Processor.apply_chat_template()` with `tokenize=True` iterates `message["content"]` as a list. Passing string content crashes with `TypeError: string indices must be integers`. **All messages must use list-format content**, even text-only:
```python
# WRONG: {"role": "user", "content": "hello"}
# RIGHT: {"role": "user", "content": [{"type": "text", "text": "hello"}]}
```

### Client Spec
- Client sends to `POST http://<host>:8080/v1/chat/completions` (port configurable in client settings)
- taco-backend runs on port 8090 — client config just needs updating to match
- Model name `loco-operator` in request body (we accept any model name)
- 30 second timeout
- No auth

### Concurrency Decision
The existing `_inference_lock` serializes Flux and LTX because FP8 layerwise casting causes CUBLAS crashes when both GPUs compute concurrently. Gemma on cuda:2 uses int4 quantization (NOT FP8 layerwise casting), so it should be safe to run concurrently. **ChatManager gets its own internal lock (like FluxManager) but does NOT join the shared `_inference_lock`**. This means prompt enhancement can run while video/image generation is in progress. If CUBLAS crashes occur, fallback: add chat to `_inference_lock`.

---

## Task 1: Install torchao dependency

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add torchao to dependencies and uv sources**

In `pyproject.toml`, add `"torchao"` to the `dependencies` list.

Also add to `[tool.uv.sources]` (torchao may need the PyTorch cu130 index, same as torch itself):
```toml
torchao = { index = "pytorch-cu130" }
```

**Step 2: Install**

Run: `uv pip install torchao --no-sync`

If that fails due to index issues, try: `uv pip install torchao --index-url https://download.pytorch.org/whl/cu130`

**Step 3: Verify**

```bash
python3 -c "from transformers import TorchAoConfig; print('TorchAoConfig OK')"
python3 -c "import torchao; print('torchao', torchao.__version__)"
```

**Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add torchao dependency for Gemma QAT quantization"
```

---

## Task 2: Add chat config to config.py

**Files:**
- Modify: `config.py:13-18`

**Step 1: Add chat config constants**

After the `FLUX_DEVICE` line in `config.py`, add:

```python
CHAT_DEVICE = "cuda:2"   # Gemma 3 12B IT for prompt enhancement (~8GB int4)
CHAT_MODEL = "google/gemma-3-12b-it-qat-q4_0-unquantized"
```

**Step 2: Verify config loads**

```bash
python3 -c "import config; print(config.CHAT_DEVICE, config.CHAT_MODEL)"
```
Expected: `cuda:2 google/gemma-3-12b-it-qat-q4_0-unquantized`

**Step 3: Commit**

```bash
git add config.py
git commit -m "feat: add CHAT_DEVICE and CHAT_MODEL config for Gemma 3 12B IT"
```

---

## Task 3: Create ChatManager class

**Files:**
- Create: `chat_manager.py`
- Test: `tests/test_chat_manager.py`

**Step 1: Write the failing test**

Create `tests/test_chat_manager.py`:

```python
"""Tests for ChatManager (no GPU required)."""

from chat_manager import ChatManager


def test_chat_manager_init():
    mgr = ChatManager()
    assert mgr.is_ready is False


def test_parse_messages_text_only():
    mgr = ChatManager()
    messages = [
        {"role": "system", "content": "You are a prompt engineer."},
        {"role": "user", "content": "a cat on a windowsill"},
    ]
    parsed = mgr._parse_messages(messages)
    assert len(parsed) == 2
    assert parsed[0]["role"] == "system"
    assert parsed[1]["role"] == "user"
    # ALL messages must use list-format content (Gemma3Processor requirement)
    assert isinstance(parsed[0]["content"], list)
    assert parsed[0]["content"][0] == {"type": "text", "text": "You are a prompt engineer."}
    assert isinstance(parsed[1]["content"], list)
    assert parsed[1]["content"][0] == {"type": "text", "text": "a cat on a windowsill"}


def test_parse_messages_with_image_url():
    mgr = ChatManager()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe this frame"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="},
                },
            ],
        }
    ]
    parsed = mgr._parse_messages(messages)
    assert len(parsed) == 1
    assert parsed[0]["role"] == "user"
    # Multimodal messages have list content with PIL Image
    content = parsed[0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image"


def test_build_openai_response():
    mgr = ChatManager()
    resp = mgr._build_response("Enhanced prompt text here")
    assert "choices" in resp
    assert len(resp["choices"]) == 1
    assert resp["choices"][0]["message"]["content"] == "Enhanced prompt text here"
    assert resp["choices"][0]["message"]["role"] == "assistant"
    assert "id" in resp
    assert "model" in resp
    assert "created" in resp
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_chat_manager.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'chat_manager'`

**Step 3: Implement ChatManager**

Create `chat_manager.py`:

```python
"""Gemma 3 12B IT chat manager for prompt enhancement.

Loads Gemma 3 12B IT QAT with int4 quantization on a dedicated GPU.
Provides OpenAI-compatible chat completions for prompt enhancement.
"""

from __future__ import annotations

import asyncio
import base64
import gc
import io
import logging
import time
import uuid

import torch
from PIL import Image

import config

logger = logging.getLogger(__name__)


class ChatManager:
    """Manages Gemma 3 chat pipeline lifecycle and inference."""

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._device = config.CHAT_DEVICE
        self._lock = asyncio.Lock()

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load the Gemma 3 12B IT model with int4 quantization."""
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration, TorchAoConfig

        logger.info("Loading Gemma 3 12B IT on %s ...", self._device)
        t0 = time.monotonic()

        model_id = config.CHAT_MODEL

        self._processor = AutoProcessor.from_pretrained(
            model_id,
            cache_dir=config.HF_CACHE_DIR,
            use_fast=True,
        )

        # TorchAoConfig quantizes shard-by-shard during loading,
        # avoiding full BF16 materialization (which would OOM on 24GB GPU).
        quantization_config = TorchAoConfig(quant_type="int4_weight_only")

        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map=self._device,
            torch_dtype=torch.bfloat16,
            cache_dir=config.HF_CACHE_DIR,
        )

        self._model = model

        elapsed = time.monotonic() - t0
        logger.info("Gemma 3 12B IT loaded in %.1fs on %s", elapsed, self._device)

    def unload(self) -> None:
        """Free GPU memory."""
        if self._model is not None:
            del self._model, self._processor
            self._model = None
            self._processor = None
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("Gemma 3 12B IT unloaded")

    def _parse_messages(self, messages: list[dict]) -> list[dict]:
        """Parse OpenAI-format messages into Gemma3Processor chat format.

        IMPORTANT: Gemma3Processor.apply_chat_template(tokenize=True) requires
        ALL message content to be in list format, even text-only messages.
        String content causes TypeError.
        """
        parsed = []
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")

            if isinstance(content, str):
                # Wrap text in list format (required by Gemma3Processor)
                parsed.append({
                    "role": role,
                    "content": [{"type": "text", "text": content}],
                })
            elif isinstance(content, list):
                # Multimodal: convert OpenAI format to Gemma format
                parts = []
                for part in content:
                    if part["type"] == "text":
                        parts.append({"type": "text", "text": part["text"]})
                    elif part["type"] == "image_url":
                        url = part["image_url"]["url"]
                        if url.startswith("data:"):
                            # data:image/png;base64,<data>
                            _, b64data = url.split(",", 1)
                            image_bytes = base64.b64decode(b64data)
                            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                        else:
                            raise ValueError(f"Unsupported image_url scheme: {url[:50]}")
                        parts.append({"type": "image", "image": image})
                parsed.append({"role": role, "content": parts})

        return parsed

    def _build_response(self, text: str) -> dict:
        """Build an OpenAI-compatible chat completions response."""
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "loco-operator",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": text,
                    },
                    "finish_reason": "stop",
                }
            ],
        }

    @torch.inference_mode()
    def _generate(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        """Generate a chat completion from parsed messages."""
        parsed = self._parse_messages(messages)

        inputs = self._processor.apply_chat_template(
            parsed,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._device)

        input_len = inputs["input_ids"].shape[-1]

        output = self._model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )

        # Decode only the new tokens
        generated = output[0][input_len:]
        text = self._processor.tokenizer.decode(generated, skip_special_tokens=True)
        return text.strip()

    async def generate_chat_completion(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> dict:
        """Generate a chat completion and return OpenAI-format response."""
        async with self._lock:
            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(
                None, self._generate, messages, temperature, max_tokens,
            )
        return self._build_response(text)
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_chat_manager.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add chat_manager.py tests/test_chat_manager.py
git commit -m "feat: add ChatManager for Gemma 3 12B IT prompt enhancement"
```

---

## Task 4: Add chat completions endpoint to server.py

**Files:**
- Modify: `server.py`
- Modify: `tests/test_server.py`

**Step 1: Write the failing test**

Add to `tests/test_server.py`:

```python
def test_chat_completions_returns_error_without_gpu():
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "loco-operator",
            "messages": [
                {"role": "system", "content": "You are a prompt engineer."},
                {"role": "user", "content": "a cat on a windowsill"},
            ],
            "temperature": 0.7,
            "max_tokens": 512,
        },
    )
    assert resp.status_code == 500
    data = resp.json()
    assert "error" in data


def test_chat_completions_validates_messages():
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "loco-operator",
            "messages": [],
        },
    )
    # Empty messages should fail validation (422) or return error
    assert resp.status_code in (422, 500)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_server.py::test_chat_completions_returns_error_without_gpu -v`
Expected: FAIL (404 — endpoint doesn't exist yet)

**Step 3: Add the endpoint to server.py**

Add imports at the top of `server.py`:

```python
from chat_manager import ChatManager
```

Add to globals section (after `uploads = UploadStore(...)`):

```python
chat = ChatManager()
```

Add to lifespan (after Flux loading):

```python
logger.info("Loading chat model on %s ...", config.CHAT_DEVICE)
chat.load()
logger.info("Chat model ready.")
```

Add request model (after existing request models):

```python
class ChatMessage(BaseModel):
    role: str
    content: str | list  # str for text, list for multimodal

class ChatCompletionRequest(BaseModel):
    model: str = "loco-operator"
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 512
```

Add health endpoint update — include `chat` status:

```python
"chat": "ready" if chat.is_ready else "not_loaded",
```

Add the endpoint (after existing endpoints, before `/v1/upload`):

```python
@app.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest) -> JSONResponse:
    if not chat.is_ready:
        return _error(500, "Chat model not loaded")
    if not body.messages:
        return _error(422, "Messages list cannot be empty")
    try:
        messages = [m.model_dump() for m in body.messages]
        result = await chat.generate_chat_completion(
            messages=messages,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
        )
        return JSONResponse(content=result)
    except Exception as exc:
        logger.exception("chat completion failed")
        return _error(500, str(exc))
```

**Note:** This endpoint does NOT use `_inference_lock` — Gemma on cuda:2 uses int4 quantization (not FP8 layerwise casting), so concurrent inference should be safe.

**Step 4: Run tests**

Run: `uv run pytest tests/test_server.py -v`
Expected: All tests PASS (including new chat tests)

**Step 5: Update health test**

Update `test_health` to also check for `chat` field:
```python
assert "chat" in data
```

**Step 6: Commit**

```bash
git add server.py tests/test_server.py
git commit -m "feat: add /v1/chat/completions endpoint for prompt enhancement"
```

---

## Task 5: Smoke test with live model

**Step 1: Start the server**

```bash
./run.sh
```

Wait for all three models to load (LTX, Flux, Gemma).

**Step 2: Test health**

```bash
curl -s http://localhost:8090/health | python3 -m json.tool
```

Expected: `{"status": "ok", "ltx": "ready", "flux": "ready", "chat": "ready"}`

**Step 3: Test text-only prompt enhancement**

```bash
curl -s http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "loco-operator",
    "messages": [
      {"role": "system", "content": "You are an expert prompt engineer for AI video generation. The user will give you a short prompt. Your job is to enhance it into a detailed, vivid prompt that will produce better results.\n\nGuidelines:\n- Expand on the scene with specific visual details: lighting, colors, textures, mood\n- Describe camera movement, action, and pacing\n- Keep the original intent and subject matter\n- Output 2-4 sentences max\n- Write only the enhanced prompt, no explanations or labels"},
      {"role": "user", "content": "a cat sitting on a windowsill"}
    ],
    "temperature": 0.7,
    "max_tokens": 512
  }' | python3 -m json.tool
```

Expected: OpenAI-format response with enhanced prompt in `choices[0].message.content`.

**Step 4: Test multimodal (base64 image)**

```bash
# Create a tiny test PNG as base64
B64=$(python3 -c "
import base64, io
from PIL import Image
img = Image.new('RGB', (64, 64), color='red')
buf = io.BytesIO()
img.save(buf, format='PNG')
print(base64.b64encode(buf.getvalue()).decode())
")

curl -s http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"loco-operator\",
    \"messages\": [
      {\"role\": \"user\", \"content\": [
        {\"type\": \"text\", \"text\": \"Describe what you see in this frame and suggest a video prompt\"},
        {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/png;base64,$B64\"}}
      ]}
    ],
    \"temperature\": 0.7,
    \"max_tokens\": 256
  }" | python3 -m json.tool
```

Expected: Response describing the red image and suggesting a prompt.

**Step 5: Verify VRAM usage**

```bash
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
```

Expected: cuda:2 shows ~6-10GB used, well within 24GB limit.

**Step 6: Test concurrent inference**

In one terminal, start a video generation:
```bash
curl -s http://localhost:8090/v1/text-to-video -d '{"prompt":"test","model":"ltx-2-3-fast","resolution":"1920x1080","duration":2,"fps":24}' -o /dev/null &
```

In another, immediately run chat completion:
```bash
curl -s http://localhost:8090/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"loco-operator","messages":[{"role":"user","content":"test prompt"}]}' | python3 -m json.tool
```

Expected: Both complete without CUBLAS errors. If crashes occur, add chat to `_inference_lock` in server.py.

---

## Task 6: Update README and commit

**Files:**
- Modify: `README.md`

**Step 1: Add chat completions documentation**

Add a new section to README.md documenting the `/v1/chat/completions` endpoint with request/response examples.

**Step 2: Commit and push**

```bash
git add README.md
git commit -m "docs: add chat completions endpoint to README"
```

---

## Resolved Issues (from validation review)

1. **apply_chat_template crash** — `_parse_messages` always returns list-format content, even for text-only messages. String content causes `TypeError` with `tokenize=True`.

2. **OOM during loading** — Uses `TorchAoConfig` + `device_map` in `from_pretrained()` instead of load-then-quantize. This quantizes shard-by-shard during loading, never materializing full BF16 in VRAM.

3. **torchao cu130 index** — Added to `[tool.uv.sources]` with `{ index = "pytorch-cu130" }`.

4. **Dead image extraction code** — Removed from `_generate`. `apply_chat_template` handles images automatically via the `{"type": "image", "image": pil_image}` content format.

5. **Missing `created` timestamp** — Added `"created": int(time.time())` to `_build_response`.

6. **Processor `use_fast` warning** — Added `use_fast=True` to `AutoProcessor.from_pretrained()`.
