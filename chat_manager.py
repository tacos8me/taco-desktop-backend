"""Chat manager that proxies to an external OpenAI-compatible server.

Forwards /v1/chat/completions requests to a sglang/vLLM server running
Gemma 3 12B IT (or any OpenAI-compatible model).

v1.18.0-rc2: also exposes ``embed`` and ``embed_batch`` helpers that
proxy ``/v1/embeddings`` to llama-swap. Returns float32-packed bytes
(~14 KB per 3584-dim Gemma embedding) ready to insert into the
``clip_embeddings`` sqlite-vec virtual table.
"""

from __future__ import annotations

import logging
import struct
import time
import uuid

import httpx

import config

logger = logging.getLogger(__name__)

# 30s timeout for chat completions
_TIMEOUT = httpx.Timeout(30.0, connect=5.0)

# v1.18.0-rc2 — embeddings model identifier on llama-swap. Defaults to
# the same model as `CHAT_MODEL` so a single Gemma load can serve both
# completions and embeddings; override the env if llama-swap exposes a
# dedicated embed model with a different ID. Stored alongside each
# embedding for migration safety when the model rolls.
EMBEDDING_MODEL_VERSION = "gemma-3-12b-nvfp4"

# Generous timeout for batch embeddings — 64 inputs at ~30 toks each is
# a few seconds on hot llama-swap, but cold-load can spike.
_EMBED_TIMEOUT = httpx.Timeout(60.0, connect=5.0)


class ChatManager:
    """Proxies chat completions to an external LLM server."""

    def __init__(self) -> None:
        self._base_url = config.CHAT_API_BASE
        self._client: httpx.AsyncClient | None = None

    @property
    def is_ready(self) -> bool:
        return self._client is not None

    def load(self) -> None:
        """Initialize the httpx client."""
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=_TIMEOUT,
        )
        logger.info("Chat proxy ready → %s", self._base_url)

    async def unload(self) -> None:
        """Close the httpx client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.info("Chat proxy closed")

    def _build_response(self, text: str) -> dict:
        """Build an OpenAI-compatible chat completions response."""
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": config.CHAT_MODEL,
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

    async def generate_chat_completion(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 512,
        model: str | None = None,
    ) -> dict:
        """Forward chat completion to external server and return OpenAI-format response."""
        clean_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
        ]

        payload = {
            "model": model or config.CHAT_MODEL,
            "messages": clean_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        resp = await self._client.post("/v1/chat/completions", json=payload)
        resp.raise_for_status()

        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return self._build_response(text)

    async def embed(self, text: str, model: str | None = None) -> bytes:
        """Embed a single string via llama-swap ``/v1/embeddings``.

        Returns little-endian float32-packed bytes ready to insert into the
        ``clip_embeddings(embedding FLOAT[3584])`` sqlite-vec virtual table.
        Length = 4 * embedding_dim. Raises :class:`httpx.HTTPError` on
        transport failure and :class:`RuntimeError` when the response shape
        doesn't match OpenAI's spec — caller is expected to surface 503.
        """
        results = await self.embed_batch([text], model=model)
        return results[0]

    async def embed_batch(
        self, texts: list[str], model: str | None = None
    ) -> list[bytes]:
        """Embed a list of strings in one HTTP call to llama-swap.

        OpenAI-compatible: ``{"model": ..., "input": [str, ...]}`` →
        ``{"data": [{"embedding": [float, ...], "index": int}, ...]}``.
        Returns one float32-packed bytes blob per input, in input order.
        Empty input list short-circuits to ``[]`` without an HTTP call.
        """
        if not texts:
            return []
        if self._client is None:
            raise RuntimeError("chat client not loaded")
        payload = {
            "model": model or EMBEDDING_MODEL_VERSION,
            "input": texts,
        }
        resp = await self._client.post(
            "/v1/embeddings", json=payload, timeout=_EMBED_TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()
        items = data.get("data") or []
        if len(items) != len(texts):
            raise RuntimeError(
                f"embed_batch: expected {len(texts)} embeddings, got {len(items)}"
            )
        # Sort by `index` to match OpenAI spec — server may reorder under
        # parallel decode; defensive even if llama.cpp preserves order today.
        items_sorted = sorted(items, key=lambda x: x.get("index", 0))
        out: list[bytes] = []
        for item in items_sorted:
            vec = item.get("embedding")
            if not isinstance(vec, list) or not vec:
                raise RuntimeError("embed_batch: malformed embedding in response")
            out.append(struct.pack(f"<{len(vec)}f", *vec))
        return out
