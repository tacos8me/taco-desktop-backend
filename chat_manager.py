"""Chat manager that proxies to an external OpenAI-compatible server.

Forwards /v1/chat/completions requests to a sglang/vLLM server running
Gemma 3 12B IT (or any OpenAI-compatible model).
"""

from __future__ import annotations

import logging
import time
import uuid

import httpx

import config

logger = logging.getLogger(__name__)

# 30s timeout for chat completions
_TIMEOUT = httpx.Timeout(30.0, connect=5.0)


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
    ) -> dict:
        """Forward chat completion to external server and return OpenAI-format response."""
        payload = {
            "model": config.CHAT_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        resp = await self._client.post("/v1/chat/completions", json=payload)
        resp.raise_for_status()

        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return self._build_response(text)
