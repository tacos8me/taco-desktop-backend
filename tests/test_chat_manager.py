"""Tests for ChatManager proxy (no GPU required)."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from chat_manager import ChatManager


def test_chat_manager_init():
    mgr = ChatManager()
    assert mgr.is_ready is False


def test_chat_manager_ready_after_load():
    mgr = ChatManager()
    mgr.load()
    assert mgr.is_ready is True


def test_build_openai_response():
    mgr = ChatManager()
    resp = mgr._build_response("Enhanced prompt text here")
    assert "choices" in resp
    assert len(resp["choices"]) == 1
    assert resp["choices"][0]["message"]["content"] == "Enhanced prompt text here"
    assert resp["choices"][0]["message"]["role"] == "assistant"
    assert resp["id"].startswith("chatcmpl-")
    assert resp["model"] == "loco-operator"
    assert "created" in resp


@pytest.mark.asyncio
async def test_generate_forwards_to_server():
    """Test that generate_chat_completion proxies to the external server."""
    mgr = ChatManager()
    mgr.load()

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [
            {"message": {"role": "assistant", "content": "A cinematic shot of a cat"}}
        ]
    }
    mock_response.raise_for_status = MagicMock()

    mgr._client.post = AsyncMock(return_value=mock_response)

    result = await mgr.generate_chat_completion(
        messages=[{"role": "user", "content": "a cat on a windowsill"}],
        temperature=0.7,
        max_tokens=512,
    )

    assert result["choices"][0]["message"]["content"] == "A cinematic shot of a cat"
    assert result["model"] == "loco-operator"

    call_args = mgr._client.post.call_args
    assert call_args[0][0] == "/v1/chat/completions"
    payload = call_args[1]["json"]
    assert payload["messages"] == [{"role": "user", "content": "a cat on a windowsill"}]
    assert payload["temperature"] == 0.7
    assert payload["max_tokens"] == 512
    assert "model" in payload
