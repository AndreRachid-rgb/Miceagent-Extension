from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from src import main
from src.providers import ChatMessage, OpenAICompatibleProvider


class _DummyProvider:
    def healthcheck(self):
        return type("S", (), {"available": True, "provider_name": "dummy", "base_url": "http://dummy"})()

    def list_models(self):
        return ["dummy-model"]


class _TestableProvider(OpenAICompatibleProvider):
    def set_client(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def close_client(self) -> None:
        await self._client.aclose()


def test_planner_start_rejects_too_many_attachments(monkeypatch: MonkeyPatch) -> None:
    main.provider = _DummyProvider()

    async def _ok_send(_message: dict[str, Any]) -> bool:
        return True

    monkeypatch.setattr(main.manager, "send_to_first", _ok_send)

    payload: dict[str, Any] = {
        "goal": "test",
        "attachments": [f"att_{i}" for i in range(main.MAX_ATTACHMENTS_PER_GOAL + 1)],
    }

    with TestClient(main.app) as client:
        response = client.post("/planner/start", json=payload)

    assert response.status_code == 400
    assert str(main.MAX_ATTACHMENTS_PER_GOAL) in response.json()["detail"]


def test_resolve_attachment_contexts_large_image_uses_text_excerpt(tmp_path: Path) -> None:
    image_path = tmp_path / "big.png"
    image_path.write_bytes(b"x" * (main.MAX_INLINE_IMAGE_BYTES + 1))

    attachment_id = "att_big"
    main.uploaded_attachments.clear()
    main.uploaded_attachments[attachment_id] = {
        "attachment_id": attachment_id,
        "filename": "big.png",
        "media_type": "image/png",
        "size": image_path.stat().st_size,
        "path": str(image_path),
        "sha256": "z" * 64,
        "created_at": "2099-01-01T00:00:00+00:00",
    }

    contexts = main.resolve_attachment_contexts([attachment_id])

    assert len(contexts) == 1
    assert "image_data_url" not in contexts[0]
    assert "too large for inline context" in contexts[0].get("text_excerpt", "")

    main.uploaded_attachments.clear()


@pytest.mark.asyncio
async def test_provider_chat_falls_back_to_text_only_for_multimodal() -> None:
    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        calls.append(body)

        first_message_content = body["messages"][0]["content"]
        is_multimodal = isinstance(first_message_content, list)

        if is_multimodal:
            return httpx.Response(
                status_code=400,
                json={"error": {"message": "multimodal not supported"}},
                request=request,
            )

        return httpx.Response(
            status_code=200,
            json={
                "model": "dummy",
                "choices": [
                    {
                        "message": {"content": "fallback ok", "tool_calls": []},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
            request=request,
        )

    transport = httpx.MockTransport(handler)
    provider = _TestableProvider(base_url="http://mock.local", api_key="x")
    await provider.close_client()
    provider.set_client(httpx.AsyncClient(base_url="http://mock.local", transport=transport))

    result = await provider.chat(
        model="dummy",
        messages=[
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": "Analyze this"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ],
            )
        ],
        tools=None,
    )

    assert result.content == "fallback ok"
    assert len(calls) == 2
    assert isinstance(calls[0]["messages"][0]["content"], list)
    assert isinstance(calls[1]["messages"][0]["content"], str)
    assert "image attachment omitted" in calls[1]["messages"][0]["content"]

    await provider.close_client()
