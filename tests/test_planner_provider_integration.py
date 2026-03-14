from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

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


class _EchoProvider:
    def healthcheck(self):
        return type("S", (), {"available": True, "provider_name": "echo", "base_url": "http://echo"})()

    def list_models(self):
        return ["echo-model"]

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str,
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
    ):
        _ = messages, model, tools, stream
        return type(
            "R",
            (),
            {
                "content": "ok",
                "thinking": "",
                "tool_calls": [],
            },
        )()


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


def test_resolve_attachment_contexts_converts_webp_for_inline_llm(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    image_path = tmp_path / "sample.webp"
    image_path.write_bytes(b"webp-bytes")

    attachment_id = "att_webp"
    main.uploaded_attachments.clear()
    main.uploaded_attachments[attachment_id] = {
        "attachment_id": attachment_id,
        "filename": "sample.webp",
        "media_type": "image/webp",
        "size": image_path.stat().st_size,
        "path": str(image_path),
        "sha256": "w" * 64,
        "created_at": "2099-01-01T00:00:00+00:00",
    }

    def _fake_normalize_inline_image_for_llm(media_type: str, payload: bytes) -> tuple[str, bytes] | None:
        if media_type == "image/webp":
            return "image/png", b"png-bytes"
        return media_type, payload

    monkeypatch.setattr(main, "normalize_inline_image_for_llm", _fake_normalize_inline_image_for_llm)

    contexts = main.resolve_attachment_contexts([attachment_id])

    assert len(contexts) == 1
    assert contexts[0]["image_data_url"].startswith("data:image/png;base64,")

    main.uploaded_attachments.clear()

    main.uploaded_attachments.clear()


def test_resolve_attachment_contexts_extracts_json_and_csv(tmp_path: Path) -> None:
    json_path = tmp_path / "data.json"
    json_path.write_text('{"name":"miceagent","ok":true}', encoding="utf-8")

    csv_path = tmp_path / "data.csv"
    csv_path.write_text("name,value\nalpha,1\nbeta,2\n", encoding="utf-8")

    main.uploaded_attachments.clear()
    main.uploaded_attachments["att_json"] = {
        "attachment_id": "att_json",
        "filename": "data.json",
        "media_type": "application/json",
        "size": json_path.stat().st_size,
        "path": str(json_path),
        "sha256": "j" * 64,
        "created_at": "2099-01-01T00:00:00+00:00",
    }
    main.uploaded_attachments["att_csv"] = {
        "attachment_id": "att_csv",
        "filename": "data.csv",
        "media_type": "text/csv",
        "size": csv_path.stat().st_size,
        "path": str(csv_path),
        "sha256": "c" * 64,
        "created_at": "2099-01-01T00:00:00+00:00",
    }

    contexts = main.resolve_attachment_contexts(["att_json", "att_csv"])

    assert len(contexts) == 2
    assert '"name": "miceagent"' in contexts[0].get("text_excerpt", "")
    assert "alpha | 1" in contexts[1].get("text_excerpt", "")

    main.uploaded_attachments.clear()


@pytest.mark.asyncio
async def test_provider_chat_falls_back_to_text_only_for_multimodal() -> None:
    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        calls.append(body)

        first_message_content = body["messages"][0]["content"]
        parts: list[dict[str, Any]] = []
        if isinstance(first_message_content, list):
            parts = cast(list[dict[str, Any]], first_message_content)
        is_standard_multimodal = (
            isinstance(first_message_content, list)
            and any(p.get("type") == "image_url" for p in parts)
        )
        is_alternate_multimodal = (
            isinstance(first_message_content, list)
            and any(p.get("type") == "image" for p in parts)
        )

        if is_standard_multimodal or is_alternate_multimodal:
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
    assert len(calls) == 4
    assert isinstance(calls[0]["messages"][0]["content"], list)
    assert isinstance(calls[1]["messages"][0]["content"], list)
    assert any(part.get("type") == "image" for part in calls[1]["messages"][0]["content"])
    assert isinstance(calls[2]["messages"][0]["content"], list)
    assert any(
        part.get("type") == "image_url" and isinstance(part.get("image_url"), str)
        for part in calls[2]["messages"][0]["content"]
    )
    assert isinstance(calls[3]["messages"][0]["content"], str)
    assert "image attachment omitted" in calls[3]["messages"][0]["content"]

    await provider.close_client()


def test_send_chat_reuses_attachments_from_chat_session(tmp_path: Path) -> None:
    txt_path = tmp_path / "context.txt"
    txt_path.write_text("contexto persistente", encoding="utf-8")

    main.provider = _EchoProvider()
    main.configured_model = "echo-model"
    main.uploaded_attachments.clear()
    main.chat_sessions.clear()

    attachment_id = "att_chat_1"
    main.uploaded_attachments[attachment_id] = {
        "attachment_id": attachment_id,
        "filename": "context.txt",
        "media_type": "text/plain",
        "size": txt_path.stat().st_size,
        "path": str(txt_path),
        "sha256": "a" * 64,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    with TestClient(main.app) as client:
        first = client.post(
            "/chat/send",
            json={
                "chat_session_id": "chat-abc",
                "messages": [{"role": "user", "content": "primeira pergunta"}],
                "attachments": [attachment_id],
            },
        )
        second = client.post(
            "/chat/send",
            json={
                "chat_session_id": "chat-abc",
                "messages": [{"role": "user", "content": "segunda pergunta"}],
                "attachments": [],
            },
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["attachments_used"] == [attachment_id]
    assert second.json()["attachments_used"] == [attachment_id]

    main.uploaded_attachments.clear()
    main.chat_sessions.clear()
