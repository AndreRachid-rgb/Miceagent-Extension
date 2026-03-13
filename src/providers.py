"""
miceagent — Provider Layer (§7)

Contrato único de provider:
  - list_models() → list[str]
  - healthcheck() → ProviderStatus
  - chat(messages, model, tools?, stream?) → ChatResult

Implementações:
  - LMStudioProvider (padrão operacional)
  - OllamaProvider (segunda opção nativa)
  - OpenAICompatibleProvider (lingua franca)
"""

from __future__ import annotations

import httpx
from dataclasses import dataclass, field
from typing import Protocol, Any, runtime_checkable

# ── Tipos ──

@dataclass
class ProviderStatus:
    available: bool
    provider_name: str
    base_url: str
    error: str | None = None

@dataclass
class ChatMessage:
    role: str  # system | user | assistant | tool
    content: str
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None

@dataclass
class ChatResult:
    content: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = "stop"
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

# ── Protocol ──

@runtime_checkable
class ChatProvider(Protocol):
    def list_models(self) -> list[str]: ...
    def healthcheck(self) -> ProviderStatus: ...
    async def chat(
        self,
        messages: list[ChatMessage],
        model: str,
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
    ) -> ChatResult: ...


# ── OpenAI-Compatible Base ──

class OpenAICompatibleProvider:
    """Base genérica para qualquer endpoint OpenAI-compatible."""

    def __init__(self, base_url: str, api_key: str = "not-needed", name: str = "openai-compatible"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.name = name
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=120.0,
        )

    def list_models(self) -> list[str]:
        with httpx.Client(base_url=self.base_url, headers={"Authorization": f"Bearer {self.api_key}"}) as c:
            try:
                resp = c.get("/v1/models")
                resp.raise_for_status()
                data = resp.json()
                return [m["id"] for m in data.get("data", [])]
            except Exception:
                return []

    def healthcheck(self) -> ProviderStatus:
        try:
            with httpx.Client(base_url=self.base_url, timeout=5.0) as c:
                resp = c.get("/v1/models")
                return ProviderStatus(
                    available=resp.status_code == 200,
                    provider_name=self.name,
                    base_url=self.base_url,
                )
        except Exception as e:
            return ProviderStatus(
                available=False,
                provider_name=self.name,
                base_url=self.base_url,
                error=str(e),
            )

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str,
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        resp = await self._client.post("/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        message = choice.get("message", {})

        return ChatResult(
            content=message.get("content"),
            tool_calls=message.get("tool_calls", []),
            finish_reason=choice.get("finish_reason", "stop"),
            model=data.get("model", model),
            usage=data.get("usage", {}),
            raw=data,
        )


# ── LM Studio ──

class LMStudioProvider(OpenAICompatibleProvider):
    """LM Studio — padrão operacional (§7.1)."""

    def __init__(self, base_url: str = "http://127.0.0.1:1234"):
        super().__init__(base_url=base_url, api_key="lm-studio", name="lm-studio")


# ── Ollama ──

class OllamaProvider(OpenAICompatibleProvider):
    """Ollama — segunda opção nativa (§7.1)."""

    def __init__(self, base_url: str = "http://127.0.0.1:11434"):
        super().__init__(base_url=base_url, api_key="ollama", name="ollama")
        # Ollama usa /api/chat em vez de /v1/chat/completions,
        # mas também suporta /v1/ compatibility mode a partir de versões recentes.

    def list_models(self) -> list[str]:
        try:
            with httpx.Client(base_url=self.base_url, timeout=5.0) as c:
                resp = c.get("/api/tags")
                resp.raise_for_status()
                data = resp.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            return super().list_models()

    def healthcheck(self) -> ProviderStatus:
        try:
            with httpx.Client(base_url=self.base_url, timeout=5.0) as c:
                resp = c.get("/api/tags")
                return ProviderStatus(
                    available=resp.status_code == 200,
                    provider_name=self.name,
                    base_url=self.base_url,
                )
        except Exception as e:
            return ProviderStatus(
                available=False,
                provider_name=self.name,
                base_url=self.base_url,
                error=str(e),
            )


# ── Factory ──

def create_provider(provider_type: str, base_url: str | None = None) -> ChatProvider:
    match provider_type:
        case "lm-studio":
            return LMStudioProvider(base_url or "http://127.0.0.1:1234")
        case "ollama":
            return OllamaProvider(base_url or "http://127.0.0.1:11434")
        case "openai-compatible":
            if not base_url:
                raise ValueError("base_url is required for openai-compatible provider")
            return OpenAICompatibleProvider(base_url=base_url)
        case _:
            raise ValueError(f"Unknown provider: {provider_type}")
