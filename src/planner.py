"""
miceagent — Planner (§7.5)

Responsabilidades:
  - Transformar objetivo do usuário em steps e tool calls
  - Montar contexto para o LLM (§7.3)
  - Gerenciar memória de step, sessão e artefato (§7.4)
  - NUNCA acessar DOM diretamente

O planner roda no backend Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
try:
    from .providers import ChatProvider, ChatMessage, ChatResult
except ImportError:
    try:
        from src.providers import ChatProvider, ChatMessage, ChatResult
    except ModuleNotFoundError:
        from providers import ChatProvider, ChatMessage, ChatResult  # type: ignore


# ── Sistema de Tools exposto ao LLM (§5.2) ──

MICEAGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "click_target",
            "description": "Click on a target element identified by its target_id from the current snapshot.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_id": {"type": "string", "description": "The target_id from the snapshot."}
                },
                "required": ["target_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "type_text",
            "description": "Type text into an input/textarea element.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_id": {"type": "string"},
                    "text": {"type": "string", "description": "The text to type."},
                    "press_enter": {"type": "boolean", "description": "Whether to press Enter after typing.", "default": False},
                },
                "required": ["target_id", "text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "select_option",
            "description": "Select an option in a <select> element by value or label.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_id": {"type": "string"},
                    "value_or_label": {"type": "string"},
                },
                "required": ["target_id", "value_or_label"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scroll_page",
            "description": "Scroll the page up or down.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {"type": "string", "enum": ["up", "down"]},
                    "pixels": {"type": "integer", "default": 400},
                },
                "required": ["direction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scroll_target_into_view",
            "description": "Scroll to make a specific target visible in the viewport.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_id": {"type": "string"},
                },
                "required": ["target_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "capture_snapshot",
            "description": "Request a new snapshot of the current page state. Use when the state may be stale.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "Why you need a new snapshot."},
                },
                "required": ["reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hover_target",
            "description": "Hover over a target element to trigger tooltips or dropdowns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_id": {"type": "string"},
                },
                "required": ["target_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "press_key",
            "description": "Press a keyboard key (e.g., Escape, Tab, ArrowDown).",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                },
                "required": ["key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": "Signal that the task is complete. Provide a summary of what was accomplished.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                },
                "required": ["summary"],
            },
        },
    },
]


# ── Memória (§7.4) ──

@dataclass
class StepMemory:
    """Último objetivo, última avaliação, último erro, último snapshot."""
    last_goal: str = ""
    last_evaluation: str = ""
    last_error: str | None = None
    last_snapshot_summary: str = ""


@dataclass
class SessionMemory:
    """Fatos estáveis da tarefa: tabs, forms preenchidos, progresso."""
    goal: str = ""
    completed_steps: list[str] = field(default_factory=list)
    filled_fields: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


# ── Planner ──

class Planner:
    def __init__(self, provider: ChatProvider, model: str):
        self.provider = provider
        self.model = model
        self.step_memory = StepMemory()
        self.session_memory = SessionMemory()
        self.message_history: list[ChatMessage] = []

    def build_system_prompt(self) -> str:
        """Contexto sugerido §7.3."""
        return """You control a Firefox extension-based browser agent called miceagent.
Use only the provided tools to interact with web pages.
If the current state is stale or ambiguous, request a new snapshot using capture_snapshot.

Rules:
- Always refer to elements by their target_id from the latest snapshot.
- After performing an action, you will receive a resnapshot automatically.
- If you cannot find the expected element, scroll or request a new snapshot.
- When the task is complete, call the 'done' tool with a summary.
- Be precise and methodical. Verify your actions produced the expected result.
- Never hallucinate target_ids. Only use IDs from the provided snapshot."""

    def build_context_message(self, snapshot: dict | None) -> str:
        """Monta o contexto do estado atual (§7.3)."""
        parts = []

        if self.session_memory.goal:
            parts.append(f"## Current Goal\n{self.session_memory.goal}")

        if self.session_memory.completed_steps:
            steps_str = "\n".join(f"- {s}" for s in self.session_memory.completed_steps[-5:])
            parts.append(f"## Recent Progress\n{steps_str}")

        if self.step_memory.last_error:
            parts.append(f"## Last Error\n{self.step_memory.last_error}")

        if snapshot:
            snap_summary = self._summarize_snapshot(snapshot)
            parts.append(f"## Current Page State\n{snap_summary}")

        return "\n\n".join(parts)

    def _summarize_snapshot(self, snapshot: dict) -> str:
        """Resumo do snapshot para o contexto do LLM."""
        lines = [
            f"URL: {snapshot.get('top_url', 'unknown')}",
            f"Title: {snapshot.get('title', 'unknown')}",
            f"Viewport: {snapshot.get('viewport', {})}",
            f"Scroll: {snapshot.get('scroll', {})}",
        ]

        for frame in snapshot.get("frames", []):
            lines.append(f"\n### Frame: {frame.get('frame_id', 'unknown')}")
            for el in frame.get("elements", []):
                if not el.get("visible", False):
                    continue
                parts = [
                    f"  [{el['target_id']}]",
                    f"<{el.get('tag', '?')}>",
                ]
                if el.get("role"):
                    parts.append(f"role={el['role']}")
                if el.get("name"):
                    parts.append(f'name="{el["name"]}"')
                text = el.get("text", "").strip()
                if text:
                    parts.append(f'"{text[:60]}"')
                if not el.get("enabled", True):
                    parts.append("[disabled]")
                lines.append(" ".join(parts))

        blocked = snapshot.get("blocked_regions", [])
        if blocked:
            lines.append(f"\n⚠️ Blocked regions: {len(blocked)} (closed shadow DOM)")

        return "\n".join(lines)

    async def think(self, snapshot: dict | None) -> ChatResult:
        """Executa um passo de planejamento."""
        if not self.message_history:
            self.message_history.append(
                ChatMessage(role="system", content=self.build_system_prompt())
            )

        context = self.build_context_message(snapshot)
        self.message_history.append(
            ChatMessage(role="user", content=context)
        )

        result = await self.provider.chat(
            messages=self.message_history,
            model=self.model,
            tools=MICEAGENT_TOOLS,
        )

        # Adicionar resposta do assistente ao histórico
        self.message_history.append(
            ChatMessage(
                role="assistant",
                content=result.content or "",
                tool_calls=result.tool_calls if result.tool_calls else None,
            )
        )

        return result

    def record_tool_result(self, tool_call_id: str, result: dict):
        """Registra resultado de uma tool no histórico."""
        self.message_history.append(
            ChatMessage(
                role="tool",
                content=str(result),
                tool_call_id=tool_call_id,
            )
        )

    def set_goal(self, goal: str):
        self.session_memory.goal = goal
        self.step_memory.last_goal = goal
        self.message_history.clear()  # reset para novo objetivo

    def record_step(self, description: str):
        self.session_memory.completed_steps.append(description)
