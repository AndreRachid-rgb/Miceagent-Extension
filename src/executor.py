"""
miceagent — Executor Arbiter (§3.1 / §5.3)

Responsabilidades:
  - Validar plano recebido do LLM antes de despachar
  - Aplicar políticas de retry/recover/cancel
  - Coordenar o fluxo: planner decide → executor valida e despacha → content runtime executa
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Callable, Awaitable

try:
    from .session_manager import session_manager
except ImportError:
    try:
        from src.session_manager import session_manager
    except ModuleNotFoundError:
        from session_manager import session_manager  # type: ignore


MAX_RETRIES = 2
MAX_STEPS_PER_SESSION = 50


class ExecutorArbiter:
    def __init__(self, send_to_extension: Callable[[dict], Awaitable[None]]):
        self.send_to_extension = send_to_extension

    async def execute_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Valida e despacha tool calls para a extensão."""
        results = []

        for tc in tool_calls:
            func = tc.get("function", {})
            tool_name = func.get("name", "")
            tool_call_id = tc.get("id", str(uuid.uuid4()))

            try:
                tool_args = json.loads(func.get("arguments", "{}"))
            except json.JSONDecodeError as e:
                results.append({
                    "tool_call_id": tool_call_id,
                    "success": False,
                    "error": f"Malformed tool arguments: {e}",
                })
                continue

            # Validação de segurança
            if not self._validate_tool(tool_name, tool_args):
                results.append({
                    "tool_call_id": tool_call_id,
                    "success": False,
                    "error": f"Tool '{tool_name}' rejected by arbiter",
                })
                continue

            # Tool "done" é final — não envia para extensão
            if tool_name == "done":
                session_manager.complete_session(tool_args.get("summary", ""))
                results.append({
                    "tool_call_id": tool_call_id,
                    "success": True,
                    "done": True,
                    "summary": tool_args.get("summary", ""),
                })
                continue

            # Gerar correlation_id
            correlation_id = session_manager.generate_correlation_id()
            session_manager.increment_step()

            # Despachar para extensão — incluir lm_tool_call_id (id do LLM) para
            # que o backend possa registrar corretamente o tool_result na memória.
            await self.send_to_extension({
                "type": "EXECUTE_TOOL",
                "tool": tool_name,
                "args": tool_args,
                "correlation_id": correlation_id,
                "lm_tool_call_id": tool_call_id,
            })

            results.append({
                "tool_call_id": tool_call_id,
                "correlation_id": correlation_id,
                "tool": tool_name,
                "dispatched": True,
            })

        return results

    def _validate_tool(self, tool_name: str, args: dict) -> bool:
        """Política de validação antes de despachar."""
        allowed_tools = {
            "click_target", "type_text", "select_option",
            "scroll_page", "scroll_target_into_view",
            "capture_snapshot", "hover_target", "press_key", "done",
        }
        if tool_name not in allowed_tools:
            return False

        # Limitar steps por sessão
        session = session_manager.get_active_session()
        if session and session.step_count >= MAX_STEPS_PER_SESSION:
            return False

        return True

    def should_retry(self, result: dict) -> bool:
        """Decide se uma ação falha deve ser retentada."""
        if result.get("success", False):
            return False
        error = result.get("error", "")
        # Falhas técnicas (target não encontrado) podem ser retentadas com re-snapshot
        retriable_errors = ["Target", "not found", "zero dimensions"]
        return any(e in error for e in retriable_errors)
