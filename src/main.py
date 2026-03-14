"""
miceagent — Backend Server (FastAPI)

Ponto de entrada do backend. Integra:
  - WebSocket para comunicação com a extensão
  - REST endpoints para sidebar e controle
  - Provider Layer, Planner e Executor
"""

from __future__ import annotations

import asyncio
import base64
import csv
import hashlib
import io
import json
import logging
import mimetypes
import re
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, TypedDict

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from .providers import create_provider, ChatProvider, ChatMessage
    from .planner import Planner
    from .executor import ExecutorArbiter
    from .session_manager import session_manager
except ImportError:
    try:
        from src.providers import create_provider, ChatProvider, ChatMessage
        from src.planner import Planner
        from src.executor import ExecutorArbiter
        from src.session_manager import session_manager
    except ModuleNotFoundError:
        from providers import create_provider, ChatProvider, ChatMessage  # type: ignore
        from planner import Planner  # type: ignore
        from executor import ExecutorArbiter  # type: ignore
        from session_manager import session_manager  # type: ignore

# ── Logging ──

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("miceagent")

# ── Configuration ──

DEFAULT_PROVIDER = "lm-studio"
DEFAULT_MODEL = "default"  # LM Studio auto-selects loaded model
UPLOAD_DIR = Path(__file__).resolve().parents[1] / "uploads"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_INLINE_IMAGE_BYTES = 8 * 1024 * 1024
MAX_UPLOAD_PREVIEW_IMAGE_BYTES = 512 * 1024
MAX_ATTACHMENTS_PER_GOAL = 8
ATTACHMENT_TTL_HOURS = 24
CHAT_SESSION_TTL_HOURS = 12
MAX_STORED_ATTACHMENTS = 400
MAX_TEXT_EXCERPT_CHARS = 4000
MAX_UPLOAD_PREVIEW_TEXT_CHARS = 800
ALLOWED_MEDIA_TYPES = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
    "text/plain",
    "text/markdown",
    "application/pdf",
    "application/json",
    "text/csv",
}
LLM_INLINE_NATIVE_IMAGE_MEDIA_TYPES = {
    "image/png",
    "image/jpeg",
}


class AttachmentRecord(TypedDict):
    attachment_id: str
    filename: str
    media_type: str
    size: int
    path: str
    sha256: str
    created_at: str


class ChatSessionRecord(TypedDict):
    attachment_ids: list[str]
    updated_at: str

# ── Connection Manager (P2 fix) ──

class ConnectionManager:
    def __init__(self):
        self._connections: set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self._connections.add(ws)
        logger.info(f"Extension connected. Total: {len(self._connections)}")

    def disconnect(self, ws: WebSocket):
        self._connections.discard(ws)
        logger.info(f"Extension disconnected. Total: {len(self._connections)}")

    async def broadcast(self, message: dict):
        dead = []
        for ws in self._connections:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._connections.discard(ws)

    async def send_to_first(self, message: dict):
        """Envia para a primeira conexão disponível."""
        for ws in list(self._connections):
            try:
                await ws.send_json(message)
                return True
            except Exception:
                self._connections.discard(ws)
        return False

    @property
    def count(self) -> int:
        return len(self._connections)


manager = ConnectionManager()

# ── State ──

provider: ChatProvider | None = None
planner: Planner | None = None
executor: ExecutorArbiter | None = None
agent_loop_task: asyncio.Task | None = None
uploaded_attachments: dict[str, AttachmentRecord] = {}
chat_sessions: dict[str, ChatSessionRecord] = {}
noop_scroll_streak = 0
configured_provider_type = DEFAULT_PROVIDER
configured_provider_base_url: str | None = None
configured_model = DEFAULT_MODEL

# ── Lifespan ──

@asynccontextmanager
async def lifespan(app: FastAPI):
    global provider, executor, configured_provider_type, configured_provider_base_url, configured_model
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    cleanup_expired_attachments()
    # Tentar inicializar o provider padrão
    try:
        provider = create_provider(DEFAULT_PROVIDER)
        configured_provider_type = DEFAULT_PROVIDER
        configured_provider_base_url = None
        configured_model = DEFAULT_MODEL
        status = provider.healthcheck()
        if status.available:
            logger.info(f"Provider '{status.provider_name}' available at {status.base_url}")
        else:
            logger.warning(f"Provider not available: {status.error}")
    except Exception as e:
        logger.warning(f"Could not initialize provider: {e}")

    executor = ExecutorArbiter(send_to_extension=manager.send_to_first)
    yield

app = FastAPI(title="miceagent Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Models ──

class AgentGoal(BaseModel):
    goal: str
    attachments: list[str] = Field(default_factory=list)

class ProviderConfig(BaseModel):
    provider_type: str = "lm-studio"
    base_url: str | None = None
    model: str = "default"

class ChatRequestMessage(BaseModel):
    role: str  # "user" | "assistant" | "system"
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatRequestMessage]
    attachments: list[str] = Field(default_factory=list)
    chat_session_id: str | None = None

# ── REST Endpoints ──

@app.get("/health")
def healthcheck():
    prov_status = provider.healthcheck() if provider else None
    return {
        "status": "up",
        "connections": manager.count,
        "provider": {
            "name": prov_status.provider_name if prov_status else None,
            "available": prov_status.available if prov_status else False,
            "configured_type": configured_provider_type,
            "configured_model": configured_model,
            "configured_base_url": configured_provider_base_url,
        } if prov_status else None,
        "active_session": session_manager.active_session_id,
    }


@app.get("/providers/config")
def get_provider_config():
    prov_status = provider.healthcheck() if provider else None
    return {
        "provider_type": configured_provider_type,
        "base_url": configured_provider_base_url,
        "model": configured_model,
        "available": prov_status.available if prov_status else False,
    }

@app.get("/providers/models")
def list_models():
    if not provider:
        return {"models": [], "error": "No provider configured"}
    models = provider.list_models()
    return {"models": models}

@app.post("/providers/configure")
def configure_provider(config: ProviderConfig):
    global provider, planner, configured_provider_type, configured_provider_base_url, configured_model
    try:
        provider = create_provider(config.provider_type, config.base_url)
        configured_provider_type = config.provider_type
        configured_provider_base_url = config.base_url
        configured_model = config.model
        status = provider.healthcheck()
        planner = Planner(provider=provider, model=config.model)
        return {"status": "configured", "provider": status.__dict__}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/chat/send")
async def send_chat(req: ChatRequest):
    """Modo Chat puro — sem snapshot, sem tools, sem agent loop."""
    if not provider:
        raise HTTPException(status_code=503, detail="No provider configured.")

    cleanup_expired_chat_sessions()

    messages_out = [ChatMessage(role=m.role, content=m.content) for m in req.messages]

    merged_attachment_ids = merge_attachment_ids(req.chat_session_id, req.attachments)

    # Resolve attachments and inject into the last user message as multimodal content
    if merged_attachment_ids:
        attachment_contexts = resolve_attachment_contexts(merged_attachment_ids)
        if attachment_contexts and messages_out and messages_out[-1].role == "user":
            last_text = messages_out[-1].content if isinstance(messages_out[-1].content, str) else ""
            multimodal: list[dict] = []
            if last_text:
                multimodal.append({"type": "text", "text": last_text})
            for att in attachment_contexts:
                if att.get("image_data_url"):
                    multimodal.append({"type": "image_url", "image_url": {"url": att["image_data_url"]}})
                else:
                    summary = f"Attachment: {att.get('filename', 'unknown')} ({att.get('media_type', '')})"
                    excerpt = att.get("text_excerpt")
                    if excerpt:
                        summary += f"\nContent:\n{excerpt}"
                    multimodal.append({"type": "text", "text": summary})
            messages_out[-1] = ChatMessage(role="user", content=multimodal)

    try:
        result = await provider.chat(messages=messages_out, model=configured_model)

        thinking_content = result.thinking or extract_thinking_content(result.content or "")
        clean_content = strip_thinking_content(result.content or "")

        return {
            "status": "success",
            "message": {
                "role": "assistant",
                "content": clean_content or "Sem resposta textual.",
                "thinking": thinking_content,
            },
            "chat_session_id": req.chat_session_id,
            "attachments_used": merged_attachment_ids,
        }
    except Exception as e:
        logger.error(f"Chat send failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/planner/start")
async def start_planning(payload: AgentGoal):
    global planner, agent_loop_task, noop_scroll_streak

    if not provider:
        return {"status": "error", "error": "No provider configured. POST /providers/configure first."}

    cleanup_expired_attachments()
    noop_scroll_streak = 0

    if len(payload.attachments) > MAX_ATTACHMENTS_PER_GOAL:
        raise HTTPException(
            status_code=400,
            detail=f"Maximo de {MAX_ATTACHMENTS_PER_GOAL} anexos por objetivo.",
        )

    attachment_contexts = resolve_attachment_contexts(payload.attachments)

    # Criar sessão
    session = session_manager.create_session(payload.goal, attachments=attachment_contexts)

    # Criar/resetar planner
    planner = Planner(provider=provider, model=configured_model)
    planner.set_goal(payload.goal, attachments=attachment_contexts)

    # Solicitar snapshot inicial
    sent = await manager.send_to_first({"type": "REQUEST_SNAPSHOT", "reason": "initial"})
    if not sent:
        return {"status": "error", "error": "No extension connected"}

    logger.info(f"Session [{session.session_id}] started with goal: {payload.goal}")
    return {
        "status": "started",
        "session_id": session.session_id,
        "goal": payload.goal,
        "attachments": [item.get("attachment_id") for item in attachment_contexts],
    }


@app.post("/uploads")
async def upload_attachment(file: UploadFile = File(...)):
    cleanup_expired_attachments()

    if len(uploaded_attachments) >= MAX_STORED_ATTACHMENTS:
        raise HTTPException(status_code=503, detail="Capacidade temporaria de anexos atingida. Tente novamente.")

    filename = sanitize_filename(file.filename or "uploaded-file")
    media_type = file.content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
    media_type = media_type.lower().strip()

    if media_type not in ALLOWED_MEDIA_TYPES:
        raise HTTPException(status_code=415, detail=f"Tipo de arquivo nao suportado: {media_type}")

    payload = await file.read()
    size = len(payload)

    if size == 0:
        raise HTTPException(status_code=400, detail="Arquivo vazio nao e permitido.")
    if size > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Arquivo excede limite maximo de 25MB.")
    if media_type.startswith("image/") and size > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Imagem excede limite maximo de 10MB.")

    attachment_id = f"att_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{len(uploaded_attachments)+1:04d}"
    extension = Path(filename).suffix
    storage_path = UPLOAD_DIR / f"{attachment_id}{extension}"
    storage_path.write_bytes(payload)
    checksum = hashlib.sha256(payload).hexdigest()

    uploaded_attachments[attachment_id] = {
        "attachment_id": attachment_id,
        "filename": filename,
        "media_type": media_type,
        "size": size,
        "path": str(storage_path),
        "sha256": checksum,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    preview = build_upload_preview(media_type, payload)

    return {
        "status": "uploaded",
        "attachment_id": attachment_id,
        "filename": filename,
        "media_type": media_type,
        "size": size,
        "sha256": checksum,
        "preview": preview,
    }


@app.get("/uploads/capabilities")
def upload_capabilities():
    return {
        "allowed_media_types": sorted(ALLOWED_MEDIA_TYPES),
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "max_image_bytes": MAX_IMAGE_BYTES,
        "max_inline_image_bytes": MAX_INLINE_IMAGE_BYTES,
        "max_attachments_per_goal": MAX_ATTACHMENTS_PER_GOAL,
    }

@app.post("/planner/stop")
async def stop_planning():
    global agent_loop_task
    if agent_loop_task and not agent_loop_task.done():
        agent_loop_task.cancel()
        agent_loop_task = None
    session_manager.fail_session("Stopped by user")
    return {"status": "stopped"}

# ── WebSocket ──

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    global agent_loop_task, noop_scroll_streak
    await manager.connect(ws)
    try:
        while True:
            data = await ws.receive_text()
            message = json.loads(data)
            msg_type = message.get("type", "")

            if msg_type == "HEARTBEAT":
                await ws.send_json({"type": "PONG"})

            elif msg_type == "SNAPSHOT_RESULT":
                payload = message.get("payload")
                if payload:
                    payload = sanitize_surrogates(payload)
                    session_manager.log_snapshot(payload)
                    logger.info(f"Snapshot received: {payload.get('top_url', '?')}")

                    # Se há um planner ativo, executar um passo
                    if planner and session_manager.get_active_session():
                        if agent_loop_task and not agent_loop_task.done():
                            agent_loop_task.cancel()
                        agent_loop_task = asyncio.create_task(
                            run_agent_step(planner, payload)
                        )

            elif msg_type == "TOOL_RESULT":
                corr_id = message.get("correlation_id", "")
                result = message.get("result", {})
                tool = message.get("tool", "")
                logger.info(f"Tool result [{corr_id}]: success={result.get('success')}")

                if tool == "scroll_page":
                    moved = did_scroll_move(result.get("diff"))
                    if moved:
                        noop_scroll_streak = 0
                    else:
                        noop_scroll_streak += 1
                        if noop_scroll_streak >= 2 and planner:
                            planner.step_memory.last_error = (
                                "Repeated scroll without movement. Avoid further scroll loops and prefer direct actions."
                            )
                            await manager.broadcast({
                                "type": "AI_RESPONSE",
                                "source": "runtime",
                                "text": "Scroll repetido sem mover a pagina. Vou evitar loop e buscar acao direta.",
                                "action": "Selecionar alvo visivel",
                                "confidence": 0.78,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            })
                else:
                    noop_scroll_streak = 0

                # Registrar resnapshot
                resnapshot = message.get("resnapshot")
                if resnapshot:
                    session_manager.log_snapshot(resnapshot)

                # Registrar no planner para próximo passo
                if planner and session_manager.get_active_session():
                    # Usar lm_tool_call_id (id do LLM) para que o registro bata
                    # com o tool_calls[i].id da mensagem de assistente.
                    lm_tool_call_id = message.get("lm_tool_call_id") or corr_id
                    planner.record_tool_result(lm_tool_call_id, {**result, "tool": tool})
                    # Continuar o loop do agente com o resnapshot
                    if agent_loop_task and not agent_loop_task.done():
                        agent_loop_task.cancel()
                    if not resnapshot and tool == "scroll_page":
                        await manager.send_to_first({"type": "REQUEST_SNAPSHOT", "reason": "post-scroll"})
                        continue
                    agent_loop_task = asyncio.create_task(
                        run_agent_step(planner, resnapshot)
                    )

            elif msg_type == "EVENT_LOG":
                event = message.get("event", {})
                session_manager.log_event(event)

            elif msg_type == "SNAPSHOT_ERROR":
                err_msg = message.get("error", "erro desconhecido")
                logger.error(f"Snapshot error: {err_msg}")
                await manager.broadcast({
                    "type": "AI_RESPONSE",
                    "source": "runtime",
                    "text": f"Nao foi possivel capturar o estado da pagina: {err_msg}",
                    "action": "Verificar se ha uma aba de navegador aberta com conteudo",
                    "confidence": 0.0,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(ws)


async def run_agent_step(active_planner: Planner, snapshot: dict | None):
    """Executa um passo do agente: planner pensa → executor despacha."""
    try:
        result = await active_planner.think(snapshot)

        if result.content:
            # Priorizar reasoning_content nativo da API; fallback para regex
            thinking_content = result.thinking or extract_thinking_content(result.content)
            clean_content = strip_thinking_content(result.content)
            await manager.broadcast({
                "type": "AI_RESPONSE",
                "source": "planner",
                "text": clean_content,
                "thinking": thinking_content,
                "action": extract_action_hint(clean_content),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        if result.tool_calls:
            dispatched = await executor.execute_tool_calls(result.tool_calls)
            for d in dispatched:
                if d.get("done"):
                    logger.info(f"Agent completed: {d.get('summary')}")
                    await manager.broadcast({
                        "type": "AI_RESPONSE",
                        "source": "planner",
                        "text": d.get("summary", "Tarefa concluida."),
                        "action": "Concluido",
                        "confidence": 0.95,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                    return
                logger.info(f"Dispatched tool: {d.get('tool')} [{d.get('correlation_id')}]")
        elif result.content:
            logger.info(f"Agent thought: {result.content[:200]}")
            # Se o modelo respondeu com texto mas sem tool calls, re-prompt
            active_planner.step_memory.last_evaluation = result.content
    except Exception as e:
        logger.error(f"Agent step failed: {e}")
        active_planner.step_memory.last_error = str(e)
        await manager.broadcast({
            "type": "AI_RESPONSE",
            "source": "runtime",
            "text": f"Erro no passo do agente: {e}",
            "action": "Verifique as configuracoes do provider e tente novamente",
            "confidence": 0.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })


def resolve_attachment_contexts(attachment_ids: list[str]) -> list[dict[str, Any]]:
    cleanup_expired_attachments()
    contexts: list[dict] = []
    for attachment_id in attachment_ids:
        item = uploaded_attachments.get(attachment_id)
        if not item:
            continue

        context = {
            "attachment_id": attachment_id,
            "filename": item["filename"],
            "media_type": item["media_type"],
            "size": item["size"],
        }

        try:
            payload = Path(item["path"]).read_bytes()
        except OSError:
            contexts.append(context)
            continue

        if item["media_type"].startswith("image/"):
            if len(payload) <= MAX_INLINE_IMAGE_BYTES:
                inline_image = normalize_inline_image_for_llm(item["media_type"], payload)
                if inline_image:
                    inline_media_type, inline_payload = inline_image
                    encoded = base64.b64encode(inline_payload).decode("ascii")
                    context["image_data_url"] = f"data:{inline_media_type};base64,{encoded}"
                else:
                    context["text_excerpt"] = (
                        "Image attachment available, but inline transport conversion failed; "
                        "use a PNG or JPEG image for direct visual analysis."
                    )
            else:
                context["text_excerpt"] = (
                    "Image attachment available but too large for inline context; "
                    "use website interaction tools when needed."
                )
        elif item["media_type"] == "application/pdf":
            context["text_excerpt"] = extract_pdf_excerpt(payload)
        elif item["media_type"] == "application/json":
            context["text_excerpt"] = extract_json_excerpt(payload)
        elif item["media_type"] == "text/csv":
            context["text_excerpt"] = extract_csv_excerpt(payload)
        elif item["media_type"].startswith("text/"):
            try:
                context["text_excerpt"] = payload.decode("utf-8", errors="ignore")[:MAX_TEXT_EXCERPT_CHARS]
            except Exception:
                pass
        else:
            context["text_excerpt"] = "Binary attachment available for website interaction."

        contexts.append(context)

    return contexts


def normalize_inline_image_for_llm(media_type: str, payload: bytes) -> tuple[str, bytes] | None:
    normalized_media_type = media_type.lower().strip()
    if normalized_media_type in LLM_INLINE_NATIVE_IMAGE_MEDIA_TYPES:
        return normalized_media_type, payload

    try:
        from PIL import Image  # type: ignore
    except Exception:
        logger.warning("Pillow unavailable; cannot convert %s for inline LLM transport.", media_type)
        return None

    try:
        with Image.open(io.BytesIO(payload)) as image:
            converted = image.convert("RGBA") if image.mode not in {"RGB", "RGBA"} else image.copy()
            out = io.BytesIO()
            converted.save(out, format="PNG")
            return "image/png", out.getvalue()
    except Exception as exc:
        logger.warning("Inline image conversion failed for %s: %s", media_type, exc)
        return None


def extract_action_hint(content: str) -> str:
    cleaned = strip_thinking_content(content)
    line = cleaned.strip().splitlines()[0] if cleaned.strip() else "Continuar execucao"
    return line[:120]


def strip_thinking_content(content: str) -> str:
    """Remove blocos <think>...</think> para mostrar apenas a resposta final.
    Suporta tags não fechadas (modelo cortou output)."""
    if not content:
        return ""
    try:
        return re.sub(r"<think>.*?(?:</think>|$)", "", content, flags=re.IGNORECASE | re.DOTALL).strip()
    except re.error:
        return content.strip()


def extract_thinking_content(content: str) -> str:
    """Extrai conteúdo de <think>...</think> para exibição em painel separado.
    Suporta tags não fechadas (modelo cortou output)."""
    if not content:
        return ""
    try:
        matches = re.findall(r"<think>(.*?)(?:</think>|$)", content, flags=re.IGNORECASE | re.DOTALL)
        return "\n\n".join(m.strip() for m in matches if m.strip())
    except re.error:
        return ""


def sanitize_surrogates(obj: Any) -> Any:
    """Remove lone surrogate characters from all string values in a JSON-like object.

    Pages may contain emoji or malformed text with lone surrogates (e.g. \\ud83e)
    that are not valid UTF-8 and cause encoding errors when serialised to JSON.
    """
    if isinstance(obj, str):
        return obj.encode("utf-8", errors="replace").decode("utf-8")
    if isinstance(obj, dict):
        return {k: sanitize_surrogates(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_surrogates(v) for v in obj]
    return obj


def sanitize_filename(filename: str) -> str:
    candidate = filename.replace("/", "_").replace("\\", "_")
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", candidate)
    while ".." in safe:
        safe = safe.replace("..", "_")
    safe = safe.strip("._-")
    return (safe[:120] or "uploaded_file")


def did_scroll_move(diff: str | None) -> bool:
    if not diff or "scrollY:" not in diff:
        return True
    try:
        values = diff.split("scrollY:", maxsplit=1)[1]
        before_str, after_str = values.split("->", maxsplit=1)
        return float(before_str) != float(after_str)
    except Exception:
        return True


def cleanup_expired_attachments() -> None:
    now = datetime.now(timezone.utc)
    ttl = timedelta(hours=ATTACHMENT_TTL_HOURS)
    expired_ids: list[str] = []

    for attachment_id, item in uploaded_attachments.items():
        created_at_raw = item.get("created_at")
        try:
            created_at = datetime.fromisoformat(created_at_raw)
        except Exception:
            expired_ids.append(attachment_id)
            continue

        if now - created_at > ttl:
            expired_ids.append(attachment_id)

    for attachment_id in expired_ids:
        item = uploaded_attachments.pop(attachment_id, None)
        if not item:
            continue
        path = item.get("path")
        if path:
            try:
                Path(path).unlink(missing_ok=True)
            except Exception:
                logger.warning("Failed to remove expired attachment file: %s", path)


def merge_attachment_ids(chat_session_id: str | None, request_attachment_ids: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()

    if chat_session_id:
        previous = chat_sessions.get(chat_session_id, {}).get("attachment_ids", [])
        for att_id in previous:
            if att_id not in seen:
                deduped.append(att_id)
                seen.add(att_id)

    for att_id in request_attachment_ids:
        if att_id not in seen:
            deduped.append(att_id)
            seen.add(att_id)

    if chat_session_id:
        chat_sessions[chat_session_id] = {
            "attachment_ids": deduped,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    return deduped


def cleanup_expired_chat_sessions() -> None:
    now = datetime.now(timezone.utc)
    ttl = timedelta(hours=CHAT_SESSION_TTL_HOURS)
    expired_ids: list[str] = []

    for session_id, item in chat_sessions.items():
        raw = item.get("updated_at")
        try:
            updated_at = datetime.fromisoformat(raw)
        except Exception:
            expired_ids.append(session_id)
            continue

        if now - updated_at > ttl:
            expired_ids.append(session_id)

    for session_id in expired_ids:
        chat_sessions.pop(session_id, None)


def extract_pdf_excerpt(payload: bytes) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        return "PDF attachment available. Install pypdf for text extraction support."

    try:
        reader = PdfReader(io.BytesIO(payload))
        chunks: list[str] = []
        for page in reader.pages[:6]:
            page_text = page.extract_text() or ""
            if page_text.strip():
                chunks.append(page_text.strip())
            if sum(len(c) for c in chunks) >= MAX_TEXT_EXCERPT_CHARS:
                break

        if not chunks:
            return "PDF attachment available, but no extractable text was found."
        return "\n\n".join(chunks)[:MAX_TEXT_EXCERPT_CHARS]
    except Exception:
        return "PDF attachment available, but text extraction failed."


def extract_json_excerpt(payload: bytes) -> str:
    try:
        data = json.loads(payload.decode("utf-8", errors="ignore"))
        formatted = json.dumps(data, ensure_ascii=True, indent=2)
        return formatted[:MAX_TEXT_EXCERPT_CHARS]
    except Exception:
        return payload.decode("utf-8", errors="ignore")[:MAX_TEXT_EXCERPT_CHARS]


def extract_csv_excerpt(payload: bytes) -> str:
    text = payload.decode("utf-8", errors="ignore")
    lines = [line for line in text.splitlines() if line.strip()][:60]
    if not lines:
        return ""

    reader = csv.reader(lines)
    rendered: list[str] = []
    for row in reader:
        rendered.append(" | ".join(cell.strip() for cell in row))
        if sum(len(item) for item in rendered) >= MAX_TEXT_EXCERPT_CHARS:
            break

    return "\n".join(rendered)[:MAX_TEXT_EXCERPT_CHARS]


def build_upload_preview(media_type: str, payload: bytes) -> dict[str, Any] | None:
    preview: dict[str, Any] = {
        "kind": "none",
    }

    if media_type.startswith("image/"):
        if len(payload) <= MAX_UPLOAD_PREVIEW_IMAGE_BYTES:
            preview["kind"] = "image"
            preview["image_data_url"] = (
                f"data:{media_type};base64,{base64.b64encode(payload).decode('ascii')}"
            )
            return preview
        preview["kind"] = "image-meta"
        preview["text_excerpt"] = "Image too large for inline preview."
        return preview

    if media_type == "application/pdf":
        preview["kind"] = "text"
        preview["text_excerpt"] = extract_pdf_excerpt(payload)[:MAX_UPLOAD_PREVIEW_TEXT_CHARS]
        return preview

    if media_type == "application/json":
        preview["kind"] = "text"
        preview["text_excerpt"] = extract_json_excerpt(payload)[:MAX_UPLOAD_PREVIEW_TEXT_CHARS]
        return preview

    if media_type == "text/csv":
        preview["kind"] = "text"
        preview["text_excerpt"] = extract_csv_excerpt(payload)[:MAX_UPLOAD_PREVIEW_TEXT_CHARS]
        return preview

    if media_type.startswith("text/"):
        preview["kind"] = "text"
        preview["text_excerpt"] = payload.decode("utf-8", errors="ignore")[:MAX_UPLOAD_PREVIEW_TEXT_CHARS]
        return preview

    preview["kind"] = "binary"
    preview["text_excerpt"] = "Binary file uploaded."
    return preview


# ── Entry point ──

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="127.0.0.1", port=8000, reload=True)
