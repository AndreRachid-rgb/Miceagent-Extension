"""
miceagent — Backend Server (FastAPI)

Ponto de entrada do backend. Integra:
  - WebSocket para comunicação com a extensão
  - REST endpoints para sidebar e controle
  - Provider Layer, Planner e Executor
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from .providers import create_provider, ChatProvider
    from .planner import Planner
    from .executor import ExecutorArbiter
    from .session_manager import session_manager
except ImportError:
    try:
        from src.providers import create_provider, ChatProvider
        from src.planner import Planner
        from src.executor import ExecutorArbiter
        from src.session_manager import session_manager
    except ModuleNotFoundError:
        from providers import create_provider, ChatProvider  # type: ignore
        from planner import Planner  # type: ignore
        from executor import ExecutorArbiter  # type: ignore
        from session_manager import session_manager  # type: ignore

# ── Logging ──

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("miceagent")

# ── Configuration ──

DEFAULT_PROVIDER = "lm-studio"
DEFAULT_MODEL = "default"  # LM Studio auto-selects loaded model

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

# ── Lifespan ──

@asynccontextmanager
async def lifespan(app: FastAPI):
    global provider, executor
    # Tentar inicializar o provider padrão
    try:
        provider = create_provider(DEFAULT_PROVIDER)
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

class ProviderConfig(BaseModel):
    provider_type: str = "lm-studio"
    base_url: str | None = None
    model: str = "default"

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
        } if prov_status else None,
        "active_session": session_manager.active_session_id,
    }

@app.get("/providers/models")
def list_models():
    if not provider:
        return {"models": [], "error": "No provider configured"}
    models = provider.list_models()
    return {"models": models}

@app.post("/providers/configure")
def configure_provider(config: ProviderConfig):
    global provider, planner
    try:
        provider = create_provider(config.provider_type, config.base_url)
        status = provider.healthcheck()
        planner = Planner(provider=provider, model=config.model)
        return {"status": "configured", "provider": status.__dict__}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/planner/start")
async def start_planning(payload: AgentGoal):
    global planner, agent_loop_task

    if not provider:
        return {"status": "error", "error": "No provider configured. POST /providers/configure first."}

    # Criar sessão
    session = session_manager.create_session(payload.goal)

    # Criar/resetar planner
    planner = Planner(provider=provider, model=DEFAULT_MODEL)
    planner.set_goal(payload.goal)

    # Solicitar snapshot inicial
    sent = await manager.send_to_first({"type": "REQUEST_SNAPSHOT", "reason": "initial"})
    if not sent:
        return {"status": "error", "error": "No extension connected"}

    logger.info(f"Session [{session.session_id}] started with goal: {payload.goal}")
    return {"status": "started", "session_id": session.session_id, "goal": payload.goal}

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
    global agent_loop_task
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
                logger.info(f"Tool result [{corr_id}]: success={result.get('success')}")

                # Registrar resnapshot
                resnapshot = message.get("resnapshot")
                if resnapshot:
                    session_manager.log_snapshot(resnapshot)

                # Registrar no planner para próximo passo
                if planner and session_manager.get_active_session():
                    planner.record_tool_result(corr_id, result)
                    # Continuar o loop do agente com o resnapshot
                    if agent_loop_task and not agent_loop_task.done():
                        agent_loop_task.cancel()
                    agent_loop_task = asyncio.create_task(
                        run_agent_step(planner, resnapshot)
                    )

            elif msg_type == "EVENT_LOG":
                event = message.get("event", {})
                session_manager.log_event(event)

            elif msg_type == "SNAPSHOT_ERROR":
                logger.error(f"Snapshot error: {message.get('error')}")

    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(ws)


async def run_agent_step(active_planner: Planner, snapshot: dict | None):
    """Executa um passo do agente: planner pensa → executor despacha."""
    try:
        result = await active_planner.think(snapshot)

        if result.tool_calls:
            dispatched = await executor.execute_tool_calls(result.tool_calls)
            for d in dispatched:
                if d.get("done"):
                    logger.info(f"Agent completed: {d.get('summary')}")
                    return
                logger.info(f"Dispatched tool: {d.get('tool')} [{d.get('correlation_id')}]")
        elif result.content:
            logger.info(f"Agent thought: {result.content[:200]}")
            # Se o modelo respondeu com texto mas sem tool calls, re-prompt
            active_planner.step_memory.last_evaluation = result.content
    except Exception as e:
        logger.error(f"Agent step failed: {e}")
        active_planner.step_memory.last_error = str(e)


# ── Entry point ──

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="127.0.0.1", port=8000, reload=True)
