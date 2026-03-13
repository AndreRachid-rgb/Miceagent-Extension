"""
miceagent — Session Manager (§3.1 / §7.4)

Gerencia sessões, correlação de eventos, memória por tab/frame,
e artefatos de execução.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ActionEvent:
    correlation_id: str
    phase: str  # planned | started | waiting | succeeded | failed | canceled
    action_type: str
    target_id: str | None
    timestamp: str
    error: str | None = None


@dataclass
class Session:
    session_id: str
    goal: str
    created_at: str
    status: str = "active"  # active | paused | completed | failed
    snapshots: list[dict[str, Any]] = field(default_factory=list)
    events: list[ActionEvent] = field(default_factory=list)
    step_count: int = 0


class SessionManager:
    def __init__(self):
        self.sessions: dict[str, Session] = {}
        self.active_session_id: str | None = None

    def create_session(self, goal: str) -> Session:
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        session = Session(
            session_id=session_id,
            goal=goal,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self.sessions[session_id] = session
        self.active_session_id = session_id
        return session

    def get_active_session(self) -> Session | None:
        if self.active_session_id:
            return self.sessions.get(self.active_session_id)
        return None

    def log_snapshot(self, snapshot: dict[str, Any]):
        session = self.get_active_session()
        if not session:
            return
        # Manter apenas os últimos 10 snapshots em memória
        session.snapshots.append(snapshot)
        if len(session.snapshots) > 10:
            session.snapshots = session.snapshots[-10:]

    def log_event(self, event: dict[str, Any]):
        session = self.get_active_session()
        if not session:
            return
        session.events.append(ActionEvent(
            correlation_id=event.get("correlation_id", ""),
            phase=event.get("phase", ""),
            action_type=event.get("action_type", ""),
            target_id=event.get("target_id"),
            timestamp=event.get("timestamp", datetime.now(timezone.utc).isoformat()),
            error=event.get("error"),
        ))

    def get_latest_snapshot(self) -> dict[str, Any] | None:
        session = self.get_active_session()
        if session and session.snapshots:
            return session.snapshots[-1]
        return None

    def increment_step(self):
        session = self.get_active_session()
        if session:
            session.step_count += 1

    def complete_session(self, summary: str = ""):
        session = self.get_active_session()
        if session:
            session.status = "completed"
            self.active_session_id = None

    def fail_session(self, error: str = ""):
        session = self.get_active_session()
        if session:
            session.status = "failed"
            self.active_session_id = None

    def generate_correlation_id(self) -> str:
        session = self.get_active_session()
        step = session.step_count if session else 0
        return f"act_{datetime.now(timezone.utc).strftime('%Y%m%d')}_{step:04d}"


session_manager = SessionManager()
