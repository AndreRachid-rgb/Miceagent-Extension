from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from src import main


def test_upload_capabilities_returns_limits() -> None:
    with TestClient(main.app) as client:
        response = client.get("/uploads/capabilities")

    assert response.status_code == 200
    body = response.json()
    assert "allowed_media_types" in body
    assert body["max_upload_bytes"] == main.MAX_UPLOAD_BYTES
    assert body["max_image_bytes"] == main.MAX_IMAGE_BYTES
    assert body["max_inline_image_bytes"] == main.MAX_INLINE_IMAGE_BYTES
    assert body["max_attachments_per_goal"] == main.MAX_ATTACHMENTS_PER_GOAL


def test_upload_rejects_unsupported_media_type(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path)
    main.uploaded_attachments.clear()

    with TestClient(main.app) as client:
        response = client.post(
            "/uploads",
            files={"file": ("script.exe", b"MZ...", "application/x-msdownload")},
        )

    assert response.status_code == 415
    assert "nao suportado" in response.json()["detail"]


def test_upload_accepts_text_file_and_returns_hash(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path)
    main.uploaded_attachments.clear()

    with TestClient(main.app) as client:
        response = client.post(
            "/uploads",
            files={"file": ("notes.txt", b"hello miceagent", "text/plain")},
        )

    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "uploaded"
    assert body["filename"] == "notes.txt"
    assert body["media_type"] == "text/plain"
    assert body["size"] == len(b"hello miceagent")
    assert len(body["sha256"]) == 64
    assert body["preview"]["kind"] == "text"
    assert "hello miceagent" in body["preview"]["text_excerpt"]

    attachment_id = body["attachment_id"]
    assert attachment_id in main.uploaded_attachments
    stored = cast(dict[str, Any], main.uploaded_attachments[attachment_id])
    assert Path(str(stored["path"])).exists()

    main.uploaded_attachments.clear()
