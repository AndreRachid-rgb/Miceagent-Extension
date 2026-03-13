from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from src import main


def test_sanitize_filename_blocks_path_and_symbols() -> None:
    sanitized = main.sanitize_filename("..\\..//evil<>name?.txt")
    assert ".." not in sanitized
    assert "/" not in sanitized
    assert "\\" not in sanitized
    assert "<" not in sanitized
    assert sanitized.endswith(".txt")


def test_did_scroll_move_detects_noop_and_movement() -> None:
    assert main.did_scroll_move("scrollY:100->100") is False
    assert main.did_scroll_move("scrollY:100->250") is True
    assert main.did_scroll_move(None) is True


def test_cleanup_expired_attachments_removes_old_entries(tmp_path: Path) -> None:
    old_file = tmp_path / "old.txt"
    old_file.write_text("old", encoding="utf-8")

    fresh_file = tmp_path / "fresh.txt"
    fresh_file.write_text("fresh", encoding="utf-8")

    main.uploaded_attachments.clear()

    old_id = "att_old"
    fresh_id = "att_fresh"

    main.uploaded_attachments[old_id] = {
        "attachment_id": old_id,
        "filename": "old.txt",
        "media_type": "text/plain",
        "size": old_file.stat().st_size,
        "path": str(old_file),
        "sha256": "x" * 64,
        "created_at": (datetime.now(timezone.utc) - timedelta(hours=main.ATTACHMENT_TTL_HOURS + 1)).isoformat(),
    }

    main.uploaded_attachments[fresh_id] = {
        "attachment_id": fresh_id,
        "filename": "fresh.txt",
        "media_type": "text/plain",
        "size": fresh_file.stat().st_size,
        "path": str(fresh_file),
        "sha256": "y" * 64,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    main.cleanup_expired_attachments()

    assert old_id not in main.uploaded_attachments
    assert fresh_id in main.uploaded_attachments
    assert not old_file.exists()
    assert fresh_file.exists()

    # cleanup test global state
    main.uploaded_attachments.clear()
