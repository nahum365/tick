from __future__ import annotations

import stat
from datetime import UTC, datetime

from tick.agents import Provider
from tick.chat import CHAT_FRAME, ChatSession, stream_turn

AT = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)


def test_frame_is_structure_only_and_carries_no_defaults_or_advice():
    lowered = CHAT_FRAME.lower()
    for forbidden in ("should", "recommend", "advice", "default", "buy", "sell"):
        assert forbidden not in lowered


def test_prose_with_a_number_is_delivered_but_never_marked_as_tool_evidence(tmp_path):
    session = ChatSession.create(tmp_path, provider=Provider.CODEX, model=None, at=AT)

    chunks = stream_turn(
        session,
        "What happened?",
        at=AT,
        adapter=lambda _transcript, _frame: (
            {"kind": "text", "text": "The answer mentions 999 in prose."},
            {"kind": "done", "model": "fixture-model"},
        ),
    )

    assert chunks[0]["kind"] == "text"
    assert "999" in chunks[0]["text"]
    assert [turn.kind for turn in session.turns()] == ["user", "text", "done"]
    assert stat.S_IMODE(session.transcript_path.stat().st_mode) == 0o600


def test_delete_removes_the_private_transcript(tmp_path):
    session = ChatSession.create(tmp_path, provider=Provider.CODEX, model=None, at=AT)
    session.delete()

    assert not session.directory.exists()
