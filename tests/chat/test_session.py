from __future__ import annotations

import stat
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from tick.agents import ModelReplyError, Provider, ProviderUnavailable
from tick.chat import CHAT_FRAME, ChatSession, SetupChatSession, stream_turn
from tick.serve.handlers import (
    APIError,
    chat_create,
    chat_turn,
    setup_chat_create,
    setup_chat_turn,
)

AT = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)


def test_frame_is_structure_only_and_carries_no_defaults_or_advice():
    lowered = CHAT_FRAME.lower()
    for forbidden in ("should", "recommend", "advice", "default", "buy", "sell"):
        assert forbidden not in lowered


def test_prose_with_a_number_is_delivered_but_never_marked_as_tool_evidence(tmp_path):
    session = ChatSession.create(
        tmp_path,
        provider=Provider.CODEX,
        model="fixture-model",
        codex_cli_version="0.149.0",
        at=AT,
    )

    chunks = tuple(
        stream_turn(
            session,
            "What happened?",
            at=AT,
            adapter=lambda _transcript, _frame: (
                {"kind": "text", "text": "The answer mentions 999 in prose."},
                {"kind": "done", "model": "fixture-model"},
            ),
        )
    )

    assert chunks[0]["kind"] == "text"
    assert "999" in chunks[0]["text"]
    assert [turn.kind for turn in session.turns()] == ["user", "text", "done"]
    assert stat.S_IMODE(session.transcript_path.stat().st_mode) == 0o600


def test_delete_removes_the_private_transcript(tmp_path):
    session = ChatSession.create(
        tmp_path,
        provider=Provider.CODEX,
        model="fixture-model",
        codex_cli_version="0.149.0",
        at=AT,
    )
    session.delete()

    assert not session.directory.exists()


def test_provider_failure_before_either_stream_is_a_409(tmp_path):
    def unavailable(_model):
        raise ProviderUnavailable(
            "codex could not name a model. Check the CLI login and create the chat again."
        )

    context = SimpleNamespace(
        home=tmp_path,
        now=lambda: AT,
        codex_chat_identity=unavailable,
    )
    bodies = (
        {"provider": "codex"},
        {"scope": "agent_draft", "provider": "codex"},
    )
    creators = (chat_create, setup_chat_create)

    for creator, body in zip(creators, bodies, strict=True):
        with pytest.raises(APIError) as refused:
            creator(context, body)
        assert refused.value.status == 409
        assert refused.value.code == "provider_unavailable"
        assert "Check the CLI login" in refused.value.reason


def test_provider_failure_during_both_streams_is_a_final_visible_refusal(tmp_path):
    def failed_stream():
        raise ModelReplyError(
            "codex lost its provider session. Run `codex login`, then send the turn again."
        )
        yield  # pragma: no cover - failure deliberately begins during iteration

    context = SimpleNamespace(
        home=tmp_path,
        now=lambda: AT,
        codex_chat_identity=lambda model: {
            "model": model or "fixture-model",
            "codex_cli_version": "0.149.0",
        },
        chat_adapter=lambda _provider, _model, _transcript, _frame: failed_stream(),
        setup_chat_adapter=(
            lambda _provider, _model, _transcript, _frame, _session: failed_stream()
        ),
    )
    _status, chat = chat_create(context, {"provider": "codex"})
    _status, setup = setup_chat_create(
        context,
        {"scope": "agent_draft", "provider": "codex"},
    )

    ordinary_chunks = tuple(chat_turn(context, chat["id"], {"text": "hello"}))
    setup_chunks = tuple(setup_chat_turn(context, setup["chat"]["id"], {"text": "hello"}))

    expected = {
        "kind": "refused",
        "code": "model_reply_error",
        "reason": "codex lost its provider session. Run `codex login`, then send the turn again.",
    }
    assert ordinary_chunks == (expected,)
    assert setup_chunks == (expected,)
    ordinary_turns = ChatSession(tmp_path, chat["id"]).turns()
    setup_turns = SetupChatSession(tmp_path, setup["chat"]["id"]).chat.turns()
    for turns in (ordinary_turns, setup_turns):
        assert [turn.kind for turn in turns][-2:] == ["user", "text"]
        assert turns[-1].payload["source"] == "provider"
        assert "codex login" in turns[-1].payload["text"]
