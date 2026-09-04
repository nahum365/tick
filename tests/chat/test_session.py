from __future__ import annotations

import json
import stat
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from tick.agents import ModelReplyError, Provider, ProviderUnavailable
from tick.chat import (
    CHAT_FRAME,
    MAX_REPLAY_CHARACTERS,
    ChatError,
    ChatSession,
    SetupChatSession,
    stream_turn,
)
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


def test_large_setup_transcript_replays_under_bound_without_losing_prose_or_latest_document(
    tmp_path,
):
    session = ChatSession.create_setup(
        tmp_path,
        provider=Provider.CODEX,
        model="fixture-model",
        codex_cli_version="0.149.0",
        scope="broker_profile",
        at=AT,
    )
    contract_body = "full-contract-body-" * 25_000
    for index in range(30):
        if index % 10 == 0:
            session.append("user", {"text": f"person text {index}"}, at=AT)
            session.append("text", {"text": f"model text {index}"}, at=AT)
        session.append(
            "tool_result",
            {
                "name": "broker_draft",
                "result": {
                    f"field-{field}": f"tool-result-{index}-{field}-" * 400 for field in range(5)
                },
            },
            at=AT,
        )
    document = {
        "server": "https://broker.example.invalid/mcp",
        "inventory_hash": "sha256:inventory",
        "tools": {
            "get_quote": {
                "category": "read.quote",
                "arguments": {"symbol": "{symbol}"},
                "result": {"price": "price"},
                "warnings": [],
                "contract": {
                    "description": contract_body,
                    "contract_hash": "sha256:contract",
                },
            }
        },
    }
    session.append(
        "document",
        {
            "document": document,
            "valid": True,
            "complete": False,
            "proof": {"get_quote": {"success": True}},
        },
        at=AT,
    )

    replay = session.turns_for_replay()
    encoded = json.dumps(replay, ensure_ascii=False, sort_keys=True)

    assert len(encoded) <= MAX_REPLAY_CHARACTERS
    assert contract_body in session.transcript_path.read_text(encoding="utf-8")
    assert contract_body not in encoded
    assert [turn["text"] for turn in replay if turn["kind"] == "user"] == [
        "person text 0",
        "person text 10",
        "person text 20",
    ]
    assert [
        turn["text"] for turn in replay if turn["kind"] == "text" and turn.get("source") is None
    ] == [
        "model text 0",
        "model text 10",
        "model text 20",
    ]
    assert (
        sum(
            turn.get("text")
            == "Earlier tool results were elided; call the tool again if you need them."
            for turn in replay
        )
        == 1
    )
    latest = next(turn for turn in reversed(replay) if turn["kind"] == "document")
    assert latest["summary"]["tools"]["get_quote"] == {
        "category": "read.quote",
        "arguments": {"symbol": "{symbol}"},
        "result": {"price": "price"},
        "warnings": [],
        "finalized": False,
        "proved": True,
        "contract_hash": "sha256:contract",
    }
    assert len(latest["document_hash"]) == 64


def test_tool_result_contract_bodies_become_hash_references_and_long_strings_name_the_tool(
    tmp_path,
):
    session = ChatSession.create(
        tmp_path,
        provider=Provider.CODEX,
        model="fixture-model",
        codex_cli_version="0.149.0",
        at=AT,
    )
    session.append(
        "tool_result",
        {
            "name": "broker_contract",
            "result": {
                "contract": {
                    "contract_hash": "sha256:one",
                    "description": "hidden body",
                },
                "contracts": [{"contract_hash": "sha256:two", "description": "hidden body two"}],
                "detail": "x" * 3_000,
            },
        },
        at=AT,
    )

    result = session.turns_for_replay()[0]["result"]

    assert result["contract"] == {"contract_hash": "sha256:one"}
    assert result["contracts"] == [{"contract_hash": "sha256:two"}]
    assert len(result["detail"]) <= 2_000
    assert "call broker_contract again for the full value" in result["detail"]
    assert "hidden body" not in json.dumps(result)


def test_replay_refuses_when_unelidable_prose_alone_exceeds_the_bound(tmp_path):
    session = ChatSession.create(
        tmp_path,
        provider=Provider.CODEX,
        model="fixture-model",
        codex_cli_version="0.149.0",
        at=AT,
    )
    session.append("user", {"text": "x" * MAX_REPLAY_CHARACTERS}, at=AT)

    with pytest.raises(ChatError) as refused:
        session.turns_for_replay()

    assert refused.value.code == "CHAT_REPLAY_TOO_LARGE"
    assert "Delete this chat and start a new one" in refused.value.reason


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
        {"scope": "agent_draft", "provider": "codex", "resume": False},
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
        {"scope": "agent_draft", "provider": "codex", "resume": False},
    )

    ordinary_chunks = tuple(chat_turn(context, chat["id"], {"text": "hello"}))
    setup_chunks = tuple(setup_chat_turn(context, setup["chat"]["id"], {"text": "hello"}))

    expected = {
        "kind": "refused",
        "code": "model_reply_error",
        "reason": "codex lost its provider session. Run `codex login`, then send the turn again.",
    }
    assert ordinary_chunks == (expected,)
    assert setup_chunks[-1] == expected
    assert setup_chunks[0]["kind"] == "progress"
    ordinary_turns = ChatSession(tmp_path, chat["id"]).turns()
    setup_turns = SetupChatSession(tmp_path, setup["chat"]["id"]).chat.turns()
    assert [turn.kind for turn in ordinary_turns][-2:] == ["user", "text"]
    assert ordinary_turns[-1].payload["source"] == "provider"
    assert [turn.kind for turn in setup_turns][-4:] == ["user", "progress", "text", "refused"]
    assert setup_turns[-2].payload["source"] == "provider"
    assert "codex login" in ordinary_turns[-1].payload["text"]
    assert "codex login" in setup_turns[-2].payload["text"]


def test_restored_broker_chat_starts_from_box_draft_counts(tmp_path, monkeypatch):
    restored = {
        "proposal": {
            "tools": {
                "accounts": {"category": "read.accounts"},
                "quote": {"category": "read.quote"},
                "transfer": {"category": "denied.transfers"},
            }
        },
        "profile": {
            "tools": {
                "accounts": {
                    "category": "read.accounts",
                    "confirmed_at": AT.isoformat(),
                    "proof": {"success": True},
                },
                "quote": {
                    "category": "read.quote",
                    "confirmed_at": AT.isoformat(),
                    "proof": None,
                },
                "transfer": {
                    "category": "denied.transfers",
                    "confirmed_at": None,
                    "proof": None,
                },
            }
        },
    }
    monkeypatch.setattr("tick.serve.handlers.broker_profile", lambda _context: restored)
    context = SimpleNamespace(
        home=tmp_path,
        now=lambda: AT,
        codex_chat_identity=lambda model: {
            "model": model or "fixture-model",
            "codex_cli_version": "0.149.0",
        },
    )

    status, response = setup_chat_create(
        context,
        {"scope": "broker_profile", "provider": "codex", "resume": True},
    )

    assert status == 201
    assert response["document"] == restored["proposal"]
    assert response["valid"] is True
    assert response["complete"] is False
    assert [turn["kind"] for turn in response["transcript"]] == ["tool_result", "text"]
    assert response["transcript"][0]["payload"]["name"] == "broker_draft"
    assert response["transcript"][1]["payload"]["text"] == (
        "Restored broker draft from your box: 2 tools mapped, 2 finalized, "
        "1 awaiting proof; say what to change, or finalize."
    )


def test_setup_resume_refusals_name_the_available_next_step(tmp_path, monkeypatch):
    context = SimpleNamespace(
        home=tmp_path,
        now=lambda: AT,
        codex_chat_identity=lambda model: {
            "model": model or "fixture-model",
            "codex_cli_version": "0.149.0",
        },
    )
    with pytest.raises(APIError) as unsupported:
        setup_chat_create(
            context,
            {"scope": "agent_draft", "provider": "codex", "resume": True},
        )
    assert "Start the agent interview again" in unsupported.value.reason

    monkeypatch.setattr(
        "tick.serve.handlers.broker_profile",
        lambda _context: {"proposal": None, "profile": None},
    )
    with pytest.raises(APIError) as missing:
        setup_chat_create(
            context,
            {"scope": "broker_profile", "provider": "codex", "resume": True},
        )
    assert "Start the broker connection first" in missing.value.reason
