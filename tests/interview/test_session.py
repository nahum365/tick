"""Conversation state, extraction provenance, and suggestion acceptance."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from tick.interview import (
    EXTRACT_TOOL_NAME,
    SUGGESTION_DISCOURAGEMENT,
    Draft,
    InterviewError,
    InterviewSession,
    meaning_bearing_fields,
)

from .conftest import FakeClient, direct_payload, load_conversation, replay


@pytest.mark.parametrize("fixture", ["rule.json", "model.json"])
def test_fixture_conversations_make_complete_all_user_drafts(
    fixture: str, tmp_path: Path, install_fake
):
    conversation = load_conversation(fixture)
    fake = install_fake(
        FakeClient(
            [direct_payload(turn) for turn in conversation["turns"]],
            conversation["model_reported"],
        )
    )

    session = replay(tmp_path / "home", conversation, fake)
    draft = session.completed_draft

    assert session.ask() is None
    assert set(draft.provenance) == meaning_bearing_fields(draft.spec)
    assert set(draft.provenance.values()) == {"user"}
    assert draft.model_reported == conversation["model_reported"]
    assert draft.to_agent() == draft.spec
    assert all(request.tools[0]["name"] == EXTRACT_TOOL_NAME for request in fake.requests)


def test_state_and_transcript_are_private(tmp_path: Path):
    session = InterviewSession.create(tmp_path / "home", provider="codex", kind="rule")
    assert stat.S_IMODE(session.directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(session.state_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(session.transcript_path.stat().st_mode) == 0o600


def test_an_untraced_extraction_is_refused_and_the_question_repeats(tmp_path: Path, install_fake):
    fake = install_fake(
        FakeClient(
            [
                {
                    "value": ["XYZ"],
                    "quoted_span": "not in the answer",
                    "asked_for_suggestion": False,
                    "suggestion": None,
                }
            ],
            "model-reported",
        )
    )
    session = InterviewSession.create(tmp_path / "home", provider="codex", kind="rule")
    question = session.ask()

    response = session.answer("Use XYZ.")

    assert "EXTRACTION_UNTRACED" in response
    assert "exact substring" in response
    assert session.ask() == question
    assert "universe" not in session.state.answers
    assert len(fake.requests) == 1


def test_an_invalid_cage_value_is_refused_and_the_question_repeats(tmp_path: Path, install_fake):
    conversation = load_conversation("rule.json")
    replies = [direct_payload(turn) for turn in conversation["turns"][:3]]
    replies.append(
        {
            "value": "0",
            "quoted_span": "zero",
            "asked_for_suggestion": False,
            "suggestion": None,
        }
    )
    install_fake(FakeClient(replies, "model-reported"))
    session = InterviewSession.create(tmp_path / "home", provider="codex", kind="rule")
    for turn in conversation["turns"][:3]:
        session.answer(turn["answer"])
    question = session.ask()

    response = session.answer("Maximum position percentage is zero.")

    assert "EXTRACTION_VALUE_INVALID" in response
    assert "greater than zero" in response
    assert session.ask() == question


def test_a_suggestion_is_discouraged_once_then_waits_for_acceptance(tmp_path: Path, install_fake):
    suggestion = {
        "value": None,
        "quoted_span": None,
        "asked_for_suggestion": True,
        "suggestion": ["XYZ"],
    }
    install_fake(FakeClient([suggestion, suggestion], "model-reported"))
    session = InterviewSession.create(tmp_path / "home", provider="codex", kind="rule")
    question = session.ask()

    first = session.answer("Can you suggest a universe?")
    second = session.answer("Please suggest a universe again.")

    assert SUGGESTION_DISCOURAGEMENT in first
    assert question in first
    assert "Type accept" in second
    assert session.ask() == question
    assert "universe" not in session.state.answers

    next_text = session.accept()
    assert next_text == session.ask()
    assert session.state.answers["universe"] == ["XYZ"]
    assert session.state.provenance["universe"] == "model:model-reported"


def test_accept_without_a_pending_suggestion_is_actionable(tmp_path: Path):
    session = InterviewSession.create(tmp_path / "home", provider="codex", kind="rule")
    with pytest.raises(InterviewError, match="NO_SUGGESTION_PENDING.*current question"):
        session.accept()


def test_a_draft_missing_provenance_refuses_with_the_field(tmp_path: Path, install_fake):
    conversation = load_conversation("rule.json")
    install_fake(
        FakeClient(
            [direct_payload(turn) for turn in conversation["turns"]],
            conversation["model_reported"],
        )
    )
    draft = replay(tmp_path / "home", conversation, None).completed_draft
    payload = draft.model_dump(mode="json")
    del payload["provenance"]["cage.max_positions"]
    incomplete = Draft.model_validate(payload)

    with pytest.raises(InterviewError, match="DRAFT_PROVENANCE_INCOMPLETE.*max_positions"):
        incomplete.to_agent()


def test_a_changed_transcript_refuses_the_completed_draft(tmp_path: Path, install_fake):
    conversation = load_conversation("rule.json")
    install_fake(
        FakeClient(
            [direct_payload(turn) for turn in conversation["turns"]],
            conversation["model_reported"],
        )
    )
    session = replay(tmp_path / "home", conversation, None)
    original = session.transcript_path.read_text(encoding="utf-8")
    session.transcript_path.write_text(
        original + '{"role":"user","content":"changed"}\n',
        encoding="utf-8",
    )

    with pytest.raises(InterviewError, match="TRANSCRIPT_HASH_MISMATCH.*new interview"):
        _ = session.completed_draft


def test_request_prompt_is_only_the_users_transcript(tmp_path: Path, install_fake):
    fake = install_fake(
        FakeClient(
            [
                {
                    "value": ["XYZ"],
                    "quoted_span": "XYZ",
                    "asked_for_suggestion": False,
                    "suggestion": None,
                }
            ],
            "model-reported",
        )
    )
    session = InterviewSession.create(tmp_path / "home", provider="codex", kind="rule")
    session.answer("Use XYZ.")

    request = fake.requests[0]
    assert request.composed_text() == "Use XYZ."
    assert session.ask() not in request.composed_text()
    assert request.tools[0]["name"] == EXTRACT_TOOL_NAME


def test_a_model_reply_for_intents_is_not_misread_as_an_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    class WrongClient:
        def propose(self, request):
            from tick.agents import ModelReply

            return ModelReply(model="model-reported", intents=())

    monkeypatch.setattr("tick.interview.session.client_for", lambda provider: WrongClient())
    session = InterviewSession.create(tmp_path / "home", provider="codex", kind="rule")
    response = session.answer("Use XYZ.")
    assert "EXTRACTION_SHAPE_INVALID" in response
    assert "different schema" in response
