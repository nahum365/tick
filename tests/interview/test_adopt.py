"""Adoption is the only creation step and starts the provenance record."""

from __future__ import annotations

from pathlib import Path

import pytest

from tick.interview import InterviewError, adopt
from tick.records import RecordKind, read
from tick.runtime import AgentRun

from .conftest import FakeClient, direct_payload, load_conversation, replay


def completed(home: Path, install_fake):
    conversation = load_conversation("rule.json")
    fake = install_fake(
        FakeClient(
            [direct_payload(turn) for turn in conversation["turns"]],
            conversation["model_reported"],
        )
    )
    return replay(home, conversation, fake).completed_draft


def test_adopt_creates_the_agent_and_makes_provenance_the_first_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, install_fake
):
    home = tmp_path / "home"
    monkeypatch.setenv("TICK_HOME", str(home))
    draft = completed(home, install_fake)

    run = adopt(draft.draft_id, max_cancels=2, approval=None)

    loaded = AgentRun.load(home, run.agent_id)
    assert loaded.state.approval.value == "each"
    records = list(read(loaded.ledger_path))
    assert len(records) == 1
    note = records[0]
    assert note.kind is RecordKind.NOTE
    assert note.payload["event"] == "adopted"
    assert note.payload["draft_id"] == draft.draft_id
    assert note.payload["spec_id"] == loaded.state.spec_id
    assert note.payload["provenance"] == draft.provenance
    assert note.payload["transcript_sha256"] == draft.transcript_sha256
    assert note.payload["provider"] == "codex"
    assert note.payload["model_reported"] == "gpt-test-rule"


def test_a_draft_creates_nothing_before_adoption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, install_fake
):
    home = tmp_path / "home"
    monkeypatch.setenv("TICK_HOME", str(home))
    completed(home, install_fake)
    assert AgentRun.list_ids(home) == []


def test_adopting_twice_refuses_without_adding_a_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, install_fake
):
    home = tmp_path / "home"
    monkeypatch.setenv("TICK_HOME", str(home))
    draft = completed(home, install_fake)
    run = adopt(draft.draft_id, max_cancels=2, approval=None)

    with pytest.raises(InterviewError, match="AGENT_ALREADY_EXISTS.*nothing was overwritten"):
        adopt(draft.draft_id, max_cancels=2, approval=None)
    assert len(list(read(run.ledger_path))) == 1
