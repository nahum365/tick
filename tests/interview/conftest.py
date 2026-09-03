"""Recorded conversations and the provider fake that replays them."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from tick.agents import StructuredReply
from tick.interview import EXTRACT_TOOL_NAME, InterviewSession

FIXTURES = Path(__file__).parent / "fixtures"


@dataclass
class FakeClient:
    replies: list[dict[str, Any]]
    model: str
    requests: list[Any] = field(default_factory=list)

    def propose(self, request):
        self.requests.append(request)
        payload = self.replies.pop(0)
        return StructuredReply(
            model=self.model,
            tool_name=EXTRACT_TOOL_NAME,
            payload=payload,
        )


def load_conversation(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def direct_payload(turn: dict[str, Any]) -> dict[str, Any]:
    return {
        "value": turn["value"],
        "quoted_span": turn["quoted_span"],
        "asked_for_suggestion": False,
        "suggestion": None,
    }


@pytest.fixture
def install_fake(monkeypatch: pytest.MonkeyPatch):
    def install(fake: FakeClient) -> FakeClient:
        monkeypatch.setattr("tick.interview.session.client_for", lambda provider: fake)
        return fake

    return install


def replay(home: Path, conversation: dict[str, Any], fake: FakeClient) -> InterviewSession:
    session = InterviewSession.create(
        home,
        provider=conversation["provider"],
        kind=conversation["kind"],
    )
    for turn in conversation["turns"]:
        session.answer(turn["answer"])
    return session
