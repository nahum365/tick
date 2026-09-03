"""Fixtures for the compiler tests: a fake provider, and no network at all.

`FakeAnthropic` replays a recorded exchange from `tests/fixtures/compile`. Each
fixture holds the request summary that was sent and the response that came
back, so a test asserts against a real shape rather than an invented one, and
replaying it checks the request the compiler builds today still matches the one
that produced the recorded answer.

The fake reads its recorded replies through the ADAPTER's own `read_reply`, so
the way a provider reply becomes a proposal is exercised by every fixture test
rather than re-implemented here — a fake with its own parser would be testing
itself.

`no_network` is autouse for the whole package: `socket.socket` raises, so a
test that reached for a provider would fail loudly instead of quietly billing
somebody's account. Slice 05 has no live path and this is how it stays true.
"""

from __future__ import annotations

import json
import socket
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tick.compile import SYSTEM_PROMPT, Proposal, ProposalRequest, read_reply

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "compile"

#: The marker a fixture uses for the system prompt, which is a constant in the
#: product rather than something a request varies.
SYSTEM_MARKER = "<SYSTEM_PROMPT>"


class NetworkUsedInATest(RuntimeError):
    """A test tried to open a socket. Nothing in Tick's tests may."""


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any socket in this package's tests an immediate, loud failure."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise NetworkUsedInATest(
            "a test opened a socket. The compiler's tests replay recorded "
            "fixtures; nothing here may reach a provider."
        )

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def load_fixture(name: str) -> dict[str, Any]:
    """One recorded exchange, by file name."""
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def reply_from(recorded: Mapping[str, Any]) -> Any:
    """Rebuild a provider reply object from its recorded form.

    Shaped like the SDK's: `.stop_reason`, `.stop_details`, and `.content` of
    blocks with `.type`, `.name` and `.input`.
    """
    blocks: list[Any] = []
    if recorded.get("text"):
        blocks.append(SimpleNamespace(type="text", text=recorded["text"]))
    if recorded.get("tool"):
        blocks.append(
            SimpleNamespace(
                type="tool_use",
                name=recorded["tool"],
                input=recorded.get("input", {}),
            )
        )
    details = recorded.get("stop_details")
    return SimpleNamespace(
        stop_reason=recorded.get("stop_reason", "tool_use" if recorded.get("tool") else "end_turn"),
        stop_details=SimpleNamespace(**details) if details else None,
        content=blocks,
    )


class FakeAnthropic:
    """A `SpecProposer` that replays one recorded fixture, in order.

    It verifies the request summary before answering: the model, the tool names
    offered, that the system prompt is the product's constant, and the
    conversation. A compiler that started sending something else would not get
    the recorded answer.
    """

    def __init__(self, fixture: Mapping[str, Any]) -> None:
        self.fixture = fixture
        self.calls: list[ProposalRequest] = []

    @classmethod
    def replaying(cls, name: str) -> FakeAnthropic:
        return cls(load_fixture(name))

    @property
    def text(self) -> str:
        return str(self.fixture["text"])

    @property
    def model(self) -> str:
        return str(self.fixture["model"])

    def propose(self, request: ProposalRequest) -> Proposal:
        index = len(self.calls)
        recorded = self.fixture["calls"]
        assert index < len(recorded), (
            f"the compiler made call {index + 1} but {self.fixture['name']} recorded "
            f"only {len(recorded)}"
        )
        self._check(recorded[index]["request"], request)
        self.calls.append(request)
        return read_reply(reply_from(recorded[index]["response"]))

    def _check(self, expected: Mapping[str, Any], request: ProposalRequest) -> None:
        assert request.model == expected["model"]
        assert [tool["name"] for tool in request.tools] == expected["tools"]
        assert expected["system"] == SYSTEM_MARKER
        assert request.system == SYSTEM_PROMPT
        if "messages" in expected:
            assert [dict(message) for message in request.messages] == expected["messages"]
            return
        assert len(request.messages) == expected["message_count"]
        last = dict(request.messages[-1])
        assert last["role"] == expected["last_message_role"]
        assert expected["last_message_contains"] in last["content"]


@pytest.fixture
def cross_buy() -> FakeAnthropic:
    """The one-call success: a spec whose every symbol and number is in the words."""
    return FakeAnthropic.replaying("simple-cross-buy.json")
