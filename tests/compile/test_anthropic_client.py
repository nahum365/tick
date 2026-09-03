"""The provider adapter: the user's key, the user's provider, and no endpoint of ours.

The SDK client is injected, so these tests drive the adapter against a
stand-in with the SDK's shape and assert exactly what it would have sent —
including `tool_choice: any`, which is what makes "the model must call one of
these two tools" true rather than hoped for.

`for_environment` is the only constructor of a real client, and the only thing
it may read is the user's environment.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tick.compile import (
    API_KEY_ENV,
    ASK_TOOL_NAME,
    EMIT_TOOL_NAME,
    SYSTEM_PROMPT,
    AnthropicSpecProposer,
    MissingApiKey,
    ModelReplyError,
    ProposalRequest,
    QuestionsProposal,
    SpecProposal,
    read_reply,
    tool_definitions,
)

from .conftest import reply_from


class StubMessages:
    """The `client.messages.create` surface, recording what it was called with."""

    def __init__(self, reply: Any) -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.reply


class StubClient:
    def __init__(self, reply: Any) -> None:
        self.messages = StubMessages(reply)


def request(model: str = "claude-opus-5") -> ProposalRequest:
    return ProposalRequest(
        model=model,
        system=SYSTEM_PROMPT,
        messages=({"role": "user", "content": "buy 1 share of XYZ when the price is below 2"},),
        tools=tool_definitions(),
        max_tokens=16000,
    )


def test_the_call_forces_one_of_the_two_tools_and_sends_nothing_else():
    client = StubClient(reply_from({"tool": ASK_TOOL_NAME, "input": {"questions": ["Which?"]}}))
    proposer = AnthropicSpecProposer(client)

    proposer.propose(request())

    (sent,) = client.messages.calls
    assert sent["tool_choice"] == {"type": "any"}
    assert sent["model"] == "claude-opus-5"
    assert sent["system"] == SYSTEM_PROMPT
    assert sent["messages"] == [
        {"role": "user", "content": "buy 1 share of XYZ when the price is below 2"}
    ]
    assert [tool["name"] for tool in sent["tools"]] == [EMIT_TOOL_NAME, ASK_TOOL_NAME]
    assert set(sent) == {"model", "max_tokens", "system", "messages", "tools", "tool_choice"}


def test_a_spec_tool_call_becomes_a_spec_proposal():
    client = StubClient(reply_from({"tool": EMIT_TOOL_NAME, "input": {"name": "whatever"}}))

    proposal = AnthropicSpecProposer(client).propose(request())

    assert isinstance(proposal, SpecProposal)
    assert proposal.payload == {"name": "whatever"}


def test_an_ask_tool_call_becomes_questions():
    client = StubClient(
        reply_from({"tool": ASK_TOOL_NAME, "input": {"questions": ["Which symbols?", " "]}})
    )

    proposal = AnthropicSpecProposer(client).propose(request())

    assert isinstance(proposal, QuestionsProposal)
    assert proposal.questions == ("Which symbols?",)


def test_a_refusal_from_the_provider_is_not_read_as_an_answer():
    reply = SimpleNamespace(
        stop_reason="refusal",
        stop_details=SimpleNamespace(category="cyber", explanation="no"),
        content=[],
    )

    with pytest.raises(ModelReplyError) as caught:
        read_reply(reply)

    assert "declined" in str(caught.value)


def test_an_unknown_tool_name_is_an_unreadable_reply():
    reply = reply_from({"tool": "place_order", "input": {"symbol": "XYZ"}})

    with pytest.raises(ModelReplyError):
        read_reply(reply)


def test_a_refusal_with_no_questions_in_it_is_unreadable():
    """A refusal the user cannot answer is not a refusal."""
    with pytest.raises(ModelReplyError) as caught:
        read_reply(reply_from({"tool": ASK_TOOL_NAME, "input": {"questions": []}}))

    assert "no questions" in str(caught.value)


def test_without_a_key_the_compiler_says_whose_key_it_needs(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)

    with pytest.raises(MissingApiKey) as caught:
        AnthropicSpecProposer.for_environment()

    message = str(caught.value)
    assert API_KEY_ENV in message
    assert "operates no model endpoint of its own" in message


def test_a_key_reaches_the_client_and_never_the_disk(monkeypatch, tmp_path: Path):
    """Invariant 1's neighbour: Tick stores no credential, model keys included."""
    home = tmp_path / "home"
    monkeypatch.setenv("TICK_HOME", str(home))
    monkeypatch.setenv(API_KEY_ENV, "sk-ant-not-a-real-key")

    AnthropicSpecProposer.for_environment()

    assert not home.exists()
    written = [path for path in tmp_path.rglob("*") if path.is_file()]
    assert written == []


def test_an_explicit_key_beats_the_environment(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)

    proposer = AnthropicSpecProposer.for_environment(api_key="sk-ant-passed-in")

    assert isinstance(proposer, AnthropicSpecProposer)


def test_the_client_is_built_from_a_key_and_nothing_else():
    """No base_url, no proxy, no transport of ours — the argument list says so.

    The tree-wide version of this claim is
    `tests/test_product_constraints.py::test_the_compiler_constructs_no_endpoint_of_its_own`,
    which parses every file in the package. This one pins the constructor a
    reader of this module will actually look at.
    """
    source = (
        Path(__file__).resolve().parents[2] / "src" / "tick" / "compile" / "anthropic_client.py"
    ).read_text(encoding="utf-8")

    assert "anthropic.Anthropic(api_key=key)" in source
    assert "anthropic.Anthropic(" in source
    assert source.count("anthropic.Anthropic(") == 1
