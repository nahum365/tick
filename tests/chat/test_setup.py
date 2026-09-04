from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from tests.runtime.conftest import build_spec
from tick.agents import Provider
from tick.broker import DiscoveredTool, propose_profile, save_proposal
from tick.broker.profile_model import proposal_reply_from_document
from tick.chat import ChatError, ChatSession, SetupChatSession, SetupScope
from tick.interview import InterviewError, InterviewSession, meaning_bearing_fields
from tick.mcpbox import BoxTools, setup_tool_definitions
from tick.serve.handlers import setup_chat_turn

NOW = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)


def _broker_tools() -> tuple[DiscoveredTool, ...]:
    schema = {"type": "object", "properties": {}, "additionalProperties": False}
    return (
        DiscoveredTool(
            name="get_quote",
            title=None,
            description="quote",
            input_schema={
                "type": "object",
                "properties": {"symbol": {"type": "string"}},
                "required": ["symbol"],
            },
            output_schema={
                "type": "object",
                "properties": {"price": {"type": "string"}},
            },
            annotations=None,
            execution=None,
        ),
        DiscoveredTool(
            name="transfer_money",
            title=None,
            description="transfer funds",
            input_schema=schema,
            output_schema=schema,
            annotations=None,
            execution=None,
        ),
    )


def _context(home, operation):
    return SimpleNamespace(
        home=home,
        now=lambda: NOW,
        broker_profile_operation=operation,
        commons_client=lambda: None,
    )


def test_broker_setup_proof_failure_becomes_a_turn_and_denial_holds(tmp_path):
    home = tmp_path / "tick-home"
    setup = SetupChatSession.create(
        home,
        scope=SetupScope.BROKER_PROFILE,
        provider=Provider.CODEX,
        model=None,
        at=NOW,
    )
    tools = _broker_tools()

    def operation(action, body):
        if action == "propose_document":
            reply = proposal_reply_from_document(body["document"], model="fixture-model")

            class Categorizer:
                version = "setup-chat-v1:fixture-model"

                def propose(self, _contracts):
                    return reply

            proposal = propose_profile(
                tools,
                server="https://broker.example.invalid/mcp",
                account_id=None,
                proposed_at=NOW,
                categorizer=Categorizer(),
            )
            save_proposal(home, proposal)
            return {
                "proposal": proposal.model_dump(mode="json"),
                "warnings": {},
                "denied": ["transfer_money"],
            }
        if action == "prove":
            literal = setup.state.document["tools"]["get_quote"]["arguments"]["hours"]
            success = literal == "regular_hours"
            return {
                "outcome": {
                    "get_quote": {
                        "success": success,
                        "detail": (
                            "proved" if success else "market_hours was refused; use regular_hours"
                        ),
                    }
                }
            }
        raise AssertionError(action)

    first = {
        "tools": [
            {
                "name": "get_quote",
                "category": "read.quote",
                "arguments": {"symbol": "{symbol}", "hours": "market_hours"},
                "result": {"price": "price"},
                "reason": "This reads one quote.",
            },
            {
                "name": "transfer_money",
                "category": "order.place",
                "arguments": {},
                "result": {},
                "reason": "This row must remain denied.",
            },
        ]
    }
    second = {
        "tools": [
            {
                **first["tools"][0],
                "arguments": {"symbol": "{symbol}", "hours": "regular_hours"},
            },
            first["tools"][1],
        ]
    }
    context = _context(home, operation)

    class FakeSetupChatClient:
        """Replay provider tool choices while the real box owns every verdict."""

        def turn(self, _provider, _model, transcript, _frame, session_id):
            box = BoxTools(context, setup_session_id=session_id)
            text = str(transcript[-1].payload["text"])
            if text == "Propose the first fixture.":
                name, arguments = "propose_broker_profile", {"document": first}
            elif text == "Prove it.":
                name, arguments = "prove_broker_draft", {"probe": {"symbol": "XYZ"}}
            else:
                name, arguments = "propose_broker_profile", {"document": second}
            result = box.call(name, arguments)
            return (
                {"kind": "tool_call", "name": name, "arguments": arguments},
                {
                    "kind": "tool_result",
                    "name": name,
                    "result": result,
                    "evidence": result["evidence"],
                },
                {"kind": "text", "text": "The box verdict is in the document panel."},
                {"kind": "done", "model": "fixture-model"},
            )

    context.setup_chat_adapter = FakeSetupChatClient().turn
    setup_chat_turn(context, setup.chat.session_id, {"text": "Propose the first fixture."})
    assert setup.state.valid is True
    assert setup.state.document["tools"]["transfer_money"]["category"].startswith("denied.")

    setup_chat_turn(context, setup.chat.session_id, {"text": "Prove it."})
    assert setup.state.valid is False
    proof_turn = next(
        turn
        for turn in reversed(setup.chat.turns())
        if turn.kind == "tool_result" and turn.payload["name"] == "prove_broker_draft"
    )
    assert "regular_hours" in str(proof_turn.payload["result"])

    setup_chat_turn(
        context,
        setup.chat.session_id,
        {"text": "Use regular_hours and emit the complete document again."},
    )
    assert setup.state.valid is True
    assert setup.state.document["tools"]["transfer_money"]["category"].startswith("denied.")


def test_agent_setup_refuses_missing_provenance_until_complete(tmp_path):
    home = tmp_path / "tick-home"
    setup = SetupChatSession.create(
        home,
        scope=SetupScope.AGENT_DRAFT,
        provider=Provider.CODEX,
        model=None,
        at=NOW,
    )
    setup.chat.append("user", {"text": "Use my complete rule document."}, at=NOW)
    box = BoxTools(
        _context(home, lambda _action, _body: {}),
        setup_session_id=setup.chat.session_id,
    )
    spec = build_spec()
    document = {
        "spec": spec.model_dump(mode="json"),
        "instructions": None,
        "approval": "each",
        "provenance": {"kind": "user"},
        "model_reported": "fixture-model",
    }

    refused = box.propose_agent_draft(document)

    assert refused["valid"] is False
    assert refused["code"] == "DRAFT_PROVENANCE_INCOMPLETE"
    assert "answer" in refused["reason"].lower()
    with pytest.raises(InterviewError) as adoption:
        InterviewSession(home, setup.chat.session_id).completed_draft.to_agent()
    assert "before adopting" in adoption.value.reason

    document["provenance"] = {field: "user" for field in meaning_bearing_fields(spec)}
    accepted = box.propose_agent_draft(document)
    assert accepted["valid"] is True
    assert setup.state.valid is True
    assert InterviewSession(home, setup.chat.session_id).completed_draft.to_agent() == spec


def test_setup_scopes_expose_disjoint_closed_tool_sets():
    broker = {tool["name"] for tool in setup_tool_definitions(SetupScope.BROKER_PROFILE)}
    agent = {tool["name"] for tool in setup_tool_definitions(SetupScope.AGENT_DRAFT)}
    assert broker == {
        "broker_inventory",
        "broker_draft",
        "propose_broker_profile",
        "prove_broker_draft",
        "broker_accounts",
    }
    assert agent == {
        "interview_script",
        "agent_draft",
        "propose_agent_draft",
        "commons_pass",
    }
    assert broker.isdisjoint(agent)


def test_setup_delete_refuses_an_ordinary_chat_id(tmp_path):
    chat = ChatSession.create(tmp_path / "tick-home", provider=Provider.CODEX, model=None, at=NOW)

    with pytest.raises(ChatError) as refused:
        SetupChatSession(chat.home, chat.session_id).delete()

    assert "Start it again" in refused.value.reason
    assert chat.metadata_path.exists()
