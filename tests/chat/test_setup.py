from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from tests.runtime.conftest import build_spec
from tick.agents import Provider
from tick.broker import DiscoveredTool, propose_profile, save_proposal
from tick.broker.profile_model import proposal_reply_from_document
from tick.chat import MAX_SETUP_MODEL_TURNS, ChatError, ChatSession, SetupChatSession, SetupScope
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


def test_broker_setup_repairs_proof_failure_without_another_person_turn(tmp_path):
    home = tmp_path / "tick-home"
    setup = SetupChatSession.create(
        home,
        scope=SetupScope.BROKER_PROFILE,
        provider=Provider.CODEX,
        model="fixture-model",
        codex_cli_version="0.149.0",
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
        if action == "prove_draft":
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

        calls = 0

        def turn(self, _provider, _model, transcript, _frame, session_id, _thread=None):
            self.calls += 1
            box = BoxTools(context, setup_session_id=session_id)
            text = str(
                next(turn["text"] for turn in reversed(transcript) if turn["kind"] == "user")
            )
            if text == "Propose the first fixture." and self.calls == 1:
                name, arguments = "propose_broker_profile", {"document": first}
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

    client = FakeSetupChatClient()
    context.setup_chat_adapter = client.turn
    chunks = tuple(
        setup_chat_turn(context, setup.chat.session_id, {"text": "Propose the first fixture."})
    )

    assert client.calls == 2
    assert setup.state.valid is True
    assert setup.state.complete is True
    assert setup.state.document["tools"]["transfer_money"]["category"].startswith("denied.")
    streamed_documents = [chunk for chunk in chunks if chunk["kind"] == "document"]
    assert all("document" not in chunk for chunk in streamed_documents)
    assert all("summary" in chunk and "document_hash" in chunk for chunk in streamed_documents)
    stored_documents = [turn for turn in setup.chat.turns() if turn.kind == "document"]
    assert stored_documents[-1].payload["document"] == setup.state.document
    assert [chunk.get("step") for chunk in chunks if chunk["kind"] == "progress"] == [
        "proposing",
        "checking",
        "proving",
        "proposing",
        "checking",
        "proving",
        "valid",
    ]
    assert chunks[-1]["text"] == (
        "The profile is complete: 1 tools mapped, 1 proved. Review it and finalize."
    )


def test_agent_setup_refuses_missing_provenance_until_complete(tmp_path):
    home = tmp_path / "tick-home"
    setup = SetupChatSession.create(
        home,
        scope=SetupScope.AGENT_DRAFT,
        provider=Provider.CODEX,
        model="fixture-model",
        codex_cli_version="0.149.0",
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


def test_setup_loop_stops_at_the_model_turn_bound_with_an_actionable_sentence(tmp_path):
    setup = SetupChatSession.create(
        tmp_path,
        scope=SetupScope.BROKER_PROFILE,
        provider=Provider.CODEX,
        model="fixture-model",
        codex_cli_version="0.149.0",
        at=NOW,
    )
    calls = 0

    def adapter(_provider, _model, _transcript, _frame, _session_id, _thread=None):
        nonlocal calls
        calls += 1
        return ({"kind": "text", "text": "I am still reading the inventory."},)

    context = _context(tmp_path, lambda _action, _body: {})
    context.setup_chat_adapter = adapter
    chunks = tuple(setup_chat_turn(context, setup.chat.session_id, {"text": "Continue."}))

    assert calls == MAX_SETUP_MODEL_TURNS
    assert chunks[-2]["step"] == "stopped"
    assert "edit the document, answer the requested values, or retry" in chunks[-1]["text"]
    assert setup.state.complete is False


def test_setup_loop_parks_for_a_probe_value_and_resumes_from_the_answer(tmp_path):
    setup = SetupChatSession.create(
        tmp_path,
        scope=SetupScope.BROKER_PROFILE,
        provider=Provider.CODEX,
        model="fixture-model",
        codex_cli_version="0.149.0",
        at=NOW,
    )
    tools = _broker_tools()
    document = {
        "tools": [
            {
                "name": "get_quote",
                "category": "read.quote",
                "arguments": {"symbol": "{symbol}"},
                "result": {"price": "price"},
                "reason": "This reads one quote.",
            },
            {
                "name": "transfer_money",
                "category": None,
                "arguments": {},
                "result": {},
                "reason": "This is not used.",
            },
        ]
    }

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
            save_proposal(tmp_path, proposal)
            return {
                "proposal": proposal.model_dump(mode="json"),
                "warnings": {},
                "denied": ["transfer_money"],
            }
        if action == "prove_draft":
            if body["probe"].get("symbol") != "XYZ":
                return {
                    "outcome": {
                        "get_quote": {
                            "success": False,
                            "unresolved": {"needs": "symbol"},
                            "detail": "this tool needs probe values for: symbol. Supply it.",
                        }
                    }
                }
            return {
                "outcome": {
                    "get_quote": {
                        "success": True,
                        "unresolved": {},
                        "detail": "proved",
                    }
                }
            }
        raise AssertionError(action)

    context = _context(tmp_path, operation)

    calls = 0

    def adapter(_provider, _model, transcript, _frame, session_id, _thread=None):
        nonlocal calls
        calls += 1
        box = BoxTools(context, setup_session_id=session_id)
        user = next(turn["text"] for turn in reversed(transcript) if turn["kind"] == "user")
        if user == "Start.":
            if calls == 1:
                name = "propose_broker_profile"
                arguments = {"document": document}
                text = "I submitted the complete profile for checking."
            else:
                return (
                    {
                        "kind": "text",
                        "text": "Which symbol should proof read? The quote mapping needs it.",
                    },
                )
        else:
            name = "prove_broker_draft"
            arguments = {"probe": {"symbol": "XYZ"}}
            text = "The box can check the supplied value."
        result = box.call(name, arguments)
        return (
            {"kind": "tool_call", "name": name, "arguments": arguments},
            {"kind": "tool_result", "name": name, "result": result, "evidence": ["checked"]},
            {"kind": "text", "text": text},
        )

    context.setup_chat_adapter = adapter
    first = tuple(setup_chat_turn(context, setup.chat.session_id, {"text": "Start."}))
    assert calls == 2
    assert first[-1]["step"] == "waiting_for_person"
    assert setup.state.waiting_for == ("symbol",)
    assert setup.state.complete is False

    second = tuple(setup_chat_turn(context, setup.chat.session_id, {"text": "Use XYZ."}))
    assert second[-1]["text"].endswith("Review it and finalize.")
    assert setup.state.complete is True
    assert setup.state.probe_values == {"symbol": "XYZ"}


def test_broker_setup_default_reads_are_compact_and_contract_is_on_demand(tmp_path, monkeypatch):
    setup = SetupChatSession.create(
        tmp_path,
        scope=SetupScope.BROKER_PROFILE,
        provider=Provider.CODEX,
        model="fixture-model",
        codex_cli_version="0.149.0",
        at=NOW,
    )
    marker = "contract-body-marker-" * 500
    contract = {
        "name": "get_quote",
        "description": marker,
        "input_schema": {"description": marker},
        "output_schema": {"description": marker},
        "shape_hash": "sha256:shape",
        "contract_hash": "sha256:contract",
    }
    proposal = {
        "server": "https://broker.example.invalid/mcp",
        "inventory_hash": "sha256:inventory",
        "tools": {
            "get_quote": {
                "contract": contract,
                "category": "read.quote",
                "arguments": {"symbol": "{symbol}"},
                "result": {"price": "price"},
                "warnings": [],
            }
        },
    }
    setup.save(
        document=proposal,
        valid=True,
        complete=False,
        waiting_for=(),
        probe_values={},
        proof={},
        verdict={"code": "BROKER_DOCUMENT_VALID", "reason": "checked"},
        at=NOW,
    )

    def operation(action, body):
        if action == "inventory":
            return {
                "server_url": proposal["server"],
                "inventory_hash": proposal["inventory_hash"],
                "contracts": [contract],
            }
        if action == "contract":
            assert body == {"name": "get_quote"}
            return {"contract": contract, "evidence": ["display_only"]}
        raise AssertionError(action)

    monkeypatch.setattr(
        "tick.serve.handlers.broker_profile",
        lambda _context: {"proposal": proposal, "profile": None},
    )
    box = BoxTools(_context(tmp_path, operation), setup_session_id=setup.chat.session_id)
    inventory = box.broker_inventory()
    draft = box.broker_draft()

    assert marker not in json.dumps(inventory)
    assert inventory["tools"][0]["category_hint"] == "read.quote"
    assert marker not in json.dumps(draft)
    assert len(json.dumps(inventory)) < 1_000
    assert len(json.dumps(draft)) < 2_000
    assert marker in json.dumps(box.broker_contract("get_quote"))


def test_setup_scopes_expose_disjoint_closed_tool_sets():
    broker = {tool["name"] for tool in setup_tool_definitions(SetupScope.BROKER_PROFILE)}
    agent = {tool["name"] for tool in setup_tool_definitions(SetupScope.AGENT_DRAFT)}
    assert broker == {
        "broker_inventory",
        "broker_contract",
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
    chat = ChatSession.create(
        tmp_path / "tick-home",
        provider=Provider.CODEX,
        model="fixture-model",
        codex_cli_version="0.149.0",
        at=NOW,
    )

    with pytest.raises(ChatError) as refused:
        SetupChatSession(chat.home, chat.session_id).delete()

    assert "Start it again" in refused.value.reason
    assert chat.metadata_path.exists()
