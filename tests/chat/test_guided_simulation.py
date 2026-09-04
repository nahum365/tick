from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.broker.test_profile import tool
from tests.chat.test_setup import NOW
from tick.agents import Provider
from tick.broker import Category, contract_for
from tick.broker.profile import prove_proposal
from tick.chat import SetupChatSession, SetupScope
from tick.serve.handlers import APIError, _evaluate_setup, setup_check_reads


def setup(home, *, goal="simulation", document=None):
    session = SetupChatSession.create(
        home,
        scope=SetupScope.BROKER_PROFILE,
        provider=Provider.CODEX,
        model="fixture-model",
        codex_cli_version="fixture",
        at=NOW,
        goal=goal,
    )
    session.save(
        document=document
        or {
            "tools": {
                "read": {"category": "read.quote"},
                "preview": {"category": "order.preflight"},
            }
        },
        valid=True,
        complete=False,
        waiting_for=(),
        probe_values={"symbol": "XYZ"},
        proof={},
        verdict={},
        at=NOW,
    )
    return session


def test_simulation_completion_ignores_preflight_without_authorizing_it(tmp_path):
    session = setup(tmp_path)
    calls = []

    def operation(action, body):
        calls.append((action, body))
        return {"outcome": {"read": {"success": True}}}

    context = SimpleNamespace(home=tmp_path, now=lambda: NOW, broker_profile_operation=operation)
    decision = _evaluate_setup(context, session)
    assert decision.status == "complete" and session.state.complete
    assert calls == [("prove_draft", {"probe": {"symbol": "XYZ"}, "reads_only": True})]
    setup_check_reads(context, session.chat.session_id, {})
    assert calls[-1] == ("prove", {"probe": {"symbol": "XYZ"}, "reads_only": True})
    assert all(action not in {"confirm", "select_account"} for action, _ in calls)
    assert SetupChatSession(tmp_path, session.chat.session_id).state.goal == "simulation"


def test_full_goal_keeps_preflight_requirement_and_empty_reads_never_complete(tmp_path):
    session = setup(tmp_path, goal="full")
    context = SimpleNamespace(
        home=tmp_path,
        now=lambda: NOW,
        broker_profile_operation=lambda *_: {"outcome": {"read": {"success": True}}},
    )
    assert _evaluate_setup(context, session).status != "complete"
    with pytest.raises(APIError):
        setup_check_reads(context, session.chat.session_id, {})
    empty = setup(tmp_path, document={"tools": {"place": {"category": "order.place"}}})
    assert _evaluate_setup(context, empty).status != "complete"


def test_missing_account_stops_for_native_account_selection(tmp_path):
    session = setup(tmp_path)
    context = SimpleNamespace(
        home=tmp_path,
        now=lambda: NOW,
        broker_profile_operation=lambda *_: {
            "outcome": {"read": {"success": False, "unresolved": {"needs": "account_id"}}}
        },
    )
    assert _evaluate_setup(context, session).status == "waiting_for_person"
    assert session.state.waiting_for == ("account_id",)


def test_simulation_probe_actually_skips_preflight_and_every_mutation():
    read = tool(
        "read", output_schema={"type": "object", "properties": {"price": {"type": "string"}}}
    )
    mappings = {
        "read": SimpleNamespace(
            category=Category.READ_QUOTE,
            contract=contract_for(read),
            arguments={},
            result={"price": "price"},
        )
    }
    for name, category in [
        ("preview", Category.ORDER_PREFLIGHT),
        ("place", Category.ORDER_PLACE),
        ("cancel", Category.ORDER_CANCEL),
    ]:
        mappings[name] = SimpleNamespace(category=category)
    calls = []
    session = SimpleNamespace(
        list_tools=lambda: [read],
        call_tool=lambda name, args: calls.append(name) or {"price": "42.00"},
    )
    proposal = SimpleNamespace(tools=mappings, account_id="account-on-box")
    result = prove_proposal(proposal, session, probe_values={}, at=NOW, reads_only=True)
    assert calls == ["read"] and set(result) == {"read"}
