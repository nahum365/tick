"""Live setup failures must reach the model before requests for probe inputs."""

import json
from types import SimpleNamespace

import pytest

from tests.serve.test_broker_ops import AT, accounts_tool, configured, row
from tick.agents import Provider
from tick.broker import Category, contract_for, load_profile, load_proposal, prove_proposal
from tick.broker.profile_model import proposal_instructions
from tick.chat import SetupChatSession, SetupScope
from tick.chat.setup_loop import SetupLoopDecision, run_setup_loop
from tick.serve.handlers import _evaluate_setup


def session(home):
    return SetupChatSession.create(
        home,
        scope=SetupScope.BROKER_PROFILE,
        provider=Provider.CODEX,
        model="fixture-model",
        codex_cli_version="0.149.0",
        at=AT,
        goal="simulation",
    )


def test_chat_receives_the_same_complete_mapping_language_as_one_shot(tmp_path):
    setup = session(tmp_path)
    frames = []

    def adapter(_transcript, frame):
        frames.append(frame)
        return ({"kind": "done"},)

    list(
        run_setup_loop(
            setup,
            now=lambda: AT,
            adapter=adapter,
            evaluate=lambda: SetupLoopDecision("waiting_for_person", "consent", (), None),
        )
    )
    guide = proposal_instructions()
    assert guide["category_grammar"]["read.quote"]["argument_placeholders"]
    assert guide["document_schema"]["properties"]["tools"]
    for rule in guide["rules"]:
        # JSON escapes quotes in the system frame.
        assert json.dumps(rule, ensure_ascii=False) in frames[0]
    assert "do not ask for a symbol yet" in frames[0]


@pytest.mark.parametrize(
    "failure",
    [
        {"items": "the broker answer carries no list at 'accounts'"},
        {"call": "'symbols' is not of type 'null', 'array'"},
        {"call": "'limit' is not of type 'integer'"},
    ],
)
def test_a_bad_mapping_is_repaired_even_when_account_and_symbol_are_missing(tmp_path, failure):
    setup = session(tmp_path)
    needs = {"success": False, "unresolved": {"needs": "account_id, symbol"}}
    outcomes = {"broken": {"success": False, "unresolved": failure}, "quote": needs}
    document = {"tools": {name: {"category": "read.quote"} for name in outcomes}}
    setup.save(
        document=document,
        valid=True,
        complete=False,
        waiting_for=("account_id", "symbol"),
        probe_values={},
        proof={},
        verdict={},
        at=AT,
    )
    context = SimpleNamespace(
        now=lambda: AT,
        broker_profile_operation=lambda *_: {"outcome": outcomes},
    )
    decision = _evaluate_setup(context, setup)
    assert decision.status == "retry"
    assert setup.state.verdict["code"] == "BROKER_PROOF_FAILED"
    assert setup.state.waiting_for == ()
    assert setup.state.proof == outcomes


def test_account_consent_precedes_market_probe_question(tmp_path):
    setup = session(tmp_path)
    outcomes = {
        "accounts": {"success": True, "unresolved": {}},
        "positions": {"success": False, "unresolved": {"needs": "account_id"}},
        "quote": {"success": False, "unresolved": {"needs": "symbol"}},
    }
    setup.save(
        document={"tools": {name: {"category": "read.quote"} for name in outcomes}},
        valid=True,
        complete=False,
        waiting_for=(),
        probe_values={},
        proof={},
        verdict={},
        at=AT,
    )
    context = SimpleNamespace(
        now=lambda: AT,
        broker_profile_operation=lambda *_: {"outcome": outcomes},
    )
    assert _evaluate_setup(context, setup).status == "waiting_for_person"
    assert setup.state.waiting_for == ("account_id",)
    assert setup.state.verdict["code"] == "BROKER_READ_ACCESS_NEEDED"


def test_account_discovery_proof_checks_rows_without_selecting_or_changing_authority(tmp_path):
    ops = configured(tmp_path, [row("fixture-one", True), row("fixture-two", False)])
    before = load_profile(tmp_path)
    outcome = ops.prove_draft({"probe": {}, "reads_only": True})["outcome"]["get_accounts"]
    assert outcome["success"] is True
    assert load_profile(tmp_path) == before
    assert load_proposal(tmp_path).account_id is None
    assert not (tmp_path / "broker" / "account-refs.json").exists()


@pytest.mark.parametrize(
    "bad",
    [
        {"agentic_allowed": "true"},
        {"account_number": 123},
        {"brokerage_account_type": None},
    ],
)
def test_unbound_account_proof_still_refuses_unusable_account_fields(tmp_path, bad):
    ops = configured(tmp_path, [row("fixture-one", True) | bad])
    outcome = ops.prove_draft({"probe": {}, "reads_only": True})["outcome"]["get_accounts"]
    assert outcome["success"] is False


def test_ambiguous_categories_are_not_proof_called(tmp_path):
    ops = configured(tmp_path, [row("fixture-one", True)])
    proposal = load_proposal(tmp_path)
    second = accounts_tool().model_copy(update={"name": "another_accounts_tool"})
    first = proposal.tools["get_accounts"]
    proposal = proposal.model_copy(
        update={
            "tools": {
                **proposal.tools,
                second.name: first.model_copy(update={"contract": contract_for(second)}),
            }
        }
    )
    ops.fake_session.list_tools = lambda: [accounts_tool(), second]
    outcomes = prove_proposal(proposal, ops.fake_session, probe_values={}, at=AT, reads_only=True)
    assert all(not outcome.success for outcome in outcomes.values())
    assert all("More than one tool" in outcome.unresolved["call"] for outcome in outcomes.values())
    assert ops.fake_session.calls == []


def test_bad_output_path_returns_observed_structure_without_broker_values(tmp_path):
    ops = configured(tmp_path, [row("private-account-fixture", True)])
    proposal = load_proposal(tmp_path)
    mapping = proposal.tools["get_accounts"]
    proposal = proposal.model_copy(
        update={
            "tools": {
                "get_accounts": mapping.model_copy(
                    update={"result": {**mapping.result, "items": "wrong.path"}}
                )
            }
        }
    )
    outcomes = prove_proposal(proposal, ops.fake_session, probe_values={}, at=AT, reads_only=True)
    result = outcomes["get_accounts"]
    assert not result.success
    assert '"accounts": [{' in result.detail
    assert '"agentic_allowed": "boolean"' in result.detail
    assert "private-account-fixture" not in result.model_dump_json()


def test_optional_symbol_array_still_needs_a_person_value_and_renders_as_an_array(tmp_path):
    ops = configured(tmp_path, [])
    tool = accounts_tool().model_copy(
        update={
            "name": "get_equity_quotes",
            "input_schema": {
                "type": "object",
                "properties": {"symbols": {"type": ["null", "array"], "items": {"type": "string"}}},
            },
        }
    )
    proposal = load_proposal(tmp_path)
    quote = proposal.tools["get_accounts"].model_copy(
        update={
            "contract": contract_for(tool),
            "category": Category.READ_QUOTE,
            "arguments": {"symbols": ["{symbol}"]},
            "result": {"price": "price"},
        }
    )
    proposal = proposal.model_copy(update={"tools": {tool.name: quote}})
    ops.fake_session.list_tools = lambda: [tool]
    outcome = prove_proposal(proposal, ops.fake_session, probe_values={}, at=AT)[tool.name]
    assert outcome.unresolved == {"needs": "symbol"}
    assert not ops.fake_session.calls
    prove_proposal(proposal, ops.fake_session, probe_values={"symbol": "XYZ"}, at=AT)
    assert ops.fake_session.calls == [(tool.name, {"symbols": ["XYZ"]})]
