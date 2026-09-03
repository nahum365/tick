"""What the runtime does with what a model proposes — one intent at a time."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from tick.agents import (
    InstructionsMissing,
    ModelAgentError,
    ModelReplyError,
    build_snapshot,
    model_id_of,
)
from tick.engine import RefusalCode, Unavailable
from tick.spec import Side

from .conftest import (
    INSTRUCTIONS,
    NOW,
    ExplodingModelClient,
    FakeModelClient,
    agent_for,
    build_model_spec,
    market,
    portfolio,
    reply_object,
)


def buy(symbol: str = "XYZ", qty: int = 2, reason: str = "my instructions said so") -> dict:
    return {"symbol": symbol, "side": "buy", "qty": qty, "reason": reason}


def sell(symbol: str = "XYZ", qty: int = 1, reason: str = "my instructions said so") -> dict:
    return {"symbol": symbol, "side": "sell", "qty": qty, "reason": reason}


def evaluate(intents: list, *, held: dict[str, int] | None = None, **kwargs):
    client = FakeModelClient.answering(intents, **kwargs)
    agent = agent_for(client)
    evaluation = agent.evaluate_tick(agent.spec, market(), portfolio(**(held or {})), NOW)
    return agent, evaluation


def only(evaluation):
    assert len(evaluation.decisions) == 1, evaluation.decisions
    return evaluation.decisions[0]


# ----------------------------------------------------------------------
# The instructions are the user's, and there is no default
# ----------------------------------------------------------------------


def test_an_agent_with_no_instructions_refuses_to_exist():
    """Tick ships no default prompt, so an empty file is not a starting point."""
    with pytest.raises(InstructionsMissing, match="Tick ships none"):
        agent_for(FakeModelClient.answering([]), instructions="")


def test_whitespace_is_not_instructions():
    with pytest.raises(InstructionsMissing):
        agent_for(FakeModelClient.answering([]), instructions="   \n\t\n")


def test_the_instructions_hash_is_of_the_users_own_bytes():
    from tick.spec import sha256_hex

    agent = agent_for(FakeModelClient.answering([]))
    assert agent.instructions_sha256 == sha256_hex(INSTRUCTIONS.encode("utf-8"))


# ----------------------------------------------------------------------
# A well-formed intent becomes an OrderIntent, priced from a real quote
# ----------------------------------------------------------------------


def test_a_valid_intent_becomes_an_order_intent_priced_from_the_tick_quote():
    _, evaluation = evaluate([buy(qty=3)])
    decision = only(evaluation)

    assert decision.fired
    assert decision.refusal is None
    intent = decision.intent
    assert intent is not None
    assert intent.symbol == "XYZ"
    assert intent.side is Side.BUY
    assert intent.qty == 3
    assert intent.est_price == evaluation.prices["XYZ"]
    assert intent.est_notional == Decimal(3) * evaluation.prices["XYZ"]
    assert intent.price_source.startswith("fixture")


def test_the_intents_source_names_the_model_so_the_runtime_can_tell():
    _, evaluation = evaluate([buy()])
    source = only(evaluation).intent.source
    assert model_id_of(source) == "claude-opus-5"
    assert model_id_of("rule:golden-cross") is None


def test_the_models_own_reason_is_carried_into_the_intent():
    _, evaluation = evaluate([buy(reason="rebalancing toward my target")])
    assert "rebalancing toward my target" in only(evaluation).intent.reason


def test_an_empty_list_of_intents_is_a_complete_answer():
    _, evaluation = evaluate([])
    assert evaluation.decisions == ()
    assert evaluation.equity == Decimal("10000.00")


# ----------------------------------------------------------------------
# Refusals: every one recorded, none silently dropped
# ----------------------------------------------------------------------


def refusal_for(intents: list, **kwargs):
    _, evaluation = evaluate(intents, **kwargs)
    decision = only(evaluation)
    assert decision.intent is None
    assert decision.refusal is not None
    return decision.refusal


def test_a_symbol_outside_the_users_universe_is_refused_and_named():
    refusal = refusal_for([buy(symbol="WXY")])
    assert refusal.code is RefusalCode.SYMBOL_OUTSIDE_UNIVERSE
    assert "WXY" in refusal.reason
    assert "will not trade outside it" in refusal.reason


def test_a_sell_larger_than_the_position_is_refused_whole_not_truncated():
    refusal = refusal_for([sell(qty=9)], held={"XYZ": 4})
    assert refusal.code is RefusalCode.SELL_EXCEEDS_POSITION
    assert "refused whole" in refusal.reason
    assert "never allowed to go short" in refusal.reason


def test_a_sell_of_a_symbol_the_account_does_not_hold_is_refused():
    refusal = refusal_for([sell(qty=1)])
    assert refusal.code is RefusalCode.NO_POSITION_TO_SELL
    assert "long-only" in refusal.reason


def test_a_sell_of_exactly_what_is_held_is_allowed():
    _, evaluation = evaluate([sell(qty=4)], held={"XYZ": 4})
    assert only(evaluation).intent is not None


@pytest.mark.parametrize(
    "raw",
    [
        {"symbol": "XYZ", "side": "short", "qty": 1, "reason": "r"},
        {"symbol": "XYZ", "side": "buy", "qty": 0, "reason": "r"},
        {"symbol": "XYZ", "side": "buy", "qty": "3", "reason": "r"},
        {"symbol": "XYZ", "side": "buy", "qty": 1, "reason": "  "},
        {"symbol": "XYZ", "side": "buy", "qty": 1},
        {"symbol": "XYZ", "side": "buy", "qty": 1, "reason": "r", "limit_price": "10"},
        "buy some XYZ",
    ],
    ids=[
        "a-side-that-does-not-exist",
        "no-shares",
        "shares-as-a-string",
        "no-reason",
        "a-missing-field",
        "a-field-nobody-declared",
        "not-an-object-at-all",
    ],
)
def test_a_malformed_intent_is_refused_with_words_rather_than_dropped(raw):
    refusal = refusal_for([raw])
    assert refusal.code in {
        RefusalCode.MODEL_OUTPUT_INVALID,
        RefusalCode.SYMBOL_OUTSIDE_UNIVERSE,
    }
    assert refusal.reason.strip()


def test_there_is_no_short_side_to_propose():
    """'short' is not a side, and the refusal says so rather than reinterpreting it."""
    refusal = refusal_for([{"symbol": "XYZ", "side": "short", "qty": 1, "reason": "r"}])
    assert "no short side" in refusal.reason


def test_an_unpriced_symbol_refuses_rather_than_being_sized_at_a_guess():
    class NoQuotes:
        def quote(self, symbol):
            return Unavailable(what=f"quote for {symbol}", reason="the feed said nothing")

        def bars(self, symbol, n):
            return Unavailable(what=f"{n} bars for {symbol}", reason="no history")

    client = FakeModelClient.answering([buy()])
    agent = agent_for(client)
    evaluation = agent.evaluate_tick(agent.spec, NoQuotes(), portfolio(), NOW)
    refusal = only(evaluation).refusal
    assert refusal.code is RefusalCode.QUOTE_UNAVAILABLE
    assert "no price was guessed" in refusal.reason


def test_every_proposal_produces_a_decision_including_the_refused_ones():
    _, evaluation = evaluate([buy(), buy(symbol="WXY"), sell(qty=5)])
    assert len(evaluation.decisions) == 3
    assert sum(1 for d in evaluation.decisions if d.intent is not None) == 1
    assert sum(1 for d in evaluation.decisions if d.refusal is not None) == 2


# ----------------------------------------------------------------------
# Provenance: which model, which words, which prompt
# ----------------------------------------------------------------------


def test_the_model_and_the_prompt_are_recorded_after_a_tick():
    agent, _ = evaluate([buy()])
    provenance = agent.provenance()

    assert provenance["kind"] == "model_agent"
    assert provenance["model"] == "claude-opus-5"
    assert provenance["model_reported"] == "claude-opus-5-20260401"
    assert provenance["instructions_sha256"] == agent.instructions_sha256
    assert len(provenance["prompt_sha256"]) == 64
    assert provenance["intents_proposed"] == 1


def test_the_prompt_hash_changes_when_the_users_instructions_change():
    first, _ = evaluate([])
    client = FakeModelClient.answering([])
    other = agent_for(client, instructions="different words entirely\n")
    other.evaluate_tick(other.spec, market(), portfolio(), NOW)
    assert first.provenance()["prompt_sha256"] != other.provenance()["prompt_sha256"]


def test_an_agent_that_has_not_been_asked_says_so_rather_than_inventing_a_hash():
    agent = agent_for(FakeModelClient.answering([]))
    provenance = agent.provenance()
    assert "prompt_sha256" not in provenance
    assert "has not been asked anything yet" in provenance["text"]


# ----------------------------------------------------------------------
# The reply that cannot be read stops the tick, and is never asked again
# ----------------------------------------------------------------------


def test_a_reply_that_calls_no_tool_stops_the_tick():
    client = FakeModelClient(
        reply_object(tool=None, text="I would rather not.", stop_reason="end_turn")
    )
    agent = agent_for(client)
    with pytest.raises(ModelReplyError, match="never from prose"):
        agent.evaluate_tick(agent.spec, market(), portfolio(), NOW)


def test_a_provider_refusal_stops_the_tick():
    client = FakeModelClient(
        reply_object(tool=None, stop_reason="refusal", stop_details={"category": "policy"})
    )
    agent = agent_for(client)
    with pytest.raises(ModelReplyError, match="declined to answer"):
        agent.evaluate_tick(agent.spec, market(), portfolio(), NOW)


def test_a_reply_that_does_not_say_which_model_answered_is_refused():
    """A decision record that cannot name the model is not a record."""
    client = FakeModelClient(reply_object(intents=[], model=""))
    agent = agent_for(client)
    with pytest.raises(ModelReplyError, match="does not say which model"):
        agent.evaluate_tick(agent.spec, market(), portfolio(), NOW)


def test_a_reply_with_no_intents_key_is_unreadable_rather_than_empty():
    client = FakeModelClient(reply_object(payload={}))
    agent = agent_for(client)
    with pytest.raises(ModelReplyError, match="no 'intents' key"):
        agent.evaluate_tick(agent.spec, market(), portfolio(), NOW)


def test_a_failing_client_is_never_asked_a_second_time():
    client = ExplodingModelClient(ModelAgentError("the provider fell over"))
    agent = agent_for(client)
    with pytest.raises(ModelAgentError):
        agent.evaluate_tick(agent.spec, market(), portfolio(), NOW)
    assert client.calls == 1


# ----------------------------------------------------------------------
# The document an agent runs is the document it was built for
# ----------------------------------------------------------------------


def test_an_agent_refuses_to_tick_a_document_it_was_not_built_for():
    agent = agent_for(FakeModelClient.answering([]))
    with pytest.raises(ValueError, match="is a different agent"):
        agent.evaluate_tick(build_model_spec(model="claude-sonnet-5"), market(), portfolio(), NOW)


def test_a_naive_moment_is_refused():
    agent = agent_for(FakeModelClient.answering([]))
    with pytest.raises(ValueError, match="timezone-aware"):
        agent.evaluate_tick(agent.spec, market(), portfolio(), datetime(2026, 9, 1, 15, 0))


def test_a_cadence_below_the_floor_refuses_before_the_model_is_asked():
    """Robinhood's undefined 'excessive market data usage' is stayed clear of."""
    from tick.engine import CadenceRefused

    client = FakeModelClient.answering([])
    spec = build_model_spec(cadence={"kind": "every_n_minutes", "n": 1})
    agent = agent_for(client, spec=spec)
    with pytest.raises(CadenceRefused):
        agent.evaluate_tick(spec, market(), portfolio(), NOW)
    assert client.requests == []


def test_quotes_are_fetched_once_per_symbol_per_tick():
    _, evaluation = evaluate([buy(), buy(qty=1)], held={"XYZ": 2})
    assert evaluation.quote_calls == 2  # the universe: XYZ and ABCD, once each
    assert evaluation.bars_calls == 0


def test_the_snapshot_helper_needs_every_input(monkeypatch):
    """No argument of the snapshot has a default; nothing is inherited."""
    import inspect

    for name, parameter in inspect.signature(build_snapshot).parameters.items():
        assert parameter.default is inspect.Parameter.empty, name


def test_a_moment_with_an_offset_other_than_utc_is_fine():
    from zoneinfo import ZoneInfo

    eastern = NOW.astimezone(ZoneInfo("America/New_York"))
    client = FakeModelClient.answering([])
    agent = agent_for(client)
    agent.evaluate_tick(agent.spec, market(eastern.astimezone(UTC)), portfolio(), eastern)
    assert client.requests
