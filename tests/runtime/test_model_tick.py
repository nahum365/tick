"""One tick of a caged model agent, through the same runner a rule agent uses.

The point of these tests is the sameness. The cage, the approval, the broker,
the record and the notification grammar are one code path for both kinds of
agent — so what is checked here is that a model's intents meet exactly the
limits a rule's intents meet, that a rejection is recorded the same way, and
that the one thing which differs (who the sentence names, and what the decision
record says about who decided) differs on purpose.

Nothing here reaches a model or a network: the client is a fake that answers
with a prepared reply, read through the adapter's own `read_model_reply`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tick.agents import (
    EMIT_TOOL_NAME,
    ModelAgent,
    ModelAgentSpec,
    ModelReplyError,
    ModelRequest,
    dump_agent_spec,
    parse_agent_spec,
    read_model_reply,
)
from tick.broker import PaperBroker
from tick.records import DataSource, RecordKind, read
from tick.runtime import AgentRun, ApprovalMode, MarketClock, Mode, Runner

from .conftest import IN_SESSION, StepClock, fixture_market

INSTRUCTIONS = "My own words. Buy XYZ when I say so.\n"

CREATED = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)


def model_document(**overrides: Any) -> dict[str, Any]:
    document = {
        "kind": "model_agent",
        "name": "Model agent under test",
        "version": 1,
        "universe": ["ABCD", "XYZ"],
        "cadence": {"kind": "daily_close"},
        "provider": "anthropic",
        "model": "claude-opus-5",
        "cage": {
            "max_position_pct": "100.00",
            "max_positions": 5,
            "max_order_notional": "1000000.00",
            "max_daily_drawdown_pct": "50.00",
            "allowed_session": "regular_hours",
        },
    }
    document.update(overrides)
    return document


def model_spec(**overrides: Any) -> ModelAgentSpec:
    spec = parse_agent_spec(model_document(**overrides))
    assert isinstance(spec, ModelAgentSpec)
    return spec


class FakeClient:
    """Answers one prepared reply, and remembers the request it was handed."""

    def __init__(self, intents: list[Any], *, model: str = "claude-opus-5-20260401") -> None:
        self._reply = SimpleNamespace(
            model=model,
            stop_reason="tool_use",
            stop_details=None,
            content=[
                SimpleNamespace(type="tool_use", name=EMIT_TOOL_NAME, input={"intents": intents})
            ],
        )
        self.requests: list[ModelRequest] = []

    def propose(self, request: ModelRequest):
        self.requests.append(request)
        return read_model_reply(self._reply)


class RefusingClient:
    """A client that will not answer. Counts how often it was asked."""

    def __init__(self) -> None:
        self.calls = 0

    def propose(self, request: ModelRequest):
        self.calls += 1
        raise ModelReplyError("the model declined to answer (category: policy)")


class Notifications(list):
    def __call__(self, sentence: str) -> None:
        self.append(sentence)


def make_agent(home: Path, spec: ModelAgentSpec, *, approval=ApprovalMode.STANDING) -> AgentRun:
    return AgentRun.create(
        home,
        spec,
        max_cancels_per_session=2,
        approval=approval,
        created_at=CREATED,
        instructions=INSTRUCTIONS,
    )


def run_tick(home: Path, intents: list[Any], *, spec: ModelAgentSpec | None = None, **kwargs):
    """One model tick against the paper broker, and everything it produced."""
    spec = spec or model_spec()
    agent = make_agent(home, spec, approval=kwargs.pop("approval", ApprovalMode.STANDING))
    client = kwargs.pop("client", None) or FakeClient(intents)
    market = fixture_market()
    broker = PaperBroker(market, starting_cash=Decimal("10000.00"), max_cancels=2)
    notifications = Notifications()
    runner = Runner(
        stamp=StepClock(),
        market_source=DataSource.FIXTURE,
        broker_source=DataSource.PAPER,
        evaluator=ModelAgent(spec, client=client, instructions=agent.instructions()),
    )
    outcome = runner.tick(
        agent,
        market=market,
        broker=broker,
        clock=MarketClock.for_2026(),
        notify=notifications,
        approve=kwargs.pop("approve", lambda intent: True),
        now=IN_SESSION,
    )
    return SimpleNamespace(
        agent=agent,
        broker=broker,
        client=client,
        outcome=outcome,
        notifications=notifications,
        records=list(read(agent.ledger_path)),
    )


def buy(symbol: str = "XYZ", qty: int = 2) -> dict[str, Any]:
    return {"symbol": symbol, "side": "buy", "qty": qty, "reason": "my instructions said so"}


# ----------------------------------------------------------------------
# The agent's own files
# ----------------------------------------------------------------------


def test_a_model_agent_is_created_with_the_users_instructions_on_disk(tick_home: Path):
    agent = make_agent(tick_home, model_spec())
    assert agent.instructions_path.read_text(encoding="utf-8") == INSTRUCTIONS
    assert agent.instructions() == INSTRUCTIONS
    assert agent.spec_path.read_text(encoding="utf-8") == dump_agent_spec(model_spec())


def test_a_model_agent_cannot_be_created_without_instructions(tick_home: Path):
    from tick.runtime import RuntimeStateError

    with pytest.raises(RuntimeStateError, match="Tick ships none"):
        AgentRun.create(
            tick_home,
            model_spec(),
            max_cancels_per_session=1,
            approval=ApprovalMode.STANDING,
            created_at=CREATED,
            instructions=None,
        )


def test_a_rule_agent_cannot_be_created_with_instructions(tick_home: Path):
    from tick.runtime import RuntimeStateError

    from .conftest import build_spec

    with pytest.raises(RuntimeStateError, match="reads no instructions file"):
        AgentRun.create(
            tick_home,
            build_spec(),
            max_cancels_per_session=1,
            approval=ApprovalMode.STANDING,
            created_at=CREATED,
            instructions="words nothing would read",
        )


def test_re_adding_with_different_instructions_is_refused_rather_than_silent(tick_home: Path):
    make_agent(tick_home, model_spec())
    from tick.runtime import RuntimeStateError

    with pytest.raises(RuntimeStateError, match="different instructions"):
        AgentRun.create(
            tick_home,
            model_spec(),
            max_cancels_per_session=2,
            approval=ApprovalMode.STANDING,
            created_at=CREATED,
            instructions="entirely different words\n",
        )


def test_an_emptied_instructions_file_refuses_rather_than_running_on_nothing(tick_home: Path):
    from tick.runtime import RuntimeStateError

    agent = make_agent(tick_home, model_spec())
    agent.instructions_path.write_text("   \n", encoding="utf-8")
    with pytest.raises(RuntimeStateError, match="will not supply one"):
        agent.instructions()


def test_a_summary_says_it_is_model_driven_and_names_the_model(tick_home: Path):
    from tick.runtime import state_summary

    agent = make_agent(tick_home, model_spec())
    summary = state_summary(agent)
    assert summary["kind"] == "model_agent"
    assert summary["model"] == "claude-opus-5"
    assert summary["rules"] == []


# ----------------------------------------------------------------------
# A model's intent goes through the cage and reaches the broker
# ----------------------------------------------------------------------


def test_a_valid_intent_is_placed_and_recorded_like_any_other(tick_home: Path):
    result = run_tick(tick_home, [buy(qty=2)])

    assert result.outcome.fills, result.outcome
    assert result.broker.state().qty("XYZ") == 2
    assert [record.kind for record in result.records] == [
        RecordKind.DECISION,
        RecordKind.ORDER,
        RecordKind.FILL,
    ]
    assert result.agent.verify_ledger().ok


def test_the_notification_names_the_model_and_never_a_rule(tick_home: Path):
    result = run_tick(tick_home, [buy(qty=1)])
    sentence = result.notifications[0]
    assert sentence.startswith("Your model agent (claude-opus-5) bought 1 XYZ at $")
    assert sentence.endswith("— simulated.")
    assert "rule" not in sentence


def test_the_decision_record_names_the_model_the_prompt_and_the_instructions(tick_home: Path):
    result = run_tick(tick_home, [buy()])
    decision = next(r for r in result.records if r.kind is RecordKind.DECISION)
    provenance = decision.payload["agent"]

    assert provenance["kind"] == "model_agent"
    assert provenance["model"] == "claude-opus-5"
    assert provenance["model_reported"] == "claude-opus-5-20260401"
    assert len(provenance["prompt_sha256"]) == 64
    assert len(provenance["instructions_sha256"]) == 64


def test_a_rule_agents_decision_record_says_a_rule_agent_decided(agent, market, clock):
    """The same key, filled by whoever decided — so a reader never has to guess."""
    from tick.engine import RuleEvaluator

    broker = PaperBroker(market, starting_cash=Decimal("10000.00"), max_cancels=1)
    runner = Runner(
        stamp=clock,
        market_source=DataSource.FIXTURE,
        broker_source=DataSource.PAPER,
        evaluator=RuleEvaluator(),
    )
    runner.tick(
        agent,
        market=market,
        broker=broker,
        clock=MarketClock.for_2026(),
        notify=Notifications(),
        approve=lambda intent: True,
        now=IN_SESSION,
    )
    decision = next(r for r in read(agent.ledger_path) if r.kind is RecordKind.DECISION)
    assert decision.payload["agent"] == {"kind": "rule_agent"}


# ----------------------------------------------------------------------
# The cage holds a model exactly as it holds a rule
# ----------------------------------------------------------------------


def test_an_intent_over_the_cages_order_ceiling_is_rejected_and_recorded(tick_home: Path):
    tight = model_spec(
        cage=model_document()["cage"] | {"max_order_notional": "100.00"},
    )
    result = run_tick(tick_home, [buy(qty=5)], spec=tight)

    assert not result.outcome.fills
    assert result.broker.state().qty("XYZ") == 0
    refusals = [r for r in result.records if r.kind is RecordKind.REFUSAL]
    assert len(refusals) == 1
    assert refusals[0].payload["stage"] == "cage"
    assert "per-order ceiling" in refusals[0].payload["rejection"]["reason"]


def test_a_cage_rejection_of_a_model_intent_is_notified_in_the_models_words(tick_home: Path):
    tight = model_spec(cage=model_document()["cage"] | {"max_order_notional": "100.00"})
    result = run_tick(tick_home, [buy(qty=5)], spec=tight)
    assert result.notifications[0].startswith(
        "Your model agent (claude-opus-5) proposed an order that was rejected:"
    )


def test_a_symbol_outside_the_universe_never_reaches_the_broker(tick_home: Path):
    result = run_tick(tick_home, [buy(symbol="WXY")])

    assert not result.outcome.fills
    refusal = next(r for r in result.records if r.kind is RecordKind.REFUSAL)
    assert refusal.payload["stage"] == "engine"
    assert refusal.payload["refusal"]["code"] == "symbol_outside_universe"


def test_a_declined_approval_places_nothing_for_a_model_agent_either(tick_home: Path):
    result = run_tick(
        tick_home,
        [buy()],
        approval=ApprovalMode.EACH,
        approve=lambda intent: False,
    )
    assert not result.outcome.fills
    assert RecordKind.FILL not in [record.kind for record in result.records]
    assert "you declined" in result.outcome.not_placed[0]


# ----------------------------------------------------------------------
# Fail safe: the model, the account read, and the kill switch
# ----------------------------------------------------------------------


def test_a_model_that_will_not_answer_stops_the_tick_and_is_not_asked_again(tick_home: Path):
    client = RefusingClient()
    result = run_tick(tick_home, [], client=client)

    assert client.calls == 1
    assert result.outcome.halted
    assert not result.outcome.fills
    note = next(r for r in result.records if r.kind is RecordKind.NOTE)
    assert note.payload["event"] == "model_failed"
    assert "was not asked again" in note.payload["reason"]
    assert result.notifications[-1].startswith("Tick stopped agent ")


def test_after_a_model_failure_the_agent_can_still_be_stopped(tick_home: Path):
    """A failure that locks the protective act is still a failure."""
    result = run_tick(tick_home, [], client=RefusingClient())
    result.agent.request_stop(reason="enough", at=IN_SESSION)
    assert result.agent.stop_requested()
    assert result.agent.verify_ledger().ok


def test_the_kill_switch_is_checked_before_the_model_is_asked(tick_home: Path):
    spec = model_spec()
    agent = make_agent(tick_home, spec)
    agent.request_stop(reason="not today", at=IN_SESSION)
    client = FakeClient([buy()])
    market = fixture_market()
    runner = Runner(
        stamp=StepClock(),
        market_source=DataSource.FIXTURE,
        broker_source=DataSource.PAPER,
        evaluator=ModelAgent(spec, client=client, instructions=INSTRUCTIONS),
    )
    outcome = runner.tick(
        agent,
        market=market,
        broker=PaperBroker(market, starting_cash=Decimal("1000.00"), max_cancels=1),
        clock=MarketClock.for_2026(),
        notify=Notifications(),
        approve=lambda intent: True,
        now=IN_SESSION,
    )
    assert outcome.stopped
    assert client.requests == []


def test_an_account_that_cannot_be_read_stops_the_run_without_evaluating(agent, market, clock):
    """An account Tick could not read is not an empty account."""
    from tick.engine import RuleEvaluator

    class UnreadableBroker:
        def state(self):
            raise RuntimeError("the brokerage dropped the connection")

        def place(self, intent):  # pragma: no cover - never reached
            raise AssertionError("nothing is placed when the account cannot be read")

        def cancel(self, order_id):  # pragma: no cover - never reached
            raise AssertionError

    notifications = Notifications()
    runner = Runner(
        stamp=clock,
        market_source=DataSource.FIXTURE,
        broker_source=DataSource.PAPER,
        evaluator=RuleEvaluator(),
    )
    outcome = runner.tick(
        agent,
        market=market,
        broker=UnreadableBroker(),
        clock=MarketClock.for_2026(),
        notify=notifications,
        approve=lambda intent: True,
        now=IN_SESSION,
    )

    assert outcome.halted
    assert not outcome.evaluated
    note = next(r for r in read(agent.ledger_path) if r.kind is RecordKind.NOTE)
    assert note.payload["event"] == "broker_failed"
    assert note.payload["action"] == "read_state"
    assert "was not retried" in note.payload["reason"]
    assert notifications[-1].startswith("Tick stopped agent ")
    assert agent.verify_ledger().ok


def test_the_agent_a_model_ticks_is_the_document_it_was_built_for(tick_home: Path):
    """A model agent holding one cage while the record cites another is refused."""
    spec = model_spec()
    other = model_spec(model="claude-sonnet-5")
    agent = make_agent(tick_home, spec)
    market = fixture_market()
    runner = Runner(
        stamp=StepClock(),
        market_source=DataSource.FIXTURE,
        broker_source=DataSource.PAPER,
        evaluator=ModelAgent(other, client=FakeClient([]), instructions=INSTRUCTIONS),
    )
    with pytest.raises(ValueError, match="is a different agent"):
        runner.tick(
            agent,
            market=market,
            broker=PaperBroker(market, starting_cash=Decimal("100.00"), max_cancels=1),
            clock=MarketClock.for_2026(),
            notify=Notifications(),
            approve=lambda intent: True,
            now=IN_SESSION,
        )


def test_a_model_agent_stays_in_paper_until_something_says_otherwise(tick_home: Path):
    """Paper is the default on every path, model agents included (invariant 2)."""
    agent = make_agent(tick_home, model_spec())
    assert agent.state.mode is Mode.PAPER
