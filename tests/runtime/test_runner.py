"""One tick end to end: the order of the checks, and what each of them refuses."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from tick.broker import Fill, PaperBroker, RejectCode
from tick.engine import OrderIntent, RuleEvaluator, Unavailable
from tick.records import DataSource, RecordKind, read
from tick.runtime import (
    AgentRun,
    ApprovalMode,
    LedgerQuarantined,
    MarketClock,
    Mode,
    ModeNotWired,
    Runner,
    Scheduler,
    TickOutcome,
)

from .conftest import (
    AFTER_CLOSE,
    HOLIDAY,
    IN_SESSION,
    WEEKEND,
    StepClock,
    always_buy,
    always_sell,
    build_spec,
    never_fires,
)


def _seed_intent(symbol: str, *, qty: int, price: str) -> OrderIntent:
    """An intent used only to put a starting position into a paper broker."""
    return OrderIntent(
        source="test:seed",
        symbol=symbol,
        side="buy",
        qty=qty,
        est_price=Decimal(price),
        est_notional=Decimal(price) * qty,
        price_asof=datetime(2026, 8, 14, 20, 0, tzinfo=UTC),
        price_source="fixture",
        reason="seeding a position the test needs",
    )


def never_approves(intent: OrderIntent) -> bool:
    return False


def always_approves(intent: OrderIntent) -> bool:
    return True


def must_not_be_asked(intent: OrderIntent) -> bool:
    raise AssertionError("standing approval must never call the approve callback")


class Notifications(list):
    """A `notify` callback that keeps what it was sent."""

    def __call__(self, sentence: str) -> None:
        self.append(sentence)


class BrokerThatFails:
    """A broker whose `place` raises, and which counts how often it was asked."""

    def __init__(self, inner: PaperBroker, error: Exception) -> None:
        self._inner = inner
        self._error = error
        self.place_calls = 0

    def state(self):
        return self._inner.state()

    def place(self, intent):
        self.place_calls += 1
        raise self._error

    def cancel(self, order_id):  # pragma: no cover - not reached in these tests
        return self._inner.cancel(order_id)


class RecordingBroker:
    """Records which methods the runtime called, to pin what it reads."""

    def __init__(self, inner: PaperBroker) -> None:
        self._inner = inner
        self.calls: list[str] = []

    def state(self):
        self.calls.append("state")
        return self._inner.state()

    def place(self, intent):
        self.calls.append("place")
        return self._inner.place(intent)

    def cancel(self, order_id):
        self.calls.append("cancel")
        return self._inner.cancel(order_id)


@pytest.fixture
def clock_2026() -> MarketClock:
    return MarketClock.for_2026()


@pytest.fixture
def runner(clock: StepClock) -> Runner:
    return Runner(
        stamp=clock,
        market_source=DataSource.FIXTURE,
        broker_source=DataSource.PAPER,
        evaluator=RuleEvaluator(),
    )


@pytest.fixture
def broker(market) -> PaperBroker:
    return PaperBroker(market, starting_cash=Decimal("10000.00"), max_cancels=3)


def tick(runner, agent, market, broker, clock_2026, notify, *, approve=must_not_be_asked, now=None):
    return runner.tick(
        agent,
        market=market,
        broker=broker,
        clock=clock_2026,
        notify=notify,
        approve=approve,
        now=now if now is not None else IN_SESSION,
    )


def kinds(agent: AgentRun) -> list[str]:
    return [record.kind.value for record in read(agent.ledger_path)]


# ----------------------------------------------------------------------
# A paper tick, end to end
# ----------------------------------------------------------------------


def test_a_paper_tick_fills_notifies_and_records(runner, agent, market, broker, clock_2026):
    sent = Notifications()
    outcome = tick(runner, agent, market, broker, clock_2026, sent)

    assert isinstance(outcome, TickOutcome)
    assert outcome.evaluated and outcome.session_open
    assert not outcome.stopped and not outcome.halted
    assert [fill.symbol for fill in outcome.fills] == ["XYZ"]
    assert outcome.fills[0].qty == 2
    assert sent == [f"Your rule 'always' fired: {outcome.fills[0].describe()} — simulated."]
    assert kinds(agent) == ["decision", "order", "fill"]
    assert agent.verify_ledger().ok


def test_the_fill_moves_the_paper_account(runner, agent, market, broker, clock_2026):
    tick(runner, agent, market, broker, clock_2026, Notifications())
    state = broker.state()
    assert state.qty("XYZ") == 2
    assert state.cash < Decimal("10000.00")


def test_the_decision_record_carries_the_whole_evaluation(
    runner, agent, market, broker, clock_2026
):
    """One record per tick, not one per rule per symbol (slice 03's open question)."""
    spec = build_spec(universe=["XYZ", "ABCD"], rules=[always_buy(), never_fires()])
    many = AgentRun.create(
        agent.home,
        spec,
        max_cancels_per_session=1,
        approval=ApprovalMode.STANDING,
        created_at=datetime(2026, 9, 1, tzinfo=UTC),
        instructions=None,
    )
    tick(runner, many, market, broker, clock_2026, Notifications())

    decisions = [r for r in read(many.ledger_path) if r.kind is RecordKind.DECISION]
    assert len(decisions) == 1
    payload = decisions[0].payload
    assert payload["broker"] == "paper"
    assert payload["profile_hash"] is None
    assert payload["inventory_hash"] is None
    assert payload["profile_sanction"] is None
    assert len(payload["decisions"]) == 4  # 2 symbols × 2 rules
    assert payload["source"] == DataSource.FIXTURE.value
    assert set(payload["prices"]) == {"XYZ", "ABCD"}
    assert payload["equity"] == "10000.00"


def test_a_rule_that_does_not_fire_notifies_nobody(runner, agent, market, broker, clock_2026):
    quiet = AgentRun.create(
        agent.home,
        build_spec(rules=[never_fires()]),
        max_cancels_per_session=1,
        approval=ApprovalMode.STANDING,
        created_at=datetime(2026, 9, 1, tzinfo=UTC),
        instructions=None,
    )
    sent = Notifications()
    outcome = tick(runner, quiet, market, broker, clock_2026, sent)

    assert sent == []
    assert outcome.fills == ()
    assert kinds(quiet) == ["decision"]


def test_the_opening_equity_is_taken_from_the_first_tick_of_the_session(
    runner, agent, market, broker, clock_2026
):
    tick(runner, agent, market, broker, clock_2026, Notifications())
    state = agent.state
    assert state.session_date.isoformat() == "2026-09-01"
    assert state.day_start_equity == Decimal("10000.00")
    assert state.last_tick == IN_SESSION


# ----------------------------------------------------------------------
# The session
# ----------------------------------------------------------------------


@pytest.mark.parametrize("moment", [AFTER_CLOSE, WEEKEND, HOLIDAY])
def test_outside_the_session_nothing_is_read_and_nothing_is_recorded(
    runner, agent, market, broker, clock_2026, moment
):
    sent = Notifications()
    outcome = tick(runner, agent, market, broker, clock_2026, sent, now=moment)

    assert not outcome.session_open
    assert not outcome.evaluated
    assert outcome.records == 0
    assert sent == []
    assert not agent.ledger_path.exists()


# ----------------------------------------------------------------------
# The kill switch
# ----------------------------------------------------------------------


def test_the_kill_switch_halts_before_any_order(runner, agent, market, broker, clock_2026):
    agent.request_stop(reason="the user asked", at=IN_SESSION)
    sent = Notifications()
    outcome = tick(runner, agent, market, broker, clock_2026, sent)

    assert outcome.stopped
    assert outcome.fills == ()
    assert broker.state().qty("XYZ") == 0
    assert kinds(agent) == ["stop"]
    assert sent == [f"Tick stopped agent {agent.agent_id!r}: the user asked — simulated."]


def test_a_stop_set_after_a_fill_halts_the_next_tick(runner, agent, market, broker, clock_2026):
    """The switch is checked on every tick, not only the first."""
    tick(runner, agent, market, broker, clock_2026, Notifications())
    agent.request_stop(reason="enough", at=IN_SESSION)
    outcome = tick(runner, agent, market, broker, clock_2026, Notifications())

    assert outcome.stopped
    assert broker.state().qty("XYZ") == 2  # unchanged by the stopped tick
    assert kinds(agent)[-1] == "stop"


def test_the_kill_switch_beats_a_mode_mismatch(runner, agent, market, broker, clock_2026):
    """Stop is checked first, so a live agent that is stopped stops rather than raising."""
    agent.save_state(agent.state.with_mode(Mode.LIVE))
    agent.request_stop(reason="the user asked", at=IN_SESSION)
    outcome = tick(runner, agent, market, broker, clock_2026, Notifications())
    assert outcome.stopped


# ----------------------------------------------------------------------
# The mode and the broker have to agree
# ----------------------------------------------------------------------


def test_a_live_agent_on_a_paper_broker_is_refused_and_the_refusal_is_recorded(
    runner, agent, market, broker, clock_2026
):
    """A simulated fill labelled live would be worse than no record at all."""
    agent.save_state(agent.state.with_mode(Mode.LIVE))
    with pytest.raises(ModeNotWired, match="is in live mode"):
        tick(runner, agent, market, broker, clock_2026, Notifications())

    assert kinds(agent) == ["note"]
    note = list(read(agent.ledger_path))[0]
    assert note.payload["event"] == "mode_broker_mismatch"
    assert note.payload["mode"] == "live"
    assert note.payload["expected_broker_source"] == "robinhood"
    assert broker.state().qty("XYZ") == 0


def test_a_paper_agent_on_a_live_broker_is_refused_in_the_same_way(
    clock, agent, market, broker, clock_2026
):
    """The mismatch is refused in both directions, not only the dangerous-looking one."""
    live_shaped = Runner(
        stamp=clock,
        market_source=DataSource.ROBINHOOD,
        broker_source=DataSource.ROBINHOOD,
        evaluator=RuleEvaluator(),
    )
    with pytest.raises(ModeNotWired, match="is in paper mode"):
        tick(live_shaped, agent, market, broker, clock_2026, Notifications())
    assert list(read(agent.ledger_path))[0].payload["expected_broker_source"] == "paper"


# ----------------------------------------------------------------------
# The ledger that must verify first
# ----------------------------------------------------------------------


def test_a_tampered_ledger_stops_the_tick_with_no_order(runner, agent, market, broker, clock_2026):
    tick(runner, agent, market, broker, clock_2026, Notifications())
    before = agent.ledger_path.read_bytes()
    agent.ledger_path.write_text(agent.ledger_path.read_text().replace('"XYZ"', '"WXY"', 1))
    tampered = agent.ledger_path.read_bytes()
    held = broker.state().qty("XYZ")

    with pytest.raises(LedgerQuarantined) as raised:
        tick(runner, agent, market, broker, clock_2026, Notifications())

    assert f"tick ledger new {agent.agent_id}" in str(raised.value)
    assert broker.state().qty("XYZ") == held  # nothing placed
    assert agent.ledger_path.read_bytes() == tampered != before  # nothing recorded


def test_the_agent_records_again_once_a_successor_is_started(
    runner, agent, market, broker, clock_2026, clock
):
    tick(runner, agent, market, broker, clock_2026, Notifications())
    agent.ledger_path.write_text(agent.ledger_path.read_text().replace('"XYZ"', '"WXY"', 1))
    agent.start_successor_ledger(clock=clock)

    outcome = tick(runner, agent, market, broker, clock_2026, Notifications())
    assert outcome.fills
    assert kinds(agent) == ["note", "decision", "order", "fill"]


# ----------------------------------------------------------------------
# Approval
# ----------------------------------------------------------------------


def test_standing_approval_never_asks(runner, agent, market, broker, clock_2026):
    outcome = tick(
        runner, agent, market, broker, clock_2026, Notifications(), approve=must_not_be_asked
    )
    assert outcome.fills


def test_declining_places_nothing_and_records_the_decline(
    runner, agent, market, broker, clock_2026
):
    agent.save_state(agent.state.with_approval(ApprovalMode.EACH))
    sent = Notifications()
    outcome = tick(runner, agent, market, broker, clock_2026, sent, approve=never_approves)

    assert outcome.fills == ()
    assert broker.state().qty("XYZ") == 0
    assert kinds(agent) == ["decision", "order", "refusal"]
    order = [r for r in read(agent.ledger_path) if r.kind is RecordKind.ORDER][0]
    assert order.payload["approved"] is False
    assert order.payload["approval"] == "each"
    assert len(sent) == 1 and "you declined" in sent[0]
    assert sent[0].endswith(" — simulated.")


def test_approving_places_the_order(runner, agent, market, broker, clock_2026):
    agent.save_state(agent.state.with_approval(ApprovalMode.EACH))
    outcome = tick(
        runner, agent, market, broker, clock_2026, Notifications(), approve=always_approves
    )
    assert [fill.qty for fill in outcome.fills] == [2]
    assert kinds(agent) == ["decision", "order", "fill"]


# ----------------------------------------------------------------------
# Refusals, rejections and the cage
# ----------------------------------------------------------------------


def test_a_fired_rule_that_cannot_be_sized_is_recorded_and_notified(
    runner, agent, market, broker, clock_2026
):
    """Long-only: a sell with no position refuses whole rather than shorting."""
    seller = AgentRun.create(
        agent.home,
        build_spec(rules=[always_sell()]),
        max_cancels_per_session=1,
        approval=ApprovalMode.STANDING,
        created_at=datetime(2026, 9, 1, tzinfo=UTC),
        instructions=None,
    )
    sent = Notifications()
    outcome = tick(runner, seller, market, broker, clock_2026, sent)

    assert outcome.fills == ()
    assert kinds(seller) == ["decision", "refusal"]
    assert len(sent) == 1
    assert "long-only" in sent[0].lower()
    assert sent[0].startswith("Your rule 'exit' fired but the order was rejected:")


def test_a_cage_rejection_is_recorded_and_notified(runner, agent, market, broker, clock_2026):
    caged = AgentRun.create(
        agent.home,
        build_spec(
            cage={
                "max_position_pct": "100.00",
                "max_positions": 5,
                "max_order_notional": "1.00",
                "max_daily_drawdown_pct": "50.00",
                "allowed_session": "regular_hours",
            }
        ),
        max_cancels_per_session=1,
        approval=ApprovalMode.STANDING,
        created_at=datetime(2026, 9, 1, tzinfo=UTC),
        instructions=None,
    )
    sent = Notifications()
    outcome = tick(runner, caged, market, broker, clock_2026, sent)

    assert outcome.fills == ()
    assert kinds(caged) == ["decision", "refusal"]
    refusal = [r for r in read(caged.ledger_path) if r.kind is RecordKind.REFUSAL][0]
    assert refusal.payload["stage"] == "cage"
    assert "per-order ceiling" in sent[0]


def test_a_broker_rejection_is_recorded_and_notified(runner, agent, market, clock_2026):
    """The cage lets it through and the account cannot pay for it.

    The account is worth $10,000 but almost all of that is in ABCD, so buying
    2 XYZ at $118 is 2.36% of equity (the cage is content) and $236 against
    $200 of cash (the broker refuses it whole rather than part-filling).
    """
    poor = PaperBroker(market, starting_cash=Decimal("10000.00"), max_cancels=1)
    poor.place(_seed_intent("ABCD", qty=245, price="40.00"))
    assert poor.state().cash == Decimal("200.00")

    sent = Notifications()
    outcome = tick(runner, agent, market, poor, clock_2026, sent)

    assert outcome.fills == ()
    assert kinds(agent) == ["decision", "order", "rejected"]
    rejected = [r for r in read(agent.ledger_path) if r.kind is RecordKind.REJECTED][0]
    assert rejected.payload["rejected"]["code"] == RejectCode.INSUFFICIENT_CASH.value
    assert "the order was rejected" in sent[0]


# ----------------------------------------------------------------------
# The broker that fails
# ----------------------------------------------------------------------


def test_a_broker_exception_stops_the_run_without_retrying(
    runner, agent, market, broker, clock_2026
):
    failing = BrokerThatFails(broker, ConnectionError("the MCP connection dropped"))
    sent = Notifications()
    outcome = tick(runner, agent, market, failing, clock_2026, sent)

    assert outcome.halted
    assert outcome.fills == ()
    assert failing.place_calls == 1  # asked once, never again
    assert "ConnectionError" in outcome.halt_reason
    assert kinds(agent) == ["decision", "order", "note"]
    note = [r for r in read(agent.ledger_path) if r.kind is RecordKind.NOTE][0]
    assert note.payload["event"] == "broker_failed"
    assert "the MCP connection dropped" in note.payload["error"]
    assert sent[0].startswith(f"Tick stopped agent {agent.agent_id!r}")


def test_after_a_broker_failure_the_user_can_still_stop_the_agent(
    runner, agent, market, broker, clock_2026
):
    """The fail-safe question: after the failure, what can the user still DO?"""
    failing = BrokerThatFails(broker, ConnectionError("dropped"))
    tick(runner, agent, market, failing, clock_2026, Notifications())

    assert agent.verify_ledger().ok  # the record survived the failure
    agent.request_stop(reason="after the failure", at=IN_SESSION)
    outcome = tick(runner, agent, market, failing, clock_2026, Notifications())
    assert outcome.stopped
    assert failing.place_calls == 1  # the stopped tick asked the broker for nothing


def test_a_later_intent_is_not_placed_after_a_failure(runner, agent, market, broker, clock_2026):
    two_rules = AgentRun.create(
        agent.home,
        build_spec(universe=["XYZ", "ABCD"], rules=[always_buy(shares=1)]),
        max_cancels_per_session=1,
        approval=ApprovalMode.STANDING,
        created_at=datetime(2026, 9, 1, tzinfo=UTC),
        instructions=None,
    )
    failing = BrokerThatFails(broker, RuntimeError("boom"))
    outcome = tick(runner, two_rules, market, failing, clock_2026, Notifications())

    assert outcome.halted
    assert failing.place_calls == 1  # two intents, one attempt


# ----------------------------------------------------------------------
# Read scoping
# ----------------------------------------------------------------------


def test_the_runtime_asks_the_broker_for_nothing_but_state_and_place(
    runner, agent, market, broker, clock_2026
):
    """Read scoping: one account's state, and the orders it places. Nothing wider."""
    recording = RecordingBroker(broker)
    tick(runner, agent, market, recording, clock_2026, Notifications())
    assert recording.calls == ["state", "place"]


def test_quotes_are_fetched_once_per_symbol_per_tick(runner, agent, market, broker, clock_2026):
    tick(runner, agent, market, broker, clock_2026, Notifications())
    decision = list(read(agent.ledger_path))[0]
    assert decision.payload["quote_calls"] == 1
    assert set(decision.payload["prices"]) == {"XYZ"}


# ----------------------------------------------------------------------
# The cancel guard
# ----------------------------------------------------------------------


def test_cancels_are_refused_beyond_the_configured_limit(runner, agent, broker, market, clock_2026):
    tick(runner, agent, market, broker, clock_2026, Notifications())
    order_id = "paper-000001"

    first = runner.cancel(agent, broker=broker, order_id=order_id)
    second = runner.cancel(agent, broker=broker, order_id=order_id)
    assert agent.state.cancels_this_session == 2

    recording = RecordingBroker(broker)
    third = runner.cancel(agent, broker=recording, order_id=order_id)

    assert third.code is RejectCode.CANCEL_LIMIT_REACHED
    assert recording.calls == []  # the broker is not even asked
    assert "the configured limit of 2" in third.reason
    assert first is not None and second is not None
    assert kinds(agent)[-1] == "refusal"


def test_the_cancel_counter_resets_when_a_new_session_opens(
    runner, agent, broker, market, clock_2026
):
    tick(runner, agent, market, broker, clock_2026, Notifications())
    runner.cancel(agent, broker=broker, order_id="paper-000001")
    assert agent.state.cancels_this_session == 1

    next_day = IN_SESSION + timedelta(days=1)
    tick(runner, agent, market, broker, clock_2026, Notifications(), now=next_day)
    assert agent.state.cancels_this_session == 0


def test_a_cancel_on_a_quarantined_ledger_refuses(runner, agent, broker, market, clock_2026):
    tick(runner, agent, market, broker, clock_2026, Notifications())
    agent.ledger_path.write_text(agent.ledger_path.read_text().replace('"XYZ"', '"WXY"', 1))
    with pytest.raises(LedgerQuarantined):
        runner.cancel(agent, broker=broker, order_id="paper-000001")


# ----------------------------------------------------------------------
# The loop
# ----------------------------------------------------------------------


class FakeTime:
    """A wall clock the test moves, and a `sleep` that moves it."""

    def __init__(self, start: datetime) -> None:
        self.now = start
        self.slept: list[float] = []

    def __call__(self) -> datetime:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now = self.now + timedelta(seconds=seconds)


def test_the_loop_ticks_on_the_schedule_and_sleeps_between(
    runner, agent, market, broker, clock_2026
):
    time = FakeTime(IN_SESSION)
    outcomes = runner.run(
        agent,
        market=market,
        broker=broker,
        clock=clock_2026,
        scheduler=Scheduler(clock_2026),
        notify=Notifications(),
        approve=must_not_be_asked,
        now=time,
        sleep=time.sleep,
        poll_seconds=600.0,
        max_ticks=2,
    )
    assert len(outcomes) == 2
    assert all(outcome.evaluated for outcome in outcomes)
    assert time.slept  # it waited for the schedule rather than spinning


def test_the_loop_never_sleeps_longer_than_the_poll_interval(
    runner, agent, market, broker, clock_2026
):
    """Liveness for the kill switch: a stop is noticed within one poll."""
    time = FakeTime(IN_SESSION)
    runner.run(
        agent,
        market=market,
        broker=broker,
        clock=clock_2026,
        scheduler=Scheduler(clock_2026),
        notify=Notifications(),
        approve=must_not_be_asked,
        now=time,
        sleep=time.sleep,
        poll_seconds=30.0,
        max_ticks=1,
    )
    assert max(time.slept) <= 30.0


def test_the_loop_stops_when_the_kill_switch_is_set(runner, agent, market, broker, clock_2026):
    agent.request_stop(reason="the user asked", at=IN_SESSION)
    time = FakeTime(IN_SESSION)
    outcomes = runner.run(
        agent,
        market=market,
        broker=broker,
        clock=clock_2026,
        scheduler=Scheduler(clock_2026),
        notify=Notifications(),
        approve=must_not_be_asked,
        now=time,
        sleep=time.sleep,
        poll_seconds=30.0,
        max_ticks=10,
    )
    assert len(outcomes) == 1 and outcomes[0].stopped
    assert time.slept == []  # it did not even wait for the schedule


def test_the_loop_stops_on_a_halt(runner, agent, market, broker, clock_2026):
    failing = BrokerThatFails(broker, RuntimeError("boom"))
    time = FakeTime(IN_SESSION)
    outcomes = runner.run(
        agent,
        market=market,
        broker=failing,
        clock=clock_2026,
        scheduler=Scheduler(clock_2026),
        notify=Notifications(),
        approve=must_not_be_asked,
        now=time,
        sleep=time.sleep,
        poll_seconds=30.0,
        max_ticks=10,
    )
    assert len(outcomes) == 1 and outcomes[0].halted
    assert failing.place_calls == 1


def test_the_poll_interval_must_be_positive(runner, agent, market, broker, clock_2026):
    time = FakeTime(IN_SESSION)
    with pytest.raises(ValueError, match="poll_seconds"):
        runner.run(
            agent,
            market=market,
            broker=broker,
            clock=clock_2026,
            scheduler=Scheduler(clock_2026),
            notify=Notifications(),
            approve=must_not_be_asked,
            now=time,
            sleep=time.sleep,
            poll_seconds=0.0,
            max_ticks=1,
        )


# ----------------------------------------------------------------------
# No silent defaults
# ----------------------------------------------------------------------


def test_the_tick_has_no_default_callbacks():
    """A default no-op `notify` or a default-approving `approve` ships dead wiring."""
    import inspect

    parameters = inspect.signature(Runner.tick).parameters
    for name in ("market", "broker", "clock", "notify", "approve", "now"):
        assert parameters[name].default is inspect.Parameter.empty
    constructor = inspect.signature(Runner.__init__).parameters
    for name in ("stamp", "market_source", "broker_source"):
        assert constructor[name].default is inspect.Parameter.empty


def test_an_unavailable_equity_stops_the_session_from_opening(
    runner, agent, market, clock_2026, tick_home: Path
):
    """A drawdown limit against an invented opening balance is invariant 5's failure."""

    class UnpriceableBroker:
        def __init__(self, inner):
            self._inner = inner

        def state(self):
            return self._inner.state()

        def place(self, intent):  # pragma: no cover - never reached
            raise AssertionError("nothing is placed when the session cannot open")

        def cancel(self, order_id):  # pragma: no cover - not used
            raise AssertionError

    class BlindMarket:
        def quote(self, symbol):
            return Unavailable(what=f"quote for {symbol}", reason="the feed is down")

        def bars(self, symbol, n):
            return Unavailable(what=f"bars for {symbol}", reason="the feed is down")

    inner = PaperBroker(market, starting_cash=Decimal("10000.00"), max_cancels=1)
    inner.place(_seed_intent("XYZ", qty=1, price="118.00"))
    sent = Notifications()
    outcome = tick(runner, agent, BlindMarket(), UnpriceableBroker(inner), clock_2026, sent)

    assert outcome.halted
    assert outcome.fills == ()
    assert kinds(agent) == ["decision", "note"]
    assert agent.state.day_start_equity is None
    assert sent[0].startswith(f"Tick stopped agent {agent.agent_id!r}")


def test_a_fill_is_a_broker_sourced_record_and_a_decision_a_market_sourced_one(
    runner, agent, market, broker, clock_2026
):
    """Provenance travels with the row: prices from the market, fills from the broker."""
    tick(runner, agent, market, broker, clock_2026, Notifications())
    by_kind = {record.kind: record for record in read(agent.ledger_path)}
    assert by_kind[RecordKind.DECISION].source is DataSource.FIXTURE
    assert by_kind[RecordKind.FILL].source is DataSource.PAPER
    assert isinstance(by_kind[RecordKind.FILL].payload["fill"], dict)


def test_the_fill_in_the_record_is_the_traded_price_not_the_estimate(
    runner, agent, market, broker, clock_2026
):
    outcome = tick(runner, agent, market, broker, clock_2026, Notifications())
    fill: Fill = outcome.fills[0]
    recorded = [r for r in read(agent.ledger_path) if r.kind is RecordKind.FILL][0]
    assert recorded.payload["fill"]["price"] == str(fill.price)
