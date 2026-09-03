"""The rule evaluator: when a rule fires, what it sizes, and when it refuses.

The load-bearing tests here are the ones about *not* acting. A missing number
produces a `Refusal` and never a zero-priced order; a sell bigger than the
position is refused whole and never quietly reduced or turned into a short.
Both are product properties before they are code properties.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from tick.engine import (
    CadenceRefused,
    RefusalCode,
    RuleEvaluator,
    Unavailable,
)
from tick.spec import Side

from .conftest import (
    ALWAYS,
    BAR_TIMES,
    CROSS_BAR,
    LAST_BAR,
    build_spec,
    compare,
    market,
    rule,
    state,
)

SMA_CROSS = compare({"kind": "sma", "n": 3}, "crosses_above", {"kind": "sma", "n": 5})


def evaluate(spec, *, now=LAST_BAR, account=None, data=None):
    return RuleEvaluator().evaluate(
        spec,
        data if data is not None else market(now),
        account if account is not None else state(),
        now,
    )


def only(decisions):
    assert len(decisions) == 1, decisions
    return decisions[0]


# ----------------------------------------------------------------------
# Firing
# ----------------------------------------------------------------------


def test_a_cross_fires_on_exactly_one_bar_of_the_series():
    spec = build_spec(
        universe=["XYZ"],
        rules=[rule("cross", SMA_CROSS, side="buy", size={"kind": "shares", "shares": 1})],
    )
    fired_on = [moment for moment in BAR_TIMES if only(evaluate(spec, now=moment)).fired]
    assert fired_on == [CROSS_BAR]


def test_a_decision_is_recorded_even_when_the_rule_does_not_fire():
    spec = build_spec(
        universe=["XYZ"],
        rules=[rule("cross", SMA_CROSS, side="buy", size={"kind": "shares", "shares": 1})],
    )
    decision = only(evaluate(spec, now=LAST_BAR))
    assert decision.fired is False
    assert decision.intent is None and decision.refusal is None
    labels = {value.label: value.value for value in decision.values}
    assert labels["sma(3)"] == Decimal("112.00000000")
    assert labels["sma(5)"] == Decimal("106.20000000")
    assert labels["sma(3)@prev"] == Decimal("106.00000000")
    assert labels["sma(5)@prev"] == Decimal("101.00000000")


def test_every_rule_is_evaluated_against_every_symbol_of_the_universe():
    spec = build_spec(
        universe=["XYZ", "ABCD"],
        rules=[
            rule("one", ALWAYS, side="buy", size={"kind": "shares", "shares": 1}),
            rule("two", ALWAYS, side="buy", size={"kind": "shares", "shares": 1}),
        ],
    )
    decisions = evaluate(spec)
    assert {(d.rule_id, d.symbol) for d in decisions} == {
        ("one", "XYZ"),
        ("one", "ABCD"),
        ("two", "XYZ"),
        ("two", "ABCD"),
    }


def test_a_naive_now_is_refused():
    spec = build_spec(
        universe=["XYZ"],
        rules=[rule("r", ALWAYS, side="buy", size={"kind": "shares", "shares": 1})],
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        RuleEvaluator().evaluate(spec, market(), state(), datetime(2026, 8, 14, 20, 0))


def test_day_of_week_is_the_eastern_weekday():
    # 01:00 UTC on Tuesday is still Monday evening in New York.
    spec = build_spec(
        universe=["XYZ"],
        rules=[
            rule(
                "monday",
                compare({"kind": "day_of_week"}, "<=", {"kind": "number", "value": "1"}),
                side="buy",
                size={"kind": "shares", "shares": 1},
            )
        ],
    )
    tuesday_utc = datetime(2026, 8, 11, 1, 0, tzinfo=UTC)
    decision = only(evaluate(spec, now=tuesday_utc, data=market(LAST_BAR)))
    assert decision.fired is True


# ----------------------------------------------------------------------
# Sizing
# ----------------------------------------------------------------------


def test_notional_sizing_floors_to_whole_shares():
    spec = build_spec(
        universe=["ABCD"],
        rules=[rule("buy", ALWAYS, side="buy", size={"kind": "notional", "notional": "750.25"})],
    )
    intent = only(evaluate(spec)).intent
    assert intent is not None
    # $750.25 at $40.00 is 18.75 shares; 18 are bought, never 19.
    assert intent.qty == 18
    assert intent.est_price == Decimal("40.00")
    assert intent.est_notional == Decimal("720.00")
    assert intent.side is Side.BUY
    assert intent.price_source == "fixture"
    assert intent.price_asof == LAST_BAR


def test_percent_of_equity_sizing_uses_equity_and_floors():
    spec = build_spec(
        universe=["ABCD"],
        rules=[
            rule(
                "buy",
                ALWAYS,
                side="buy",
                size={"kind": "pct_of_equity", "pct_of_equity": "10.00"},
            )
        ],
    )
    # equity = 1000 cash + 10 XYZ at 118.00 = 2180.00; 10% = 218.00; /40 = 5.45.
    account = state("1000.00", XYZ=(10, "90.00"))
    intent = only(evaluate(spec, account=account)).intent
    assert intent is not None
    assert intent.qty == 5
    assert intent.est_notional == Decimal("200.00")


def test_buying_all_spends_the_cash_on_hand():
    spec = build_spec(
        universe=["ABCD"],
        rules=[rule("buy", ALWAYS, side="buy", size={"kind": "all"})],
    )
    intent = only(evaluate(spec, account=state("101.00"))).intent
    assert intent is not None
    assert intent.qty == 2  # 101.00 / 40.00 floors to 2


def test_selling_all_means_the_whole_position():
    spec = build_spec(
        universe=["ABCD"],
        rules=[rule("exit", ALWAYS, side="sell", size={"kind": "all"})],
    )
    intent = only(evaluate(spec, account=state("0.00", ABCD=(7, "38.00")))).intent
    assert intent is not None
    assert intent.side is Side.SELL
    assert intent.qty == 7


def test_a_size_that_rounds_to_nothing_refuses_rather_than_ordering_zero():
    spec = build_spec(
        universe=["ABCD"],
        rules=[rule("buy", ALWAYS, side="buy", size={"kind": "all"})],
    )
    decision = only(evaluate(spec, account=state("39.99")))
    assert decision.fired is True
    assert decision.intent is None
    assert decision.refusal is not None
    assert decision.refusal.code is RefusalCode.SIZE_ROUNDS_TO_ZERO
    assert "whole shares only" in decision.refusal.reason


# ----------------------------------------------------------------------
# Long-only
# ----------------------------------------------------------------------


def test_a_sell_larger_than_the_position_is_refused_whole_not_truncated():
    spec = build_spec(
        universe=["ABCD"],
        rules=[rule("exit", ALWAYS, side="sell", size={"kind": "shares", "shares": 10})],
    )
    decision = only(evaluate(spec, account=state("0.00", ABCD=(3, "38.00"))))
    assert decision.fired is True
    assert decision.intent is None
    assert decision.refusal is not None
    assert decision.refusal.code is RefusalCode.SELL_EXCEEDS_POSITION
    assert "long-only" in decision.refusal.reason
    assert "never reduced" in decision.refusal.reason


def test_a_sell_with_nothing_held_never_opens_a_short():
    spec = build_spec(
        universe=["ABCD"],
        rules=[rule("exit", ALWAYS, side="sell", size={"kind": "all"})],
    )
    decision = only(evaluate(spec, account=state("100.00")))
    assert decision.intent is None
    assert decision.refusal is not None
    assert decision.refusal.code is RefusalCode.NO_POSITION_TO_SELL
    assert "never opens a short" in decision.refusal.reason


def test_a_notional_sell_bigger_than_the_position_is_refused_too():
    spec = build_spec(
        universe=["ABCD"],
        rules=[rule("exit", ALWAYS, side="sell", size={"kind": "notional", "notional": "400.00"})],
    )
    decision = only(evaluate(spec, account=state("0.00", ABCD=(3, "38.00"))))
    assert decision.refusal is not None
    assert decision.refusal.code is RefusalCode.SELL_EXCEEDS_POSITION


# ----------------------------------------------------------------------
# Unavailability never becomes a number
# ----------------------------------------------------------------------


def test_a_missing_quote_refuses_instead_of_pricing_the_order_at_zero():
    spec = build_spec(
        universe=["XYZ"],
        rules=[
            rule(
                "buy",
                compare({"kind": "cash"}, ">", {"kind": "number", "value": "0"}),
                side="buy",
                size={"kind": "notional", "notional": "100.00"},
            )
        ],
    )
    # A moment before the first bar: the condition is decidable, the price is not.
    decision = only(evaluate(spec, now=datetime(2026, 1, 1, tzinfo=UTC)))
    assert decision.fired is True
    assert decision.intent is None
    assert decision.refusal is not None
    assert decision.refusal.code is RefusalCode.QUOTE_UNAVAILABLE
    assert "price we do not have" in decision.refusal.reason


def test_too_little_history_refuses_the_rule():
    spec = build_spec(
        universe=["WXY"],
        rules=[rule("cross", SMA_CROSS, side="buy", size={"kind": "shares", "shares": 1})],
    )
    decision = only(evaluate(spec))
    assert decision.fired is False
    assert decision.refusal is not None
    assert decision.refusal.code is RefusalCode.BARS_UNAVAILABLE
    assert "2 bars exist" in decision.refusal.reason
    assert any(value.unavailable is not None for value in decision.values)


def test_a_symbol_with_no_data_at_all_refuses_rather_than_being_skipped():
    spec = build_spec(
        universe=["ZZZ"],
        rules=[rule("buy", ALWAYS, side="buy", size={"kind": "shares", "shares": 1})],
    )
    decision = only(evaluate(spec))
    assert decision.refusal is not None
    assert "no fixture series" in decision.refusal.reason
    # A refusal is read by a person: it names the symbol once, not twice.
    assert "for ZZZ for ZZZ" not in decision.refusal.reason
    assert decision.refusal.reason.count("bars for ZZZ") == 1


def test_percent_of_equity_refuses_when_the_account_cannot_be_priced():
    spec = build_spec(
        universe=["ABCD"],
        rules=[
            rule(
                "buy",
                ALWAYS,
                side="buy",
                size={"kind": "pct_of_equity", "pct_of_equity": "10.00"},
            )
        ],
    )
    # ZZZ is held but has no series at all, so the account cannot be priced —
    # while ABCD, the symbol being traded, quotes perfectly well.
    account = state("1000.00", ZZZ=(5, "10.00"))
    decision = only(evaluate(spec, account=account))
    assert decision.intent is None
    assert decision.refusal is not None
    assert decision.refusal.code is RefusalCode.EQUITY_UNAVAILABLE
    assert "percentage of an unknown number is not a size" in decision.refusal.reason


# ----------------------------------------------------------------------
# Unavailability only matters where it could change the answer
# ----------------------------------------------------------------------


def _all_of(*conditions):
    return {"kind": "all_of", "of": list(conditions)}


def _any_of(*conditions):
    return {"kind": "any_of", "of": list(conditions)}


NEVER = compare({"kind": "cash"}, "<", {"kind": "number", "value": "0"})
UNAVAILABLE_TERM = compare({"kind": "sma", "n": 400}, ">", {"kind": "number", "value": "0"})


def test_all_of_with_a_definitely_false_child_is_false_not_a_refusal():
    spec = build_spec(
        universe=["XYZ"],
        rules=[
            rule(
                "r",
                _all_of(NEVER, UNAVAILABLE_TERM),
                side="buy",
                size={"kind": "shares", "shares": 1},
            )
        ],
    )
    decision = only(evaluate(spec))
    assert decision.fired is False
    assert decision.refusal is None


def test_any_of_with_a_definitely_true_child_fires_despite_a_missing_sibling():
    spec = build_spec(
        universe=["ABCD"],
        rules=[
            rule(
                "r",
                _any_of(ALWAYS, UNAVAILABLE_TERM),
                side="buy",
                size={"kind": "shares", "shares": 1},
            )
        ],
    )
    decision = only(evaluate(spec))
    assert decision.fired is True
    assert decision.intent is not None


def test_all_of_with_only_an_undecidable_child_refuses():
    spec = build_spec(
        universe=["XYZ"],
        rules=[
            rule(
                "r",
                _all_of(ALWAYS, UNAVAILABLE_TERM),
                side="buy",
                size={"kind": "shares", "shares": 1},
            )
        ],
    )
    decision = only(evaluate(spec))
    assert decision.fired is False
    assert decision.refusal is not None
    assert decision.refusal.code is RefusalCode.BARS_UNAVAILABLE


def test_not_of_an_undecidable_condition_stays_undecidable():
    spec = build_spec(
        universe=["XYZ"],
        rules=[
            rule(
                "r",
                {"kind": "not", "of": UNAVAILABLE_TERM},
                side="buy",
                size={"kind": "shares", "shares": 1},
            )
        ],
    )
    assert only(evaluate(spec)).refusal is not None


# ----------------------------------------------------------------------
# Cadence and data cost
# ----------------------------------------------------------------------


def test_a_cadence_below_the_floor_refuses_before_the_tick_runs():
    spec = build_spec(
        universe=["XYZ"],
        rules=[rule("r", ALWAYS, side="buy", size={"kind": "shares", "shares": 1})],
        cadence={"kind": "every_n_minutes", "n": 2},
    )
    with pytest.raises(CadenceRefused, match="floor of 5 minutes"):
        evaluate(spec)


def test_a_cadence_at_the_floor_runs():
    spec = build_spec(
        universe=["XYZ"],
        rules=[rule("r", ALWAYS, side="buy", size={"kind": "shares", "shares": 1})],
        cadence={"kind": "every_n_minutes", "n": 5},
    )
    assert evaluate(spec)


def test_a_symbol_is_quoted_once_however_many_rules_need_it():
    spec = build_spec(
        universe=["ABCD"],
        rules=[
            rule("one", ALWAYS, side="buy", size={"kind": "notional", "notional": "100.00"}),
            rule("two", ALWAYS, side="buy", size={"kind": "notional", "notional": "200.00"}),
            rule("three", ALWAYS, side="buy", size={"kind": "notional", "notional": "300.00"}),
        ],
    )
    evaluation = RuleEvaluator().evaluate_tick(spec, market(), state(), LAST_BAR)
    assert evaluation.quote_calls == 1
    assert len(evaluation.decisions) == 3


def test_only_the_symbols_a_tick_needed_are_priced():
    spec = build_spec(
        universe=["ABCD", "XYZ"],
        rules=[
            rule(
                "r",
                compare({"kind": "price"}, "<", {"kind": "number", "value": "50"}),
                side="buy",
                size={"kind": "shares", "shares": 1},
            )
        ],
    )
    evaluation = RuleEvaluator().evaluate_tick(spec, market(), state(), LAST_BAR)
    # XYZ is evaluated from bars and never fires, so it is never quoted.
    assert set(evaluation.prices) == {"ABCD"}


def test_equity_is_priced_every_tick_for_the_cage():
    spec = build_spec(
        universe=["ABCD"],
        rules=[rule("r", ALWAYS, side="buy", size={"kind": "shares", "shares": 1})],
    )
    account = state("1000.00", XYZ=(10, "90.00"))
    evaluation = RuleEvaluator().evaluate_tick(spec, market(), account, LAST_BAR)
    assert evaluation.equity == Decimal("2180.00")


def test_a_held_symbol_outside_the_universe_is_still_in_scope():
    spec = build_spec(
        universe=["ABCD"],
        rules=[rule("r", ALWAYS, side="buy", size={"kind": "shares", "shares": 1})],
    )
    account = state("1000.00", XYZ=(1, "90.00"))
    evaluation = RuleEvaluator().evaluate_tick(spec, market(), account, LAST_BAR)
    assert not isinstance(evaluation.equity, Unavailable)
    assert set(evaluation.prices) == {"ABCD", "XYZ"}
