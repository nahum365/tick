"""The condition grammar is closed, and each node validates its own parameters.

CLAUDE.md invariant 3 — the spec decides — rests on this being true: if the
grammar had an escape hatch, or accepted a window of zero bars and let the
engine improvise, then what an agent did would not be derivable from what the
document said.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from tick.spec import (
    INDICATOR_KINDS,
    OPERAND_KINDS,
    AllOf,
    AnyOf,
    Cash,
    ChangePct,
    Compare,
    ComparisonOp,
    DayOfWeek,
    Ema,
    Not,
    NumberLiteral,
    PositionQty,
    Price,
    Sma,
    condition_depth,
    indicators_in,
)


def test_grammar_is_closed_over_the_documented_indicators():
    assert INDICATOR_KINDS == {
        "price",
        "sma",
        "ema",
        "change_pct",
        "position_qty",
        "position_pct_of_equity",
        "cash",
        "day_of_week",
    }
    assert OPERAND_KINDS == INDICATOR_KINDS | {"number"}


@pytest.mark.parametrize(
    ("model", "field", "label"),
    [(Sma, "n", "sma"), (Ema, "n", "ema"), (ChangePct, "n_bars", "change_pct")],
)
@pytest.mark.parametrize("bad", [0, -1, -50])
def test_window_indicators_refuse_a_window_below_one(model, field, label, bad):
    with pytest.raises(ValidationError) as caught:
        model(**{field: bad})
    assert f"{label}({bad}): {field} must be >= 1" in str(caught.value)


def test_window_indicators_accept_one():
    assert Sma(n=1).label() == "sma(1)"
    assert Ema(n=200).label() == "ema(200)"
    assert ChangePct(n_bars=3).label() == "change_pct(3)"


def test_number_literal_keeps_the_decimal_it_was_given():
    literal = NumberLiteral(value=Decimal("12.50"))
    assert literal.value == Decimal("12.50")
    assert literal.label() == "number(12.50)"


def test_number_literal_refuses_a_binary_float():
    with pytest.raises(ValidationError) as caught:
        NumberLiteral(value=1.1)
    assert "binary float" in str(caught.value)


@pytest.mark.parametrize("op", ["<", ">", ">=", "<="])
def test_plain_comparisons_accept_any_operand(op):
    condition = Compare(left=Cash(), op=ComparisonOp(op), right=NumberLiteral(value=Decimal(500)))
    assert condition.op.value == op
    assert not condition.op.is_cross


@pytest.mark.parametrize("op", ["crosses_above", "crosses_below"])
def test_a_cross_accepts_two_series(op):
    condition = Compare(left=Price(), op=ComparisonOp(op), right=Sma(n=50))
    assert condition.op.is_cross


@pytest.mark.parametrize("op", ["crosses_above", "crosses_below"])
def test_a_cross_accepts_a_constant_threshold(op):
    """`price crosses_above 200` is the canonical use and stays legal."""
    condition = Compare(left=Price(), op=ComparisonOp(op), right=NumberLiteral(value=Decimal(200)))
    assert condition.right.label() == "number(200)"


@pytest.mark.parametrize("left", [Cash(), PositionQty(), DayOfWeek()])
def test_a_cross_refuses_a_left_side_with_no_history(left):
    with pytest.raises(ValidationError) as caught:
        Compare(left=left, op=ComparisonOp.CROSSES_ABOVE, right=Sma(n=5))
    assert f"crosses_above: {left.label()} has no per-bar history to cross" in str(caught.value)


def test_a_cross_refuses_a_right_side_with_no_history():
    with pytest.raises(ValidationError) as caught:
        Compare(left=Price(), op=ComparisonOp.CROSSES_BELOW, right=Cash())
    assert "crosses_below: cash has no per-bar history to cross" in str(caught.value)


def test_combinators_need_at_least_one_child():
    for combinator in (AllOf, AnyOf):
        with pytest.raises(ValidationError) as caught:
            combinator(of=[])
        assert "at least 1 item" in str(caught.value)


def test_condition_depth_counts_nesting():
    leaf = Compare(left=Price(), op=ComparisonOp.GT, right=NumberLiteral(value=Decimal(1)))
    assert condition_depth(leaf) == 1
    assert condition_depth(Not(of=leaf)) == 2
    assert condition_depth(AllOf(of=[leaf, Not(of=leaf)])) == 3


def test_indicators_in_lists_every_operand_in_order():
    leaf = Compare(left=Price(), op=ComparisonOp.GT, right=Sma(n=5))
    other = Compare(left=Cash(), op=ComparisonOp.LT, right=NumberLiteral(value=Decimal(10)))
    found = indicators_in(AnyOf(of=[leaf, Not(of=other)]))
    assert [item.label() for item in found] == ["price", "sma(5)", "cash", "number(10)"]
