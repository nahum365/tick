"""Indicator arithmetic: exact, defined, and never quietly shortened.

The values below are written as literals on purpose. An indicator is a *named*
number that ends up in the record and in a notification, so its definition has
to be pinned by something other than the code that computes it — otherwise a
changed window or a changed EMA seed is invisible.
"""

from __future__ import annotations

from decimal import Context, Decimal, localcontext

import pytest

from tick.engine import (
    EMA_WINDOW_MULTIPLE,
    Unavailable,
    change_pct,
    crosses_above,
    crosses_below,
    ema,
    latest_close,
    required_bars,
    sma,
)
from tick.spec import (
    Cash,
    ChangePct,
    DayOfWeek,
    Ema,
    NumberLiteral,
    PositionQty,
    Price,
    Sma,
)

CLOSES = [
    Decimal(value)
    for value in (
        "100.00",
        "98.00",
        "96.00",
        "94.00",
        "92.00",
        "95.00",
        "100.00",
        "106.00",
        "112.00",
        "118.00",
    )
]


def test_latest_close_is_the_last_bar():
    assert latest_close(CLOSES) == Decimal("118.00000000")


def test_sma_is_the_mean_of_the_last_n_closes():
    assert sma(CLOSES, 3) == Decimal("112.00000000")
    assert sma(CLOSES, 5) == Decimal("106.20000000")


def test_sma_of_a_repeating_quotient_is_pinned_to_the_quantum():
    assert sma(CLOSES[:6], 3) == Decimal("93.66666667")


def test_sma_refuses_rather_than_shortening_its_window():
    result = sma(CLOSES[:4], 10)
    assert isinstance(result, Unavailable)
    assert result.what == "sma(10)"
    assert result.reason == "needs 10 bars of history, 4 available"


def test_ema_uses_a_fixed_window_seeded_by_the_sma_of_its_first_half():
    # window = the last 2*3 closes; seed = mean(92, 95, 100); k = 2/(3+1).
    assert EMA_WINDOW_MULTIPLE == 2
    assert ema(CLOSES, 3) == Decimal("112.20833333")


def test_ema_needs_twice_its_window():
    assert isinstance(ema(CLOSES[:5], 3), Unavailable)
    assert not isinstance(ema(CLOSES[:6], 3), Unavailable)


def test_change_pct_is_a_percentage_over_n_bars():
    # 5 bars back from 118.00 is 92.00: (118 - 92) / 92 * 100.
    assert change_pct(CLOSES, 5) == Decimal("28.26086957")
    assert change_pct(CLOSES, 1) == Decimal("5.35714286")


def test_change_pct_needs_one_more_bar_than_its_window():
    result = change_pct(CLOSES[:5], 5)
    assert isinstance(result, Unavailable)
    assert "needs 6 bars" in result.reason


def test_change_pct_refuses_a_zero_reference_rather_than_dividing():
    closes = [Decimal("0"), Decimal("10")]
    result = change_pct(closes, 1)
    assert isinstance(result, Unavailable)
    assert "has no value" in result.reason


@pytest.mark.parametrize("n", [0, -1])
def test_a_window_below_one_is_a_programming_error(n):
    with pytest.raises(ValueError):
        sma(CLOSES, n)


# ----------------------------------------------------------------------
# Crossings — the definition, bar by bar
# ----------------------------------------------------------------------


def test_crosses_above_needs_the_previous_bar_at_or_below():
    assert crosses_above(Decimal("2"), Decimal("1"), Decimal("1"), Decimal("1")) is True
    # already above on the previous bar: not a crossing, however far above.
    assert crosses_above(Decimal("9"), Decimal("2"), Decimal("1"), Decimal("1")) is False
    # equal on this bar is not yet above.
    assert crosses_above(Decimal("1"), Decimal("0"), Decimal("1"), Decimal("1")) is False


def test_crosses_below_is_the_mirror():
    assert crosses_below(Decimal("1"), Decimal("2"), Decimal("2"), Decimal("2")) is True
    assert crosses_below(Decimal("0"), Decimal("1"), Decimal("2"), Decimal("2")) is False
    assert crosses_below(Decimal("2"), Decimal("3"), Decimal("2"), Decimal("2")) is False


def test_a_touch_then_a_rise_crosses_exactly_once():
    # left touches right (equal), then rises: the crossing bar is the rise.
    assert crosses_above(Decimal("5"), Decimal("4"), Decimal("5"), Decimal("5")) is False
    assert crosses_above(Decimal("6"), Decimal("5"), Decimal("5"), Decimal("5")) is True
    assert crosses_above(Decimal("7"), Decimal("6"), Decimal("5"), Decimal("5")) is False


# ----------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------


def test_the_host_decimal_context_cannot_change_an_indicator():
    """A host that narrows the thread's precision must not move our numbers."""
    with localcontext(Context(prec=6)):
        crippled = (sma(CLOSES, 3), ema(CLOSES, 3), change_pct(CLOSES, 5))
    assert crippled == (
        Decimal("112.00000000"),
        Decimal("112.20833333"),
        Decimal("28.26086957"),
    )


# ----------------------------------------------------------------------
# How much history each operand needs
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("node", "now_bars", "with_previous"),
    [
        (Price(), 1, 2),
        (Sma(n=20), 20, 21),
        (Ema(n=10), 20, 21),
        (ChangePct(n_bars=5), 6, 7),
        (NumberLiteral(value=Decimal("1")), 0, 0),
        (Cash(), 0, 0),
        (PositionQty(), 0, 0),
        (DayOfWeek(), 0, 0),
    ],
)
def test_required_bars_counts_the_previous_bar_too(node, now_bars, with_previous):
    assert required_bars(node, previous=False) == now_bars
    assert required_bars(node, previous=True) == with_previous
