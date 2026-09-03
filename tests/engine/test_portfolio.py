"""The account: long-only positions, and equity as a join that can refuse.

Equity is where CLAUDE.md invariant 5 has teeth. Cash and quantities come from
the broker, prices come from market data, and the product of the two is the
only place a number can be invented — so one missing price refuses the whole
figure instead of quietly reporting a smaller account.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from tick.engine import PortfolioState, Position, Quote, Unavailable

from .conftest import LAST_BAR, state


def quote(symbol: str, price: str) -> Quote:
    return Quote(symbol=symbol, price=Decimal(price), asof=LAST_BAR, source="fixture")


def test_equity_is_cash_plus_the_value_of_every_holding():
    account = state("1000.00", XYZ=(10, "90.00"), ABCD=(5, "38.00"))
    equity = account.equity({"XYZ": quote("XYZ", "118.00"), "ABCD": quote("ABCD", "40.00")})
    assert equity == Decimal("2380.00")


def test_one_missing_price_refuses_the_whole_equity():
    account = state("1000.00", XYZ=(10, "90.00"), ABCD=(5, "38.00"))
    equity = account.equity(
        {
            "XYZ": quote("XYZ", "118.00"),
            "ABCD": Unavailable(what="quote for ABCD", reason="the feed is down"),
        }
    )
    assert isinstance(equity, Unavailable)
    assert "quote for ABCD" in equity.reason
    assert "cannot be priced" in equity.reason


def test_a_held_symbol_nobody_quoted_refuses_rather_than_counting_zero():
    account = state("1000.00", XYZ=(10, "90.00"))
    equity = account.equity({})
    assert isinstance(equity, Unavailable)
    assert "no quote was fetched for the held symbol XYZ" in equity.reason


def test_an_empty_account_is_worth_its_cash():
    assert state("250.00").equity({}) == Decimal("250.00")


def test_market_value_of_an_unheld_symbol_is_zero_without_needing_a_price():
    account = state("100.00")
    assert account.market_value("XYZ", Unavailable(what="quote for XYZ", reason="down")) == 0


def test_market_value_of_a_held_symbol_needs_the_price():
    account = state("100.00", XYZ=(3, "10.00"))
    missing = account.market_value("XYZ", Unavailable(what="quote for XYZ", reason="down"))
    assert isinstance(missing, Unavailable)
    assert missing.what == "quote for XYZ"
    assert account.market_value("XYZ", quote("XYZ", "118.00")) == Decimal("354.00")


def test_qty_of_an_unheld_symbol_is_zero():
    assert state("100.00").qty("XYZ") == 0


@pytest.mark.parametrize("qty", [0, -3])
def test_a_position_is_never_zero_or_short(qty):
    with pytest.raises(ValidationError, match="long-only"):
        Position(symbol="XYZ", qty=qty, avg_cost=Decimal("10"))


def test_a_position_needs_a_real_average_cost():
    with pytest.raises(ValidationError, match="avg_cost"):
        Position(symbol="XYZ", qty=1, avg_cost=Decimal("0"))


def test_cash_never_goes_negative():
    with pytest.raises(ValidationError, match="cannot go negative"):
        PortfolioState(cash=Decimal("-0.01"), positions={})


def test_a_position_cannot_be_filed_under_another_symbol():
    with pytest.raises(ValidationError, match="holds a position in"):
        PortfolioState(
            cash=Decimal("1"),
            positions={"XYZ": Position(symbol="ABCD", qty=1, avg_cost=Decimal("10"))},
        )


def test_a_state_is_frozen_once_built():
    account = state("100.00")
    with pytest.raises(ValidationError):
        account.cash = Decimal("200.00")


def test_a_binary_float_is_not_money():
    with pytest.raises(ValidationError, match="binary float"):
        PortfolioState(cash=1000.55, positions={})
