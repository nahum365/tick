"""The per-tick cache: scoped reads, asked once, never retried.

Robinhood may terminate MCP connectivity for undefined "excessive market data
usage" (Customer Agreement §29), so how much Tick asks is a product property
with a test, not a matter of taste. These pin the two rules: only the symbols
this tick is entitled to read, and each question asked at most once.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from tick.engine import (
    MarketDataContractError,
    Quote,
    SymbolOutsideScope,
    TickMarketCache,
    Unavailable,
)

from .conftest import LAST_BAR, StubMarket, market


def _quote(symbol: str, price: str) -> Quote:
    return Quote(symbol=symbol, price=Decimal(price), asof=LAST_BAR, source="stub")


def test_a_symbol_outside_the_tick_scope_is_refused_not_fetched():
    stub = StubMarket({"XYZ": _quote("XYZ", "10")})
    cache = TickMarketCache(stub, frozenset({"XYZ"}))
    with pytest.raises(SymbolOutsideScope, match="not in this tick's scope"):
        cache.quote("ABCD")
    assert stub.quote_asks == []


def test_the_scope_cannot_be_empty():
    with pytest.raises(ValueError, match="at least one permitted symbol"):
        TickMarketCache(StubMarket({}), frozenset())


def test_a_quote_is_fetched_once_per_tick():
    stub = StubMarket({"XYZ": _quote("XYZ", "10")})
    cache = TickMarketCache(stub, frozenset({"XYZ"}))
    first = cache.quote("XYZ")
    second = cache.quote("XYZ")
    assert first == second
    assert stub.quote_asks == ["XYZ"]
    assert cache.quote_calls == 1


def test_an_unavailable_quote_is_not_retried_inside_a_tick():
    stub = StubMarket({"XYZ": Unavailable(what="quote for XYZ", reason="feed down")})
    cache = TickMarketCache(stub, frozenset({"XYZ"}))
    assert isinstance(cache.quote("XYZ"), Unavailable)
    assert isinstance(cache.quote("XYZ"), Unavailable)
    assert cache.quote_calls == 1


def test_a_shorter_history_is_served_from_the_longer_one_already_fetched():
    cache = TickMarketCache(market(), frozenset({"XYZ"}))
    fifty = cache.bars("XYZ", 5)
    three = cache.bars("XYZ", 3)
    assert not isinstance(fifty, Unavailable)
    assert not isinstance(three, Unavailable)
    assert three == fifty[-3:]
    assert cache.bars_calls == 1


def test_a_longer_history_than_was_cached_is_fetched_once_more():
    cache = TickMarketCache(market(), frozenset({"XYZ"}))
    cache.bars("XYZ", 3)
    cache.bars("XYZ", 6)
    cache.bars("XYZ", 4)
    assert cache.bars_calls == 2


def test_an_unavailable_history_is_not_re_asked_for_more():
    stub = StubMarket(
        {},
        {("WXY", 5): Unavailable(what="bars for WXY", reason="only 2 bars")},
    )
    cache = TickMarketCache(stub, frozenset({"WXY"}))
    assert isinstance(cache.bars("WXY", 5), Unavailable)
    assert isinstance(cache.bars("WXY", 9), Unavailable)
    assert stub.bars_asks == [("WXY", 5)]


def test_a_port_that_returns_fewer_bars_than_asked_breaks_its_contract():
    real = market().bars("XYZ", 2)
    stub = StubMarket({}, {("XYZ", 5): real})
    cache = TickMarketCache(stub, frozenset({"XYZ"}))
    with pytest.raises(MarketDataContractError, match="returned 2 bars"):
        cache.bars("XYZ", 5)


def test_cached_quotes_reports_only_what_was_actually_asked():
    cache = TickMarketCache(market(), frozenset({"XYZ", "ABCD"}))
    cache.quote("XYZ")
    assert set(cache.cached_quotes()) == {"XYZ"}
