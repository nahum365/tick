"""Market data: the typed missing number, and the fixture port's clock.

`Unavailable` is the shape CLAUDE.md invariant 5 takes in this package. These
tests pin that it is a value with words in it, that it cannot be confused with
a number, and that the fixture port never answers a question about a bar that
has not happened yet.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from tick.engine import (
    Bar,
    FixtureDataError,
    FixtureMarketData,
    MarketDataPort,
    Quote,
    Unavailable,
)

from .conftest import BAR_TIMES, LAST_BAR, MARKET_FIXTURES, market


def test_unavailable_carries_what_and_why():
    missing = Unavailable(what="quote for XYZ", reason="the feed is down")
    assert "quote for XYZ" in str(missing)
    assert "the feed is down" in str(missing)
    assert not isinstance(missing, Decimal)


@pytest.mark.parametrize(
    ("what", "reason"),
    [("", "why"), ("what", ""), ("  ", "why")],
)
def test_unavailable_refuses_to_be_wordless(what, reason):
    with pytest.raises(ValidationError):
        Unavailable(what=what, reason=reason)


def test_unavailable_can_be_named_for_a_symbol():
    scoped = Unavailable(what="sma(20)", reason="needs 20 bars").scoped("XYZ")
    assert scoped.what == "sma(20) for XYZ"
    assert scoped.reason == "needs 20 bars"


def test_a_quote_must_name_where_it_came_from():
    with pytest.raises(ValidationError, match="provenance is not optional"):
        Quote(symbol="XYZ", price=Decimal("10"), asof=LAST_BAR, source="  ")


def test_a_quote_price_is_never_zero_or_negative():
    with pytest.raises(ValidationError, match="must be > 0"):
        Quote(symbol="XYZ", price=Decimal("0"), asof=LAST_BAR, source="fixture")


def _bar(**overrides) -> dict:
    document = {
        "ts": LAST_BAR,
        "open": Decimal("10"),
        "high": Decimal("11"),
        "low": Decimal("9"),
        "close": Decimal("10.5"),
        "volume": 100,
    }
    document.update(overrides)
    return document


def test_a_bar_refuses_an_impossible_range():
    with pytest.raises(ValidationError, match="below its low"):
        Bar(**_bar(high=Decimal("8")))


def test_a_bar_refuses_a_close_outside_its_range():
    with pytest.raises(ValidationError, match="outside its low/high"):
        Bar(**_bar(close=Decimal("12")))


def test_a_bar_needs_an_aware_timestamp():
    with pytest.raises(ValidationError):
        Bar(**_bar(ts=datetime(2026, 8, 14, 20, 0)))


def test_the_fixture_port_satisfies_the_market_data_protocol():
    assert isinstance(market(), MarketDataPort)


def test_a_quote_is_the_last_bar_at_or_before_now():
    data = market(BAR_TIMES[6])
    quote = data.quote("XYZ")
    assert isinstance(quote, Quote)
    assert quote.price == Decimal("100.00")
    assert quote.asof == BAR_TIMES[6]
    assert quote.source == "fixture"


def test_a_bar_after_now_does_not_exist_yet():
    just_before = market(BAR_TIMES[6] - timedelta(seconds=1))
    quote = just_before.quote("XYZ")
    assert isinstance(quote, Quote)
    assert quote.asof == BAR_TIMES[5]


def test_a_symbol_with_no_series_is_unavailable_not_empty():
    result = market().quote("ZZZ")
    assert isinstance(result, Unavailable)
    assert "no fixture series" in result.reason


def test_a_moment_before_the_first_bar_has_no_quote():
    result = market(datetime(2026, 1, 1, tzinfo=UTC)).quote("XYZ")
    assert isinstance(result, Unavailable)
    assert "no bar exists at or before" in result.reason


def test_bars_returns_exactly_n_oldest_first():
    bars = market().bars("XYZ", 3)
    assert not isinstance(bars, Unavailable)
    assert [bar.close for bar in bars] == [
        Decimal("106.00"),
        Decimal("112.00"),
        Decimal("118.00"),
    ]
    assert bars[0].ts < bars[-1].ts


def test_too_little_history_is_unavailable_never_a_shorter_list():
    result = market().bars("WXY", 5)
    assert isinstance(result, Unavailable)
    assert "2 bars exist" in result.reason
    assert "5 were needed" in result.reason


def test_the_clock_can_be_moved_without_reloading():
    data = market(BAR_TIMES[0])
    moved = data.at(BAR_TIMES[9])
    assert data.now == BAR_TIMES[0]
    assert moved.now == BAR_TIMES[9]
    assert isinstance(data.bars("XYZ", 2), Unavailable)
    assert not isinstance(moved.bars("XYZ", 2), Unavailable)


def test_a_naive_now_is_refused():
    with pytest.raises(ValueError, match="timezone-aware"):
        FixtureMarketData.from_directory(MARKET_FIXTURES, now=datetime(2026, 8, 14, 20, 0))


def test_the_loader_ships_no_data_of_its_own(tmp_path):
    with pytest.raises(FixtureDataError, match="no \\*.json market fixtures"):
        FixtureMarketData.from_directory(tmp_path, now=LAST_BAR)


def test_a_series_must_be_strictly_ordered(tmp_path):
    path = tmp_path / "XYZ.json"
    path.write_text(
        json.dumps(
            {
                "symbol": "XYZ",
                "bars": [
                    {
                        "ts": "2026-08-04T20:00:00+00:00",
                        "open": "10",
                        "high": "11",
                        "low": "9",
                        "close": "10",
                        "volume": 1,
                    },
                    {
                        "ts": "2026-08-03T20:00:00+00:00",
                        "open": "10",
                        "high": "11",
                        "low": "9",
                        "close": "10",
                        "volume": 1,
                    },
                ],
            }
        )
    )
    with pytest.raises(FixtureDataError, match="strictly ordered"):
        FixtureMarketData.from_file(path, now=LAST_BAR)


def test_fixture_numbers_are_exact_decimals_not_floats():
    bars = market().bars("ABCD", 6)
    assert not isinstance(bars, Unavailable)
    assert bars[0].close == Decimal("39.50")
    assert all(isinstance(bar.close, Decimal) for bar in bars)
