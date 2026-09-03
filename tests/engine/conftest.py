"""Shared fixtures for the engine tests.

Nothing here touches the network, a broker, or `~/.tick`. The market fixtures
in `tests/fixtures/market` are the only files read, and they are test material:
they are not examples, presets or starter strategies, and no part of the
product surfaces them. Symbols are placeholders (`XYZ`, `ABCD`, `WXY`) for the
same reason — Tick authors no strategies and names no real securities.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from tick.engine import (
    BarsResult,
    FixtureMarketData,
    PortfolioState,
    Position,
    QuoteResult,
)
from tick.spec import StrategySpec, parse_spec

MARKET_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "market"

#: The close of the last bar in every fixture series.
LAST_BAR = datetime(2026, 8, 14, 20, 0, tzinfo=UTC)

#: The bar on which XYZ's sma(3) crosses above its sma(5).
CROSS_BAR = datetime(2026, 8, 11, 20, 0, tzinfo=UTC)

#: Every bar timestamp in the fixture series, oldest first.
BAR_TIMES = [
    datetime(2026, 8, day, 20, 0, tzinfo=UTC) for day in (3, 4, 5, 6, 7, 10, 11, 12, 13, 14)
]


def market(now: datetime = LAST_BAR) -> FixtureMarketData:
    """The fixture market seen from `now`."""
    return FixtureMarketData.from_directory(MARKET_FIXTURES, now=now)


def state(cash: str = "10000.00", **positions: tuple[int, str]) -> PortfolioState:
    """An account: `state("5000.00", XYZ=(10, "100.00"))`."""
    return PortfolioState(
        cash=Decimal(cash),
        positions={
            symbol: Position(symbol=symbol, qty=qty, avg_cost=Decimal(avg))
            for symbol, (qty, avg) in positions.items()
        },
    )


def spec_document(
    *,
    universe: list[str],
    rules: list[dict[str, Any]],
    cadence: dict[str, Any] | None = None,
    cage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A spec as a raw document, so a test can bend one field."""
    return {
        "name": "Engine test spec",
        "version": 1,
        "universe": universe,
        "cadence": cadence if cadence is not None else {"kind": "daily_close"},
        "rules": rules,
        "cage": cage
        if cage is not None
        else {
            "max_position_pct": "100.00",
            "max_positions": 5,
            "max_order_notional": "1000000.00",
            "max_daily_drawdown_pct": "50.00",
            "allowed_session": "regular_hours",
        },
    }


def build_spec(**kwargs: Any) -> StrategySpec:
    return parse_spec(spec_document(**kwargs))


def compare(left: dict[str, Any], op: str, right: dict[str, Any]) -> dict[str, Any]:
    return {"kind": "compare", "left": left, "op": op, "right": right}


def rule(
    rule_id: str,
    when: dict[str, Any],
    *,
    side: str,
    size: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": rule_id,
        "when": when,
        "then": {"side": side, "size": size, "order_type": "market"},
    }


#: A condition that is always true, for tests about what happens after firing.
ALWAYS = compare({"kind": "price"}, ">", {"kind": "number", "value": "0"})


class StubMarket:
    """A market-data port that answers from a script and counts what it was asked.

    Used to pin the cache's behaviour: what gets asked, how often, and what
    happens when a port breaks its own contract.
    """

    def __init__(
        self,
        quotes: dict[str, QuoteResult],
        bars: dict[tuple[str, int], BarsResult] | None = None,
    ) -> None:
        self._quotes = quotes
        self._bars = bars if bars is not None else {}
        self.quote_asks: list[str] = []
        self.bars_asks: list[tuple[str, int]] = []

    def quote(self, symbol: str) -> QuoteResult:
        self.quote_asks.append(symbol)
        return self._quotes[symbol]

    def bars(self, symbol: str, n: int) -> BarsResult:
        self.bars_asks.append((symbol, n))
        return self._bars[(symbol, n)]


@pytest.fixture
def fixture_market() -> FixtureMarketData:
    return market()
