"""Shared fixtures for the runtime tests.

Everything here writes under `tmp_path`. `TICK_HOME` is pointed at a temporary
directory for every test in this package, autouse, because the failure it
prevents — a test appending to the developer's own agent record — is silent,
permanent and append-only.

The specs and market series are the placeholder ones (`XYZ`, `ABCD`, `WXY`).
Tick authors no strategies and names no real securities, in tests as anywhere
else; the fixtures under `tests/fixtures/market` are test material and no
command exposes them.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from tick.engine import FixtureMarketData
from tick.records import TICK_HOME_ENV
from tick.runtime import EASTERN, AgentRun, ApprovalMode
from tick.spec import StrategySpec, parse_spec

MARKET_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "market"

#: The close of the last bar in every market fixture series.
LAST_BAR = datetime(2026, 8, 14, 20, 0, tzinfo=UTC)

#: A Tuesday inside the 2026 calendar, mid-session, in ET.
IN_SESSION = datetime(2026, 9, 1, 11, 0, tzinfo=EASTERN)

#: The same day, after the close.
AFTER_CLOSE = datetime(2026, 9, 1, 18, 0, tzinfo=EASTERN)

#: A Saturday.
WEEKEND = datetime(2026, 9, 5, 11, 0, tzinfo=EASTERN)

#: Labor Day 2026.
HOLIDAY = datetime(2026, 9, 7, 11, 0, tzinfo=EASTERN)


class StepClock:
    """A clock that advances a minute per reading, so record timestamps are chosen."""

    def __init__(self, start: datetime | None = None) -> None:
        self.next = start or datetime(2026, 9, 1, 15, 0, tzinfo=UTC)
        self.step = timedelta(minutes=1)

    def __call__(self) -> datetime:
        moment = self.next
        self.next = moment + self.step
        return moment


@pytest.fixture(autouse=True)
def tick_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    """`TICK_HOME` under `tmp_path`, for every runtime test."""
    home = tmp_path / "tick-home"
    monkeypatch.setenv(TICK_HOME_ENV, str(home))
    yield home


@pytest.fixture
def clock() -> StepClock:
    return StepClock()


def fixture_market(now: datetime = IN_SESSION) -> FixtureMarketData:
    """The placeholder market series, seen from `now`.

    Seen from a moment inside the session under test, so every bar in the
    fixtures is visible and a quote is the last one's close.
    """
    return FixtureMarketData.from_directory(MARKET_FIXTURES, now=now)


@pytest.fixture
def market() -> FixtureMarketData:
    return fixture_market()


def spec_document(
    *,
    universe: list[str] | None = None,
    rules: list[dict[str, Any]] | None = None,
    cadence: dict[str, Any] | None = None,
    cage: dict[str, Any] | None = None,
    name: str = "Runtime test spec",
) -> dict[str, Any]:
    """A spec as a raw document, so a test can bend one field."""
    return {
        "name": name,
        "version": 1,
        "universe": universe if universe is not None else ["XYZ"],
        "cadence": cadence if cadence is not None else {"kind": "daily_close"},
        "rules": rules if rules is not None else [always_buy()],
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


def always_buy(rule_id: str = "always", shares: int = 2) -> dict[str, Any]:
    """A rule whose condition is always true, for tests about what follows firing."""
    return {
        "id": rule_id,
        "when": {
            "kind": "compare",
            "left": {"kind": "price"},
            "op": ">",
            "right": {"kind": "number", "value": "0"},
        },
        "then": {
            "side": "buy",
            "size": {"kind": "shares", "shares": shares},
            "order_type": "market",
        },
    }


def always_sell(rule_id: str = "exit", shares: int = 2) -> dict[str, Any]:
    return {
        "id": rule_id,
        "when": {
            "kind": "compare",
            "left": {"kind": "price"},
            "op": ">",
            "right": {"kind": "number", "value": "0"},
        },
        "then": {
            "side": "sell",
            "size": {"kind": "shares", "shares": shares},
            "order_type": "market",
        },
    }


def never_fires(rule_id: str = "never") -> dict[str, Any]:
    return {
        "id": rule_id,
        "when": {
            "kind": "compare",
            "left": {"kind": "price"},
            "op": "<",
            "right": {"kind": "number", "value": "0"},
        },
        "then": {
            "side": "buy",
            "size": {"kind": "shares", "shares": 1},
            "order_type": "market",
        },
    }


def build_spec(**kwargs: Any) -> StrategySpec:
    return parse_spec(spec_document(**kwargs))


@pytest.fixture
def spec() -> StrategySpec:
    return build_spec()


@pytest.fixture
def agent(tick_home: Path, spec: StrategySpec) -> AgentRun:
    return AgentRun.create(
        tick_home,
        spec,
        max_cancels_per_session=2,
        approval=ApprovalMode.STANDING,
        created_at=datetime(2026, 9, 1, 12, 0, tzinfo=UTC),
        instructions=None,  # a rule agent reads none, and passing one is refused
    )


def paper_cash(amount: str = "10000.00") -> Decimal:
    return Decimal(amount)
