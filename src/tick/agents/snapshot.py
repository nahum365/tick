"""The snapshot: everything the model is told, and nothing Tick made up.

One tick, rendered as a JSON object: the moment, the universe the user's own
document named, a quote for each of those symbols, what the account actually
holds, its cash, its equity, and the cage the runtime will enforce whatever the
model answers.

Three properties, and each is a rule rather than a habit.

**No number is fabricated.** A quote Tick could not get renders as
`{"unavailable": "<reason>"}` — never a zero, never a stale price, never a
silently dropped symbol. The model is told what is missing in the same words
the record will carry, so an agent that decides anyway is deciding on a stated
absence rather than on a fiction (CLAUDE.md invariant 5).

**Money is a string.** Every `Decimal` crosses into JSON as its exact decimal
string, the way it does in the spec and in the record. A JSON float is a binary
approximation and this product has refused those since slice 01.

**Nothing here is an instruction.** The snapshot is a table of facts. It
contains no ranking, no shortlist, no "candidates", no suggestion of what to do
with any of it — Tick contributes the numbers and the limits, and the whole of
what to do with them comes from the user's own instructions file.

Robinhood-sourced numbers reach this object and go exactly one place: into the
request the USER's own model key pays for, from the user's own machine. There
is no Tick-side copy and no second destination; the Market Data Addendum's
no-redistribution rule is the reason and the architecture is the enforcement.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from decimal import Decimal
from typing import Any

from tick.engine import PortfolioState, Quote, QuoteResult, Unavailable
from tick.spec import Cage, canonical_dumps

__all__ = ["build_snapshot", "snapshot_json"]


def build_snapshot(
    *,
    universe: list[str],
    cadence_kind: str,
    quotes: Mapping[str, QuoteResult],
    portfolio: PortfolioState,
    equity: Decimal | Unavailable,
    cage: Cage,
    now: datetime,
) -> dict[str, Any]:
    """One tick as plain JSON. Every argument is required; nothing is derived.

    `quotes` is the tick's cache — asked once per symbol, for the universe and
    the held symbols and nothing wider — so building a snapshot never widens
    what Tick reads from a broker that may terminate connectivity for usage it
    judges excessive.
    """
    return {
        "as_of": now.isoformat(),
        "cadence": cadence_kind,
        "universe": sorted(universe),
        "quotes": {symbol: _quote(quotes.get(symbol)) for symbol in sorted(universe)},
        "positions": [
            {
                "symbol": symbol,
                "quantity": portfolio.positions[symbol].qty,
                "average_cost": str(portfolio.positions[symbol].avg_cost),
                "quote": _quote(quotes.get(symbol)),
            }
            for symbol in sorted(portfolio.positions)
        ],
        "cash": str(portfolio.cash),
        "equity": _number(equity),
        "cage": {
            "max_position_pct": str(cage.max_position_pct),
            "max_positions": cage.max_positions,
            "max_order_notional": str(cage.max_order_notional),
            "max_daily_drawdown_pct": str(cage.max_daily_drawdown_pct),
            "allowed_session": cage.allowed_session.value,
            "long_only": True,
            "enforced_by": (
                "the runtime, after your reply and outside your reach. An intent that "
                "breaks one of these limits is rejected and recorded; it is never "
                "reduced to one that fits."
            ),
        },
    }


def snapshot_json(snapshot: Mapping[str, Any]) -> str:
    """The snapshot's one encoding — the same canonical form the record uses."""
    return canonical_dumps(dict(snapshot))


def _quote(result: QuoteResult | None) -> dict[str, Any]:
    """A price with its provenance, or the stated reason there is none."""
    if result is None:
        return {"unavailable": "no quote was fetched for this symbol on this tick"}
    if isinstance(result, Unavailable):
        return {"unavailable": result.reason}
    assert isinstance(result, Quote)
    return {
        "price": str(result.price),
        "as_of": result.asof.isoformat(),
        "source": result.source,
    }


def _number(value: Decimal | Unavailable) -> dict[str, Any]:
    if isinstance(value, Unavailable):
        return {"unavailable": value.reason}
    return {"value": str(value)}
