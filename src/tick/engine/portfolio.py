"""The account as the engine sees it: cash, long positions, and equity.

Equity is a **join**, not a stored number. The broker supplies quantities and
cash; market data supplies prices; equity exists only where the two meet. So
`equity()` takes the quotes as an argument and returns `Unavailable` when any
held symbol has no price — it never treats a missing price as zero, which
would report a funded account as empty and let a percentage-of-equity rule
size an order against a fiction.

Positions are long-only. There is no short side anywhere in Tick: a Robinhood
Agentic account may place long equity orders, and a sell closes a position it
already has (see `engine/decisions.py` and `broker/paper.py`, which refuse
rather than truncate).
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal

from pydantic import model_validator

from .base import EngineModel, ExactDecimal, engine_arithmetic, quantize_money
from .market import QuoteResult, Unavailable

__all__ = ["PortfolioState", "Position"]


class Position(EngineModel):
    """A long holding: whole shares at an average cost."""

    symbol: str
    qty: int
    avg_cost: ExactDecimal

    @model_validator(mode="after")
    def _check(self) -> Position:
        if not self.symbol.strip():
            raise ValueError("a position must name its symbol")
        if self.qty < 1:
            raise ValueError(
                f"position {self.symbol}: qty ({self.qty}) must be >= 1; "
                f"Tick is long-only, and a closed position is not held at all"
            )
        if self.avg_cost <= 0:
            raise ValueError(f"position {self.symbol}: avg_cost ({self.avg_cost}) must be > 0")
        return self


class PortfolioState(EngineModel):
    """Cash and long positions at one moment, as reported by a broker."""

    cash: ExactDecimal
    positions: Mapping[str, Position]

    @model_validator(mode="after")
    def _check(self) -> PortfolioState:
        if self.cash < 0:
            raise ValueError(f"cash ({self.cash}) must be >= 0; a cash account cannot go negative")
        for symbol, position in self.positions.items():
            if position.symbol != symbol:
                raise ValueError(f"positions[{symbol!r}] holds a position in {position.symbol!r}")
        return self

    @property
    def symbols(self) -> frozenset[str]:
        """Every symbol held. Needed to price the account."""
        return frozenset(self.positions)

    def qty(self, symbol: str) -> int:
        """Shares held of `symbol`; zero when none are."""
        position = self.positions.get(symbol)
        return position.qty if position is not None else 0

    def market_value(self, symbol: str, quote: QuoteResult) -> Decimal | Unavailable:
        """What the holding in `symbol` is worth at `quote`."""
        held = self.qty(symbol)
        if held == 0:
            return Decimal(0)
        if isinstance(quote, Unavailable):
            # The quote already names the symbol; renaming it would say it twice.
            return quote
        with engine_arithmetic():
            return quantize_money(Decimal(held) * quote.price)

    def equity(self, quotes: Mapping[str, QuoteResult]) -> Decimal | Unavailable:
        """Cash plus the value of every holding, or `Unavailable` if any price is missing.

        Refusing the whole figure on one missing price is the point. A partial
        equity is a smaller equity, and a smaller equity silently shrinks every
        percentage computed from it.
        """
        with engine_arithmetic():
            total = self.cash
            for symbol in sorted(self.positions):
                quote = quotes.get(symbol)
                if quote is None:
                    return Unavailable(
                        what="equity",
                        reason=f"no quote was fetched for the held symbol {symbol}",
                    )
                if isinstance(quote, Unavailable):
                    return Unavailable(
                        what="equity",
                        reason=f"{quote.what} is unavailable ({quote.reason}), "
                        f"so the account cannot be priced",
                    )
                total += Decimal(self.positions[symbol].qty) * quote.price
            return quantize_money(total)
