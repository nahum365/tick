"""Market data: the port, its values, and the one shape a missing number takes.

`MarketDataPort` is the only way the engine learns a price. It has exactly two
methods and both of them may answer **`Unavailable`** — a typed value carrying
what was asked for and why it could not be answered. That is CLAUDE.md
invariant 5 given a type: there is no `None` to be read as zero, no `0.0`
standing in for "we did not get a price", and nothing downstream can consume an
`Unavailable` by accident, because it is not a number.

Provenance is explicit for the same reason. A `Quote` carries its `asof` and
its `source`, so a fill priced from a quote can always be traced back to where
that price came from and how old it was.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import AwareDatetime, model_validator

from .base import EngineModel, ExactDecimal

__all__ = [
    "Bar",
    "BarsResult",
    "MarketDataPort",
    "Quote",
    "QuoteResult",
    "Unavailable",
]


class Unavailable(EngineModel):
    """A number that could not be obtained, and why.

    Never a zero, never a guess. `what` names the thing that is missing in the
    words a human would use for it (`"quote for XYZ"`, `"sma(20)"`); `reason`
    says what stopped it.
    """

    what: str
    reason: str

    @model_validator(mode="after")
    def _check(self) -> Unavailable:
        if not self.what.strip():
            raise ValueError("Unavailable.what must name what is missing")
        if not self.reason.strip():
            raise ValueError("Unavailable.reason must say why it is missing")
        return self

    def scoped(self, symbol: str) -> Unavailable:
        """The same unavailability, named for a symbol: `sma(20) for XYZ`."""
        return Unavailable(what=f"{self.what} for {symbol}", reason=self.reason)

    def __str__(self) -> str:
        return f"{self.what} is unavailable: {self.reason}"


class Bar(EngineModel):
    """One OHLCV bar. `ts` is the bar's closing timestamp, timezone-aware."""

    ts: AwareDatetime
    open: ExactDecimal
    high: ExactDecimal
    low: ExactDecimal
    close: ExactDecimal
    volume: int

    @model_validator(mode="after")
    def _check(self) -> Bar:
        for name in ("open", "high", "low", "close"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"bar {name} ({value}) must be > 0")
        if self.high < self.low:
            raise ValueError(f"bar high ({self.high}) is below its low ({self.low})")
        if not (self.low <= self.open <= self.high):
            raise ValueError(f"bar open ({self.open}) is outside its low/high range")
        if not (self.low <= self.close <= self.high):
            raise ValueError(f"bar close ({self.close}) is outside its low/high range")
        if self.volume < 0:
            raise ValueError(f"bar volume ({self.volume}) must be >= 0")
        return self


class Quote(EngineModel):
    """The last trade price of a symbol, with where it came from and when."""

    symbol: str
    price: ExactDecimal
    asof: AwareDatetime
    source: str

    @model_validator(mode="after")
    def _check(self) -> Quote:
        if not self.symbol.strip():
            raise ValueError("a quote must name its symbol")
        if self.price <= 0:
            raise ValueError(f"quote price ({self.price}) must be > 0")
        if not self.source.strip():
            raise ValueError("a quote must name its source; provenance is not optional")
        return self


#: What `quote()` answers: a price, or the reason there is none.
QuoteResult = Quote | Unavailable

#: What `bars()` answers: exactly `n` bars oldest-first, or the reason there
#: are not that many.
BarsResult = list[Bar] | Unavailable


@runtime_checkable
class MarketDataPort(Protocol):
    """Everything the engine is allowed to know about the market.

    Implementations are local (fixtures, slice 02) or remote (the broker's own
    market data, slice 06). Neither may answer a missing number with a zero.
    """

    def quote(self, symbol: str) -> QuoteResult:
        """The latest trade price of `symbol`, or why there is none."""
        ...

    def bars(self, symbol: str, n: int) -> BarsResult:
        """The last `n` bars of `symbol`, oldest first — exactly `n`, or `Unavailable`.

        A port that has fewer than `n` bars answers `Unavailable`. It never
        answers a shorter list, because a shorter list silently changes what
        every indicator computed from it means.
        """
        ...
