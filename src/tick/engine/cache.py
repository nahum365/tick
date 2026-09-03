"""The per-tick market-data cache — scoped reads, asked once, never retried.

Robinhood's Customer Agreement §29 lets them terminate MCP connectivity for
undefined "excessive market data usage", so how often Tick asks is a product
constraint, not an optimisation. Two rules do that work here:

- **Scope.** A tick may read only the symbols the user's spec named, plus the
  symbols the account actually holds (which have to be priced to know what the
  account is worth). Anything else raises `SymbolOutsideScope` rather than
  quietly widening what Tick reads.
- **Once.** Every distinct question is asked at most once per tick and the
  answer — including `Unavailable` — is kept. A failure is never retried
  inside a tick: on a data failure the runtime stops and tells the user, it
  does not hammer the socket (CLAUDE.md invariant 8, fail safe).

Bars are cached as the longest series fetched for a symbol, so a rule wanting
20 bars after another asked for 50 is served from what is already in hand.
"""

from __future__ import annotations

from .errors import MarketDataContractError, SymbolOutsideScope
from .market import Bar, BarsResult, MarketDataPort, QuoteResult, Unavailable

__all__ = ["TickMarketCache"]


class TickMarketCache:
    """One tick's view of the market: scoped, memoised, and counted.

    `permitted` is the exact set of symbols this tick may read. It is required
    — a cache that defaulted to "everything" would be no scope at all.
    """

    def __init__(self, market: MarketDataPort, permitted: frozenset[str]) -> None:
        if not permitted:
            raise ValueError("a tick cache needs at least one permitted symbol")
        self._market = market
        self._permitted = permitted
        self._quotes: dict[str, QuoteResult] = {}
        self._series: dict[str, list[Bar]] = {}
        self._bars_unavailable: dict[str, tuple[int, Unavailable]] = {}
        #: Underlying port calls made, for tests and for the record's data cost.
        self.quote_calls = 0
        self.bars_calls = 0

    @property
    def permitted(self) -> frozenset[str]:
        return self._permitted

    def cached_quotes(self) -> dict[str, QuoteResult]:
        """Every quote already fetched this tick. Asking does not fetch more."""
        return dict(self._quotes)

    def _check_scope(self, symbol: str) -> None:
        if symbol not in self._permitted:
            raise SymbolOutsideScope(
                f"{symbol} is not in this tick's scope "
                f"({', '.join(sorted(self._permitted))}); Tick reads only the spec's "
                f"universe and the symbols the account holds"
            )

    def quote(self, symbol: str) -> QuoteResult:
        """The symbol's quote, fetched at most once per tick."""
        self._check_scope(symbol)
        cached = self._quotes.get(symbol)
        if cached is not None:
            return cached
        self.quote_calls += 1
        result = self._market.quote(symbol)
        self._quotes[symbol] = result
        return result

    def bars(self, symbol: str, n: int) -> BarsResult:
        """The last `n` bars, served from the longest series already fetched."""
        self._check_scope(symbol)
        if n < 1:
            raise ValueError(f"bars(n={n}): n must be >= 1")

        cached = self._series.get(symbol)
        if cached is not None and len(cached) >= n:
            return list(cached[-n:])

        failed = self._bars_unavailable.get(symbol)
        if failed is not None and n >= failed[0]:
            # A shorter history was already refused; asking for more cannot help.
            return failed[1]

        self.bars_calls += 1
        result = self._market.bars(symbol, n)
        if isinstance(result, Unavailable):
            previous = self._bars_unavailable.get(symbol)
            if previous is None or n < previous[0]:
                self._bars_unavailable[symbol] = (n, result)
            return result
        if len(result) != n:
            raise MarketDataContractError(
                f"bars({symbol!r}, {n}) returned {len(result)} bars; a port returns "
                f"exactly n bars or Unavailable"
            )
        if cached is None or len(result) > len(cached):
            self._series[symbol] = list(result)
        return result
