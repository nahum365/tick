"""A market-data port backed by JSON files, with a clock the caller controls.

This is how the engine is tested and how paper runs are replayed: a series per
symbol, a `now` that the caller moves, and answers computed from nothing but
those files. It performs no network I/O of any kind — reading the files it was
handed is the only thing it touches.

**It ships no data.** The path is always supplied by the caller, there is no
bundled series and no default directory, and nothing in the product surfaces
one. Tick authors no strategies and names no securities: the fixtures under
`tests/` are test material, they are not examples, presets or starters, and no
command exposes them.

`now` is a hard boundary, not a hint. A bar timestamped after `now` does not
exist yet — that is what makes a replay honest, because a run that could see
the next bar would answer questions the live runtime cannot.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from pydantic import ValidationError

from .errors import FixtureDataError
from .market import Bar, BarsResult, Quote, QuoteResult, Unavailable

__all__ = ["FixtureMarketData"]


class FixtureMarketData:
    """Deterministic market data from a directory of JSON series files.

    Each file is `{"symbol": "XYZ", "bars": [{"ts": ..., "open": ..., "high":
    ..., "low": ..., "close": ..., "volume": ...}, ...]}`, oldest bar first,
    with every number written as a JSON string so no binary float can enter.
    """

    #: Provenance recorded on every quote this port produces.
    SOURCE = "fixture"

    def __init__(self, series: Mapping[str, tuple[Bar, ...]], now: datetime) -> None:
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("FixtureMarketData needs a timezone-aware `now`")
        self._series = dict(series)
        self._now = now

    @property
    def now(self) -> datetime:
        return self._now

    @property
    def symbols(self) -> frozenset[str]:
        return frozenset(self._series)

    def at(self, now: datetime) -> FixtureMarketData:
        """The same data seen from a different moment. The series are not re-read."""
        return FixtureMarketData(self._series, now)

    @classmethod
    def from_directory(
        cls, directory: str | os.PathLike[str], *, now: datetime
    ) -> FixtureMarketData:
        """Load every `*.json` series in `directory`."""
        root = Path(directory)
        if not root.is_dir():
            raise FixtureDataError(f"{root} is not a directory of market fixtures")
        series: dict[str, tuple[Bar, ...]] = {}
        for path in sorted(root.glob("*.json")):
            symbol, bars = cls._read_series(path)
            if symbol in series:
                raise FixtureDataError(f"{path.name}: {symbol} already loaded from another file")
            series[symbol] = bars
        if not series:
            raise FixtureDataError(f"{root} holds no *.json market fixtures")
        return cls(series, now)

    @classmethod
    def from_file(cls, path: str | os.PathLike[str], *, now: datetime) -> FixtureMarketData:
        """Load one series file."""
        symbol, bars = cls._read_series(Path(path))
        return cls({symbol: bars}, now)

    @staticmethod
    def _read_series(path: Path) -> tuple[str, tuple[Bar, ...]]:
        try:
            document = json.loads(path.read_text(encoding="utf-8"), parse_float=Decimal)
        except (OSError, json.JSONDecodeError) as exc:
            raise FixtureDataError(f"could not read market fixture {path}: {exc}") from exc
        if not isinstance(document, Mapping):
            raise FixtureDataError(f"{path}: a market fixture is a JSON object")
        symbol = document.get("symbol")
        raw_bars = document.get("bars")
        if not isinstance(symbol, str) or not symbol:
            raise FixtureDataError(f"{path}: a market fixture must name its symbol")
        if not isinstance(raw_bars, list) or not raw_bars:
            raise FixtureDataError(f"{path}: {symbol} has no bars")
        bars: list[Bar] = []
        for index, raw in enumerate(raw_bars):
            try:
                bar = Bar.model_validate(raw)
            except ValidationError as exc:
                raise FixtureDataError(
                    f"{path}: bar {index} of {symbol} is invalid: {exc}"
                ) from exc
            if bars and bar.ts <= bars[-1].ts:
                raise FixtureDataError(
                    f"{path}: {symbol} bar {index} is timestamped {bar.ts}, not after "
                    f"the previous bar ({bars[-1].ts}); a series is oldest-first "
                    f"and strictly ordered"
                )
            bars.append(bar)
        return symbol, tuple(bars)

    # ------------------------------------------------------------------
    # MarketDataPort
    # ------------------------------------------------------------------

    def _visible(self, symbol: str) -> tuple[Bar, ...] | None:
        series = self._series.get(symbol)
        if series is None:
            return None
        return tuple(bar for bar in series if bar.ts <= self._now)

    def quote(self, symbol: str) -> QuoteResult:
        """The close of the most recent bar at or before `now`."""
        visible = self._visible(symbol)
        if visible is None:
            return Unavailable(
                what=f"quote for {symbol}",
                reason=f"no fixture series is loaded for {symbol}",
            )
        if not visible:
            return Unavailable(
                what=f"quote for {symbol}",
                reason=f"no bar exists at or before {self._now.isoformat()}",
            )
        last = visible[-1]
        return Quote(symbol=symbol, price=last.close, asof=last.ts, source=self.SOURCE)

    def bars(self, symbol: str, n: int) -> BarsResult:
        """The last `n` bars at or before `now`, oldest first — exactly `n` or `Unavailable`."""
        if n < 1:
            raise ValueError(f"bars(n={n}): n must be >= 1")
        visible = self._visible(symbol)
        if visible is None:
            return Unavailable(
                what=f"bars for {symbol}",
                reason=f"no fixture series is loaded for {symbol}",
            )
        if len(visible) < n:
            return Unavailable(
                what=f"bars for {symbol}",
                reason=(
                    f"{len(visible)} bars exist at or before {self._now.isoformat()}, "
                    f"{n} were needed"
                ),
            )
        return list(visible[-n:])
