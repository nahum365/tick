"""Indicator computation from bars — deterministic, exact, and short-history-safe.

Every function here takes a closing-price series (oldest first, `Decimal`) and
returns either a `Decimal` or an `Unavailable`. None of them shortens a window
to fit the data it was given: an average of "the last 20 closes" computed from
12 closes is a different number wearing the same name, which is the fabrication
CLAUDE.md invariant 5 forbids. Too little history is `Unavailable`, and the
rule that needed it refuses.

**The definitions, exactly.** Each is written down because "SMA" names a family
and the record has to mean one thing:

| Indicator | Value | Bars needed |
|---|---|---|
| `price` | the latest bar's close | 1 |
| `sma(n)` | mean of the last `n` closes | `n` |
| `ema(n)` | over a `2n` window: seed = SMA of its first `n` closes, then
  `close·k + prev·(1−k)` with `k = 2/(n+1)` | `2n` |
| `change_pct(k)` | `(close[-1] − close[-1−k]) / close[-1−k] × 100` | `k + 1` |

The EMA window is fixed at `2n` deliberately. A textbook EMA depends on every
bar since the series began, so its value would drift with how much history a
port happened to return; pinning the window makes the same bars produce the
same number on every machine and in every replay of the record.

**Crossings.** `crosses_above` is true when the left side was at or below the
right side on the previous bar and is strictly above it on this one:

    left[t-1] <= right[t-1]  and  left[t] > right[t]

`crosses_below` is the mirror (`>=` then `<`). Written this way a cross fires
on exactly one bar: once left is above, the previous bar is above too, so the
first clause is false on every later bar. Equality is treated as "not yet
crossed", so a series that touches its average and then rises fires once, at
the bar where it rises.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal

from tick.spec import (
    ChangePct,
    Ema,
    IndicatorNode,
    NumberLiteral,
    Price,
    Sma,
)

from .base import engine_arithmetic, quantize_value
from .market import Unavailable

__all__ = [
    "EMA_WINDOW_MULTIPLE",
    "change_pct",
    "crosses_above",
    "crosses_below",
    "ema",
    "latest_close",
    "required_bars",
    "sma",
]

#: `ema(n)` is computed over the last `EMA_WINDOW_MULTIPLE * n` closes.
EMA_WINDOW_MULTIPLE = 2


def _too_short(what: str, needed: int, have: int) -> Unavailable:
    return Unavailable(
        what=what,
        reason=f"needs {needed} bars of history, {have} available",
    )


def latest_close(closes: Sequence[Decimal]) -> Decimal | Unavailable:
    """The most recent close — what `price` means inside a condition."""
    if len(closes) < 1:
        return _too_short("price", 1, len(closes))
    return quantize_value(closes[-1])


def sma(closes: Sequence[Decimal], n: int) -> Decimal | Unavailable:
    """Mean of the last `n` closes."""
    if n < 1:
        raise ValueError(f"sma({n}): n must be >= 1")
    if len(closes) < n:
        return _too_short(f"sma({n})", n, len(closes))
    with engine_arithmetic():
        total = sum(closes[-n:], start=Decimal(0))
        return quantize_value(total / n)


def ema(closes: Sequence[Decimal], n: int) -> Decimal | Unavailable:
    """Exponential moving average over a fixed `2n`-bar window (see module doc)."""
    if n < 1:
        raise ValueError(f"ema({n}): n must be >= 1")
    needed = EMA_WINDOW_MULTIPLE * n
    if len(closes) < needed:
        return _too_short(f"ema({n})", needed, len(closes))
    with engine_arithmetic():
        window = closes[-needed:]
        value = sum(window[:n], start=Decimal(0)) / n
        k = Decimal(2) / (Decimal(n) + 1)
        for close in window[n:]:
            value = close * k + value * (1 - k)
        return quantize_value(value)


def change_pct(closes: Sequence[Decimal], n_bars: int) -> Decimal | Unavailable:
    """Percent change over the last `n_bars` bars; `5` means +5%."""
    if n_bars < 1:
        raise ValueError(f"change_pct({n_bars}): n_bars must be >= 1")
    needed = n_bars + 1
    if len(closes) < needed:
        return _too_short(f"change_pct({n_bars})", needed, len(closes))
    base = closes[-needed]
    if base == 0:
        return Unavailable(
            what=f"change_pct({n_bars})",
            reason="the reference close is 0; a percentage change from it has no value",
        )
    with engine_arithmetic():
        return quantize_value((closes[-1] - base) / base * 100)


def crosses_above(
    left_now: Decimal, left_prev: Decimal, right_now: Decimal, right_prev: Decimal
) -> bool:
    """At or below on the previous bar, strictly above on this one."""
    return left_prev <= right_prev and left_now > right_now


def crosses_below(
    left_now: Decimal, left_prev: Decimal, right_now: Decimal, right_prev: Decimal
) -> bool:
    """At or above on the previous bar, strictly below on this one."""
    return left_prev >= right_prev and left_now < right_now


def required_bars(node: IndicatorNode, *, previous: bool) -> int:
    """How many bars this operand needs; one more when its previous value is needed.

    Operands with no per-bar history (cash, a position, the weekday, a literal)
    need none — the grammar already refuses to cross on them.
    """
    if isinstance(node, Price):
        base = 1
    elif isinstance(node, Sma):
        base = node.n
    elif isinstance(node, Ema):
        base = EMA_WINDOW_MULTIPLE * node.n
    elif isinstance(node, ChangePct):
        base = node.n_bars + 1
    elif isinstance(node, NumberLiteral):
        return 0
    else:
        return 0
    return base + 1 if previous else base
