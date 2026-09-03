"""The cage — the deterministic limits, applied to intents from any source.

`apply_cage` is a pure function over a batch of `OrderIntent`s. It knows
nothing about who proposed them, which is the whole design: today they come
from spec rules, tomorrow from a model agent whose judgment nobody can audit,
and the limits have to be the same code either way. The model has judgment;
the cage has authority (CLAUDE.md invariant 3), and authority that could be
bypassed by proposing orders through a different door is not authority.

**The cage bounds what an agent may take ON, never what it may close.** Every
size and concentration limit applies to buys. A sell is an exit, and a limit
that blocks an exit is a limit that traps the user in the position it was
supposed to protect them from — the inverted fail-safe. So a $9,000 sell passes
a $2,500 order ceiling, a sell passes while a drawdown halt is stopping every
buy, and a sell passes when equity cannot be computed at all. The one check
that applies to both sides is the session, because outside it nothing can
execute anyway.

**Precedence is fixed and stated**, so two simultaneously-true reasons always
produce the same recorded one: session → equity → drawdown → order notional →
position concentration → position count.

The regular-hours check is the `session` argument rather than a clock in here.
The engine has no clock; slice 04 owns the ET market calendar and passes the
answer in. `SessionState.NOT_EVALUATED` is the honest value for a caller that
has no clock yet — it is spelled out rather than defaulted, so nobody acquires
"the market was open" by omission.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from enum import StrEnum

from tick.spec import Cage, Side

from .base import EngineModel, engine_arithmetic, quantize_value
from .decisions import OrderIntent
from .market import Unavailable

__all__ = [
    "CageCode",
    "CageOutcome",
    "CageRejection",
    "SessionState",
    "apply_cage",
]


class SessionState(StrEnum):
    """Whether the market is open, as judged by whoever owns a clock."""

    OPEN = "open"
    CLOSED = "closed"
    #: No clock was consulted. Slice 04 replaces this with a real answer.
    NOT_EVALUATED = "not_evaluated"


class CageCode(StrEnum):
    """Which limit stopped an order."""

    SESSION_CLOSED = "session_closed"
    EQUITY_UNAVAILABLE = "equity_unavailable"
    PRICE_UNAVAILABLE = "price_unavailable"
    DAILY_DRAWDOWN_HALT = "daily_drawdown_halt"
    MAX_ORDER_NOTIONAL = "max_order_notional"
    MAX_POSITION_PCT = "max_position_pct"
    MAX_POSITIONS = "max_positions"


class CageRejection(EngineModel):
    """One intent the cage refused, and the sentence a human reads."""

    intent: OrderIntent
    code: CageCode
    reason: str

    def __str__(self) -> str:
        return f"{self.intent.describe()} rejected by the cage: {self.reason}"


class CageOutcome(EngineModel):
    """What survived the cage, and what did not."""

    allowed: tuple[OrderIntent, ...]
    rejected: tuple[CageRejection, ...]


def apply_cage(
    cage: Cage,
    intents: Sequence[OrderIntent],
    *,
    held: Mapping[str, int],
    prices: Mapping[str, Decimal],
    equity: Decimal | Unavailable,
    day_start_equity: Decimal,
    session: SessionState,
) -> CageOutcome:
    """Split `intents` into what the cage permits and what it refuses.

    `held` is the account's share count per symbol, `prices` the quote price
    per symbol this tick, `equity` the account's value (or why it is unknown),
    and `day_start_equity` what it was worth when the session opened. All are
    required: a cage that computed its own inputs would be a cage with an
    opinion, and every one of these numbers has to come from a source that can
    say "unavailable".
    """
    if day_start_equity <= 0:
        raise ValueError(
            f"day_start_equity ({day_start_equity}) must be > 0; the daily drawdown "
            f"limit is a percentage of it and cannot be computed from zero"
        )

    allowed: list[OrderIntent] = []
    rejected: list[CageRejection] = []
    pending_qty: dict[str, int] = {}
    open_symbols = {symbol for symbol, qty in held.items() if qty > 0}

    halted = _drawdown_breached(cage, equity, day_start_equity)

    for intent in intents:
        rejection = _judge(
            intent,
            cage,
            held=held,
            prices=prices,
            equity=equity,
            day_start_equity=day_start_equity,
            session=session,
            halted=halted,
            pending_qty=pending_qty,
            open_symbols=open_symbols,
        )
        if rejection is not None:
            rejected.append(rejection)
            continue
        allowed.append(intent)
        if intent.side is Side.BUY:
            pending_qty[intent.symbol] = pending_qty.get(intent.symbol, 0) + intent.qty
            open_symbols.add(intent.symbol)

    return CageOutcome(allowed=tuple(allowed), rejected=tuple(rejected))


def _drawdown_breached(
    cage: Cage,
    equity: Decimal | Unavailable,
    day_start_equity: Decimal,
) -> bool:
    if isinstance(equity, Unavailable):
        return False  # handled per-intent: buys refuse on unknown equity anyway
    with engine_arithmetic():
        floor = day_start_equity * (1 - cage.max_daily_drawdown_pct / 100)
        return equity <= floor


def _judge(
    intent: OrderIntent,
    cage: Cage,
    *,
    held: Mapping[str, int],
    prices: Mapping[str, Decimal],
    equity: Decimal | Unavailable,
    day_start_equity: Decimal,
    session: SessionState,
    halted: bool,
    pending_qty: Mapping[str, int],
    open_symbols: set[str],
) -> CageRejection | None:
    if session is SessionState.CLOSED:
        return CageRejection(
            intent=intent,
            code=CageCode.SESSION_CLOSED,
            reason=(
                f"the market is closed and this spec allows {cage.allowed_session.value} "
                f"only, so {intent.describe()} is not placed."
            ),
        )

    if intent.side is Side.SELL:
        # An exit is never blocked by a size, concentration or drawdown limit.
        return None

    if isinstance(equity, Unavailable):
        return CageRejection(
            intent=intent,
            code=CageCode.EQUITY_UNAVAILABLE,
            reason=(
                f"{equity}, so the cage cannot check what {intent.describe()} would do "
                f"to the account's concentration or drawdown. Buys stop; exits do not."
            ),
        )

    if halted:
        with engine_arithmetic():
            drop = quantize_value((day_start_equity - equity) / day_start_equity * 100)
        return CageRejection(
            intent=intent,
            code=CageCode.DAILY_DRAWDOWN_HALT,
            reason=(
                f"equity is down {drop}% today against a cage limit of "
                f"{cage.max_daily_drawdown_pct}%; all buying is halted for the session. "
                f"{intent.describe()} is not placed."
            ),
        )

    if intent.est_notional > cage.max_order_notional:
        return CageRejection(
            intent=intent,
            code=CageCode.MAX_ORDER_NOTIONAL,
            reason=(
                f"{intent.describe()} is ${intent.est_notional}, over the cage's "
                f"${cage.max_order_notional} per-order ceiling."
            ),
        )

    price = prices.get(intent.symbol)
    if price is None:
        return CageRejection(
            intent=intent,
            code=CageCode.PRICE_UNAVAILABLE,
            reason=(
                f"no price for {intent.symbol} this tick, so the cage cannot check what "
                f"{intent.describe()} would do to concentration. The order is not placed."
            ),
        )

    projected_qty = held.get(intent.symbol, 0) + pending_qty.get(intent.symbol, 0) + intent.qty
    with engine_arithmetic():
        projected_pct = quantize_value(Decimal(projected_qty) * price / equity * 100)
    if projected_pct > cage.max_position_pct:
        return CageRejection(
            intent=intent,
            code=CageCode.MAX_POSITION_PCT,
            reason=(
                f"{intent.describe()} would make {intent.symbol} {projected_pct}% of "
                f"equity, over the cage's {cage.max_position_pct}% per-position limit."
            ),
        )

    if intent.symbol not in open_symbols and len(open_symbols) >= cage.max_positions:
        return CageRejection(
            intent=intent,
            code=CageCode.MAX_POSITIONS,
            reason=(
                f"the account already holds {len(open_symbols)} positions and the cage "
                f"allows {cage.max_positions}; opening {intent.symbol} would be one too "
                f"many."
            ),
        )

    return None
