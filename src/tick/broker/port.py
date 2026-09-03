"""The broker port: the only surface through which an order can exist.

`PaperBroker` simulates fills locally on the user's machine. `ProfileBroker`
adapts a broker MCP only through a freshly verified profile, whose tool names
and complete contracts are discovered at runtime (CLAUDE.md invariant 7).
That is why this port uses Tick's vocabulary rather than guessing at theirs.

Every method answers with a value, never with a silence. `place` returns a
`Fill` or a `Rejected` carrying a reason a human can read; there is no `None`
meaning "something went wrong", because a caller that cannot tell an
unfilled order from a missing one has no basis to decide what to do next.

Only long orders exist here. A `Side.SELL` closes a position the account
already holds; a sell for more than is held is `Rejected`, never truncated to
a smaller sell and never allowed to open a short.
"""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import AwareDatetime, model_validator

from tick.engine import EngineModel, ExactDecimal, OrderIntent, PortfolioState
from tick.spec import Side

__all__ = [
    "BrokerPort",
    "BrokerOrder",
    "CancelResult",
    "Cancelled",
    "Fill",
    "PlaceResult",
    "RejectCode",
    "Rejected",
    "OrderOutcomeUnknown",
    "OrderState",
]


class RejectCode(StrEnum):
    """Why a broker refused. Every one has served words on the other side."""

    QUOTE_UNAVAILABLE = "quote_unavailable"
    INSUFFICIENT_CASH = "insufficient_cash"
    SELL_EXCEEDS_POSITION = "sell_exceeds_position"
    NO_POSITION_TO_SELL = "no_position_to_sell"
    UNKNOWN_ORDER = "unknown_order"
    NOT_CANCELLABLE = "not_cancellable"
    CANCEL_LIMIT_REACHED = "cancel_limit_reached"
    # Added in slice 06, for the broker that is a real brokerage.
    CAPABILITY_UNMAPPED = "capability_unmapped"
    ACCOUNT_UNAVAILABLE = "account_unavailable"
    BROKER_REFUSED = "broker_refused"
    ORDER_OUTCOME_UNKNOWN = "order_outcome_unknown"


class OrderState(StrEnum):
    """The three broker observations reconciliation can act on."""

    WORKING = "working"
    FILLED = "filled"
    CANCELLED = "cancelled"


class Fill(EngineModel):
    """An order that executed: what was traded, at what price, when."""

    order_id: str
    symbol: str
    side: Side
    qty: int
    price: ExactDecimal
    ts: AwareDatetime

    @model_validator(mode="after")
    def _check(self) -> Fill:
        if not self.order_id.strip():
            raise ValueError("a fill must carry its order id")
        if self.qty < 1:
            raise ValueError(f"fill qty ({self.qty}) must be >= 1")
        if self.price <= 0:
            raise ValueError(f"fill price ({self.price}) must be > 0")
        return self

    def describe(self) -> str:
        """Mechanical past tense, for notifications: `bought 12 XYZ at $184.20`."""
        verb = "bought" if self.side is Side.BUY else "sold"
        return f"{verb} {self.qty} {self.symbol} at ${self.price}"


class Rejected(EngineModel):
    """An order (or a cancel) the broker would not accept, and why."""

    code: RejectCode
    reason: str

    @model_validator(mode="after")
    def _check(self) -> Rejected:
        if not self.reason.strip():
            raise ValueError(f"{self.code.value}: a rejection must say why")
        return self

    def __str__(self) -> str:
        return self.reason


class OrderOutcomeUnknown(Rejected):
    """Placement returned a broker id but no terminal fill outcome."""

    broker_order_id: str

    @model_validator(mode="after")
    def _unknown(self) -> OrderOutcomeUnknown:
        if self.code is not RejectCode.ORDER_OUTCOME_UNKNOWN:
            raise ValueError("an unknown order outcome must carry ORDER_OUTCOME_UNKNOWN")
        if not self.broker_order_id.strip():
            raise ValueError("an unknown order outcome must carry the broker order id")
        return self


class BrokerOrder(EngineModel):
    """One account-scoped order observation from ``read.orders``."""

    order_id: str
    state: OrderState
    symbol: str | None
    side: Side | None
    qty: int | None
    price: Decimal | None
    at: AwareDatetime | None

    @model_validator(mode="after")
    def _terminal_fill(self) -> BrokerOrder:
        if not self.order_id.strip():
            raise ValueError("a broker order must carry its id")
        fill_values = (self.symbol, self.side, self.qty, self.price, self.at)
        if self.state is OrderState.FILLED and any(value is None for value in fill_values):
            raise ValueError(
                "a filled broker order must carry symbol, side, quantity, price, and time"
            )
        if self.qty is not None and self.qty < 1:
            raise ValueError("an observed filled quantity must be positive")
        if self.price is not None and self.price <= 0:
            raise ValueError("an observed fill price must be positive")
        return self


class Cancelled(EngineModel):
    """An order that was cancelled before it executed."""

    order_id: str
    ts: AwareDatetime


PlaceResult = Fill | Rejected
CancelResult = Cancelled | Rejected


@runtime_checkable
class BrokerPort(Protocol):
    """What the runtime may ask of a brokerage — nothing wider."""

    def state(self) -> PortfolioState:
        """Cash and long positions of the one account this broker is scoped to."""
        ...

    def place(self, intent: OrderIntent) -> PlaceResult:
        """Place `intent` as a market order, or say why it was not placed."""
        ...

    def cancel(self, order_id: str) -> CancelResult:
        """Cancel a working order, or say why it was not cancelled."""
        ...

    def orders(self) -> tuple[BrokerOrder, ...]:
        """Account-scoped broker observations used before the next tick."""
        ...
