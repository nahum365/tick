"""The paper broker — a simulation that runs entirely on the user's machine.

Paper is the default and live is an explicit, logged act (CLAUDE.md invariant
2), so this is the broker every agent meets first. It is deliberately boring:
market orders fill at the current quote, cash and positions move by exact
`Decimal` arithmetic, and the sequence of order ids is deterministic so two
replays of the same fixtures produce the same record.

Four properties are enforced rather than assumed, and each has a test:

- **Cash never goes negative.** A buy that costs more than the cash on hand is
  rejected whole; it is never partially filled into an overdraft.
- **Quantities never go negative.** A sell larger than the position is rejected
  whole. Tick is long-only: it closes positions and never opens a short, so
  there is no arithmetic path to a negative quantity.
- **No fill without a price.** If the market port cannot quote the symbol, the
  order is rejected. It is never filled at the intent's estimate, at the last
  price seen, or at zero.
- **Cancels are rate-limited.** Robinhood may terminate MCP connectivity for
  patterns it judges abusive, and a runaway cancel loop is the classic one, so
  the guard lives in the broker where every caller meets it. `max_cancels` is
  required — a default would be a limit nobody chose.

Average cost is the running weighted average of what was actually paid. It is
never inferred, and a position that closes takes its average cost with it.
"""

from __future__ import annotations

from decimal import ROUND_HALF_EVEN, Decimal

from tick.engine import (
    MarketDataPort,
    OrderIntent,
    PortfolioState,
    Position,
    Unavailable,
    engine_arithmetic,
    quantize_money,
)
from tick.spec import Side

from .port import BrokerOrder, CancelResult, Fill, OrderState, PlaceResult, RejectCode, Rejected

__all__ = ["AVG_COST_QUANTUM", "PaperBroker"]

#: Average cost is a computed number, not a traded price, so it keeps more
#: places than cents — rounding it to cents would drift the basis across fills.
AVG_COST_QUANTUM = Decimal("0.000001")


class PaperBroker:
    """A local simulation of a long-only cash brokerage account.

    `market` supplies fill prices, `starting_cash` funds the account, and
    `max_cancels` bounds cancel attempts for the life of this broker. All three
    are required.
    """

    def __init__(
        self,
        market: MarketDataPort,
        starting_cash: Decimal,
        max_cancels: int,
    ) -> None:
        if isinstance(starting_cash, float):
            raise TypeError("starting_cash is a Decimal; a binary float is not exact money")
        if starting_cash <= 0:
            raise ValueError(f"starting_cash ({starting_cash}) must be > 0")
        if max_cancels < 0:
            raise ValueError(f"max_cancels ({max_cancels}) must be >= 0")
        self._market = market
        self._cash = Decimal(starting_cash)
        self._positions: dict[str, Position] = {}
        self._max_cancels = max_cancels
        self._cancels = 0
        self._sequence = 0
        self._fills: list[Fill] = []

    # ------------------------------------------------------------------
    # BrokerPort
    # ------------------------------------------------------------------

    def state(self) -> PortfolioState:
        """Cash and long positions, as a value nothing downstream can mutate."""
        return PortfolioState(cash=self._cash, positions=dict(self._positions))

    def place(self, intent: OrderIntent) -> PlaceResult:
        """Fill `intent` at the current quote, or reject it with a reason."""
        quote = self._market.quote(intent.symbol)
        if isinstance(quote, Unavailable):
            return Rejected(
                code=RejectCode.QUOTE_UNAVAILABLE,
                reason=(
                    f"{intent.describe()} was not placed: {quote}. A simulated fill at "
                    f"a price nobody quoted would be a fabricated number in the record."
                ),
            )

        price = quote.price
        with engine_arithmetic():
            cost = quantize_money(Decimal(intent.qty) * price)

        if intent.side is Side.BUY:
            if cost > self._cash:
                return Rejected(
                    code=RejectCode.INSUFFICIENT_CASH,
                    reason=(
                        f"{intent.describe()} costs ${cost} but the account holds "
                        f"${self._cash}. The order is refused whole; a cash account is "
                        f"never overdrawn and never partially filled into one."
                    ),
                )
            return self._fill_buy(intent, price, cost, quote.asof)

        held = self._positions.get(intent.symbol)
        if held is None:
            return Rejected(
                code=RejectCode.NO_POSITION_TO_SELL,
                reason=(
                    f"{intent.describe()} was not placed: the account holds no "
                    f"{intent.symbol}. Tick is long-only and never opens a short."
                ),
            )
        if intent.qty > held.qty:
            return Rejected(
                code=RejectCode.SELL_EXCEEDS_POSITION,
                reason=(
                    f"{intent.describe()} exceeds the {held.qty} {intent.symbol} held. "
                    f"The order is refused whole — never reduced to a smaller sell, and "
                    f"never allowed to go short."
                ),
            )
        return self._fill_sell(intent, price, cost, quote.asof, held)

    def cancel(self, order_id: str) -> CancelResult:
        """Paper market orders fill on placement, so there is nothing to cancel.

        The attempt still counts against `max_cancels`: the guard exists to stop
        a loop, and a loop that only ever asks for unknown ids is still a loop.
        """
        if self._cancels >= self._max_cancels:
            return Rejected(
                code=RejectCode.CANCEL_LIMIT_REACHED,
                reason=(
                    f"this session has already cancelled {self._cancels} times, the "
                    f"configured limit. Further cancels are refused: repeated "
                    f"cancellation is a pattern brokers terminate connectivity over."
                ),
            )
        self._cancels += 1
        known = any(fill.order_id == order_id for fill in self._fills)
        if not known:
            return Rejected(
                code=RejectCode.UNKNOWN_ORDER,
                reason=f"no order {order_id} was placed by this broker.",
            )
        return Rejected(
            code=RejectCode.NOT_CANCELLABLE,
            reason=(
                f"order {order_id} already filled; a paper market order executes on "
                f"placement and cannot be cancelled."
            ),
        )

    def orders(self) -> tuple[BrokerOrder, ...]:
        """Paper market orders are terminal immediately, but expose the same port."""
        return tuple(
            BrokerOrder(
                order_id=fill.order_id,
                state=OrderState.FILLED,
                symbol=fill.symbol,
                side=fill.side,
                qty=fill.qty,
                price=fill.price,
                at=fill.ts,
            )
            for fill in self._fills
        )

    # ------------------------------------------------------------------
    # Simulation internals
    # ------------------------------------------------------------------

    @property
    def fills(self) -> tuple[Fill, ...]:
        """Every fill this broker produced, in order."""
        return tuple(self._fills)

    @property
    def cancels(self) -> int:
        """Cancel attempts made against this broker."""
        return self._cancels

    def _next_order_id(self) -> str:
        self._sequence += 1
        return f"paper-{self._sequence:06d}"

    def _record(self, fill: Fill) -> Fill:
        self._fills.append(fill)
        return fill

    def _fill_buy(self, intent: OrderIntent, price: Decimal, cost: Decimal, ts) -> Fill:
        existing = self._positions.get(intent.symbol)
        with engine_arithmetic():
            if existing is None:
                qty = intent.qty
                avg_cost = price
            else:
                qty = existing.qty + intent.qty
                paid = existing.avg_cost * existing.qty + price * intent.qty
                avg_cost = paid / qty
            avg_cost = avg_cost.quantize(AVG_COST_QUANTUM, rounding=ROUND_HALF_EVEN)
            self._cash = self._cash - cost
        self._positions[intent.symbol] = Position(symbol=intent.symbol, qty=qty, avg_cost=avg_cost)
        return self._record(
            Fill(
                order_id=self._next_order_id(),
                symbol=intent.symbol,
                side=Side.BUY,
                qty=intent.qty,
                price=price,
                ts=ts,
            )
        )

    def _fill_sell(
        self,
        intent: OrderIntent,
        price: Decimal,
        proceeds: Decimal,
        ts,
        held: Position,
    ) -> Fill:
        remaining = held.qty - intent.qty
        with engine_arithmetic():
            self._cash = self._cash + proceeds
        if remaining == 0:
            del self._positions[intent.symbol]
        else:
            # Selling does not change what the remaining shares cost.
            self._positions[intent.symbol] = Position(
                symbol=intent.symbol, qty=remaining, avg_cost=held.avg_cost
            )
        return self._record(
            Fill(
                order_id=self._next_order_id(),
                symbol=intent.symbol,
                side=Side.SELL,
                qty=intent.qty,
                price=price,
                ts=ts,
            )
        )
