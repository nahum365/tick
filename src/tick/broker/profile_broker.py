"""The broker adapter over a freshly verified, per-tool profile binding.

``ProfileBroker`` cannot be built from a persisted profile.  It accepts only a
``VerifiedSessionProfile`` created from an open session whose complete live
inventory was fetched and compared.  The adapter checks authorization again at
the final call boundary, so a caller cannot bypass drift handling by skipping a
CLI readiness check.  Changed reads return ``Unavailable`` or raise before
``tools/call``; unrelated exact matches remain usable.

Before every mutating call the adapter refreshes the inventory, verifies the
exact tool contract, rechecks the kill switch, validates the fully rendered
arguments against the pinned input schema, and only then sends.  MCP has no
atomic "call only if contract equals X" operation: Tick guarantees the latest
advertised contract matched, not that a malicious server behaves like it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Literal

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from tick.engine import (
    Bar,
    BarsResult,
    OrderIntent,
    PortfolioState,
    Position,
    Quote,
    QuoteResult,
    Unavailable,
)
from tick.spec import Side

from .errors import BrokerUnavailable, CapabilityUnmapped, ToolResultUnreadable
from .port import (
    BrokerOrder,
    Cancelled,
    CancelResult,
    Fill,
    OrderOutcomeUnknown,
    OrderState,
    PlaceResult,
    RejectCode,
    Rejected,
)
from .profile import Category, ProfileTool, VerifiedSessionProfile, server_host
from .toolmap import decimal_at, dig, text_at, timestamp_at, whole_at

__all__ = ["ProfileBroker"]


class _ProfileQuote(Quote):
    """A broker quote carrying the profile's explicit redistribution class."""

    price_source: str
    data_class: Literal["display_only"]


class _ProfileBar(Bar):
    """A broker bar carrying its host provenance and display-only class."""

    price_source: str
    data_class: Literal["display_only"]


class ProfileBroker:
    """Market data and orders through one verified profile and account."""

    def __init__(
        self,
        verified: VerifiedSessionProfile,
        *,
        max_cancels: int,
        kill_switch: Callable[[], bool],
        approval_mode: str,
    ) -> None:
        if not isinstance(verified, VerifiedSessionProfile):
            raise TypeError(
                "ProfileBroker requires a VerifiedSessionProfile from an open, fully "
                "listed session; a raw persisted profile cannot authorize calls"
            )
        if max_cancels < 0:
            raise ValueError(f"max_cancels ({max_cancels}) must be >= 0")
        self._verified = verified
        self._max_cancels = max_cancels
        self._kill_switch = kill_switch
        if approval_mode not in {"each", "standing"}:
            raise ValueError("approval_mode must be each or standing")
        self._approval_mode = approval_mode
        self._last_place_called = False
        self._cancels = 0
        hook = getattr(verified.session, "on_tools_changed", None)
        if callable(hook):
            hook(verified.revoke)

    @property
    def account_id(self) -> str:
        account_id = self._verified.profile.account_id
        if account_id is None:
            raise CapabilityUnmapped(
                "no eligible broker account is selected. Read accounts and select one before "
                "using an account-scoped capability."
            )
        return account_id

    @property
    def profile_hash(self) -> str:
        return self._verified.profile.profile_hash

    @property
    def inventory_hash(self) -> str:
        return self._verified.inventory_hash

    @property
    def profile_sanction(self) -> str:
        return self._verified.profile.sanction

    @property
    def broker_name(self) -> str:
        return "profile"

    @property
    def data_class(self) -> str:
        return self._verified.profile.data_class

    @property
    def price_source(self) -> str:
        return server_host(self._verified.profile.server)

    @property
    def cancels(self) -> int:
        return self._cancels

    def quote(self, symbol: str) -> QuoteResult:
        what = f"quote for {symbol}"
        try:
            mapping, payload = self._call(
                Category.READ_QUOTE, {"symbol": symbol}, require_proof=False
            )
        except (CapabilityUnmapped, ToolResultUnreadable) as exc:
            return Unavailable(
                what=what,
                reason=f"{exc} You can inspect the exact tool with `tick broker status`.",
            )
        price = decimal_at(payload, mapping.result["price"], what)
        if isinstance(price, Unavailable):
            return price
        if price <= 0:
            return Unavailable(what=what, reason=f"the broker quoted {price}, which is not a price")
        asof = timestamp_at(payload, mapping.result["asof"], f"{what} timestamp")
        if isinstance(asof, Unavailable):
            return asof
        return _ProfileQuote(
            symbol=symbol,
            price=price,
            asof=asof,
            source=self.price_source,
            price_source=self.price_source,
            data_class=self.data_class,
        )

    def bars(self, symbol: str, n: int) -> BarsResult:
        what = f"{n} bars for {symbol}"
        if n < 1:
            return Unavailable(what=what, reason="the requested bar count must be at least one")
        try:
            self._verified.profile.mapping_for(Category.READ_HISTORY)
        except CapabilityUnmapped:
            return Unavailable(
                what=what,
                reason="no history tool is mapped in this profile; confirm one or use paper data",
            )
        try:
            mapping, payload = self._call(
                Category.READ_HISTORY,
                {"symbol": symbol, "count": n},
                require_proof=False,
            )
        except (CapabilityUnmapped, ToolResultUnreadable) as exc:
            return Unavailable(what=what, reason=str(exc))
        rows = dig(payload, mapping.result["items"])
        if not isinstance(rows, Sequence) or isinstance(rows, str | bytes):
            return Unavailable(
                what=what,
                reason=f"the broker answer has no list at {mapping.result['items']!r}",
            )
        if len(rows) != n:
            return Unavailable(
                what=what,
                reason=f"the broker returned {len(rows)} bars; exactly {n} are required",
            )
        bars: list[Bar] = []
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                return Unavailable(what=what, reason=f"bar {index} is not an object")
            values = {
                role: decimal_at(row, mapping.result[role], f"{role} of bar {index}")
                for role in ("open", "high", "low", "close")
            }
            volume = whole_at(row, mapping.result["volume"], f"volume of bar {index}")
            at = timestamp_at(row, mapping.result["timestamp"], f"timestamp of bar {index}")
            unavailable = [
                value for value in (*values.values(), volume, at) if isinstance(value, Unavailable)
            ]
            if unavailable:
                return Unavailable(
                    what=what, reason="; ".join(value.reason for value in unavailable)
                )
            bars.append(
                _ProfileBar(
                    ts=at,  # type: ignore[arg-type]
                    open=values["open"],  # type: ignore[arg-type]
                    high=values["high"],  # type: ignore[arg-type]
                    low=values["low"],  # type: ignore[arg-type]
                    close=values["close"],  # type: ignore[arg-type]
                    volume=volume,  # type: ignore[arg-type]
                    price_source=self.price_source,
                    data_class=self.data_class,
                )
            )
        return bars

    def state(self) -> PortfolioState:
        # Authorize both dependencies before either call. A drifted positions
        # tool must not leave a usable cash number behind in memory or a record.
        self._verified.mapping_for(Category.READ_BALANCES, require_proof=False)
        self._verified.mapping_for(Category.READ_POSITIONS, require_proof=False)
        return PortfolioState(cash=self._cash(), positions=self._positions())

    def place(self, intent: OrderIntent) -> PlaceResult:
        self._last_place_called = False
        try:
            for category in (
                Category.ORDER_PLACE,
                Category.READ_QUOTE,
                Category.READ_POSITIONS,
                Category.READ_BALANCES,
                Category.READ_ORDERS,
            ):
                self._verified.mapping_for(
                    category,
                    require_proof=(
                        category is not Category.ORDER_PLACE or self._approval_mode == "standing"
                    ),
                )
        except CapabilityUnmapped as exc:
            return Rejected(code=RejectCode.CAPABILITY_UNMAPPED, reason=str(exc))
        if intent.side is Side.SELL:
            refusal = self._refuse_sell(intent)
            if refusal is not None:
                return refusal
        try:
            mapping, payload = self._call(
                Category.ORDER_PLACE,
                {
                    "account_id": self.account_id,
                    "symbol": intent.symbol,
                    "side": intent.side.value,
                    "qty": intent.qty,
                },
                require_proof=self._approval_mode == "standing",
            )
        except CapabilityUnmapped as exc:
            return Rejected(code=RejectCode.CAPABILITY_UNMAPPED, reason=str(exc))
        except ToolResultUnreadable as exc:
            return Rejected(
                code=RejectCode.BROKER_REFUSED,
                reason=f"{intent.describe()} was refused before placement: {exc}",
            )
        return self._fill_from(payload, mapping, intent)

    def cancel(self, order_id: str) -> CancelResult:
        if self._cancels >= self._max_cancels:
            return Rejected(
                code=RejectCode.CANCEL_LIMIT_REACHED,
                reason=(
                    f"this session has already cancelled {self._cancels} times, the "
                    "configured limit. Further cancels are refused; stop the agent or "
                    "inspect the order at the broker."
                ),
            )
        # The guard bounds attempts, not successful cancellations. A rejected
        # request still contributes to the activity pattern the broker sees.
        self._cancels += 1
        try:
            mapping, payload = self._call(
                Category.ORDER_CANCEL,
                {"account_id": self.account_id, "order_id": order_id},
                require_proof=True,
            )
        except CapabilityUnmapped as exc:
            return Rejected(code=RejectCode.CAPABILITY_UNMAPPED, reason=str(exc))
        except ToolResultUnreadable as exc:
            return Rejected(
                code=RejectCode.BROKER_REFUSED,
                reason=f"order {order_id} was not cancelled: {exc}",
            )
        accepted_path = mapping.result.get("accepted")
        if accepted_path is not None:
            accepted = dig(payload, accepted_path)
            if accepted is not True:
                return Rejected(
                    code=RejectCode.BROKER_REFUSED,
                    reason=(
                        f"the broker did not accept cancellation of {order_id}. Inspect the "
                        "order at the broker or try again after its state changes."
                    ),
                )
            # This is the box-observed acceptance time, not a timestamp invented as
            # broker evidence; final cancellation state remains an orders read.
            return Cancelled(order_id=order_id, ts=datetime.now(UTC))
        cancelled_path = mapping.result.get("cancelled_at")
        if cancelled_path is None:
            return Rejected(
                code=RejectCode.ORDER_OUTCOME_UNKNOWN,
                reason=(
                    f"the cancel of {order_id} has neither an accepted flag nor a broker "
                    "timestamp mapping. It may have taken effect; inspect it at the broker."
                ),
            )
        at = timestamp_at(payload, cancelled_path, f"cancel time for {order_id}")
        if isinstance(at, Unavailable):
            return Rejected(
                code=RejectCode.ORDER_OUTCOME_UNKNOWN,
                reason=(
                    f"the cancel of {order_id} returned no readable time ({at.reason}). "
                    "It may have taken effect; inspect it at the broker before acting."
                ),
            )
        return Cancelled(order_id=order_id, ts=at)

    def order_ids(self) -> tuple[str, ...]:
        mapping, payload = self._call(
            Category.READ_ORDERS,
            {"account_id": self.account_id},
            require_proof=False,
        )
        identifiers: list[str] = []
        for row in self._rows_for_account(payload, mapping, "orders"):
            identifier = text_at(row, mapping.result["order_id"], "order id")
            if isinstance(identifier, Unavailable):
                raise ToolResultUnreadable(
                    f"the broker listed an order with no readable id ({identifier.reason})"
                )
            identifiers.append(identifier)
        return tuple(identifiers)

    def orders(self) -> tuple[BrokerOrder, ...]:
        """Read account-scoped statuses with every fill number explicitly mapped."""
        mapping, payload = self._call(
            Category.READ_ORDERS,
            {"account_id": self.account_id},
            require_proof=False,
        )
        required = {"status", "symbol", "side", "quantity", "price", "filled_at"}
        absent = sorted(required - set(mapping.result))
        if absent:
            raise ToolResultUnreadable(
                f"the confirmed read.orders mapping has no paths for {absent}. Re-propose "
                "and confirm this tool before reconciling an order."
            )
        orders: list[BrokerOrder] = []
        for row in self._rows_for_account(payload, mapping, "orders"):
            identifier = text_at(row, mapping.result["order_id"], "order id")
            if isinstance(identifier, Unavailable):
                raise ToolResultUnreadable(
                    f"the broker listed an order with no readable id ({identifier.reason})"
                )
            raw_status = text_at(row, mapping.result["status"], f"status of {identifier}")
            if isinstance(raw_status, Unavailable):
                raise ToolResultUnreadable(str(raw_status))
            normalized = raw_status.lower().replace("-", "_")
            if normalized in {"filled", "executed", "complete", "completed"}:
                state = OrderState.FILLED
            elif normalized in {"cancelled", "canceled", "void", "rejected"}:
                state = OrderState.CANCELLED
            elif normalized in {"working", "open", "pending", "queued", "submitted"}:
                state = OrderState.WORKING
            else:
                raise ToolResultUnreadable(
                    f"order {identifier} has unrecognized status {raw_status!r}. Inspect "
                    "the broker and reconfirm the status mapping."
                )
            if state is not OrderState.FILLED:
                orders.append(
                    BrokerOrder(
                        order_id=identifier,
                        state=state,
                        symbol=None,
                        side=None,
                        qty=None,
                        price=None,
                        at=None,
                    )
                )
                continue
            symbol = text_at(row, mapping.result["symbol"], f"symbol of {identifier}")
            side_text = text_at(row, mapping.result["side"], f"side of {identifier}")
            qty = whole_at(row, mapping.result["quantity"], f"quantity of {identifier}")
            price = decimal_at(row, mapping.result["price"], f"price of {identifier}")
            at = timestamp_at(row, mapping.result["filled_at"], f"fill time of {identifier}")
            missing = [
                value
                for value in (symbol, side_text, qty, price, at)
                if isinstance(value, Unavailable)
            ]
            if missing:
                raise ToolResultUnreadable(
                    f"filled order {identifier} is unreadable "
                    f"({'; '.join(str(x) for x in missing)}). "
                    "Inspect it at the broker before continuing."
                )
            try:
                side = Side(str(side_text).lower())
            except ValueError as exc:
                raise ToolResultUnreadable(
                    f"filled order {identifier} has unrecognized side {side_text!r}. Inspect it "
                    "at the broker before continuing."
                ) from exc
            orders.append(
                BrokerOrder(
                    order_id=identifier,
                    state=state,
                    symbol=str(symbol),
                    side=side,
                    qty=qty,  # type: ignore[arg-type]
                    price=price,  # type: ignore[arg-type]
                    at=at,  # type: ignore[arg-type]
                )
            )
        return tuple(orders)

    def call_named(self, name: str, values: Mapping[str, Any]) -> Any:
        """Always refuse raw-name routing, including denied and newly added tools."""
        state = self._verified.states.get(name)
        label = state.value if state is not None else "unknown"
        raise CapabilityUnmapped(
            f"direct call of {name!r} is refused ({label}). Broker routing is a positive "
            "allowlist of confirmed categories; use a ProfileBroker capability instead."
        )

    @property
    def last_place_called(self) -> bool:
        """Whether the latest ``place`` crossed the exact verified tool boundary."""
        return self._last_place_called

    @property
    def verified_profile(self) -> VerifiedSessionProfile:
        return self._verified

    def _call(
        self,
        category: Category,
        values: Mapping[str, Any],
        *,
        require_proof: bool,
    ) -> tuple[ProfileTool, Any]:
        mapping = self._verified.mapping_for(category, require_proof=require_proof)
        if category.mutating:
            self._verified.refresh_tool(mapping.contract.name)
            if category is Category.ORDER_PLACE:
                for dependency in (
                    Category.ORDER_PLACE,
                    Category.READ_QUOTE,
                    Category.READ_POSITIONS,
                    Category.READ_BALANCES,
                    Category.READ_ORDERS,
                ):
                    self._verified.mapping_for(
                        dependency,
                        require_proof=(
                            dependency is not Category.ORDER_PLACE
                            or self._approval_mode == "standing"
                        ),
                    )
            if self._kill_switch():
                raise BrokerUnavailable(
                    f"the kill switch is set; {category.value} was refused after contract "
                    "verification and before tools/call. Remove the STOP file only if you "
                    "want this agent to run again."
                )
        arguments = mapping.render(values)
        try:
            Draft202012Validator.check_schema(mapping.contract.input_schema)
            Draft202012Validator(mapping.contract.input_schema).validate(arguments)
        except (SchemaError, ValidationError) as exc:
            raise ToolResultUnreadable(
                f"rendered arguments for {mapping.contract.name} do not satisfy the pinned "
                f"input schema ({exc.message}). Fix and reconfirm the mapping; no broker "
                "call was made."
            ) from exc
        if category is Category.ORDER_PLACE:
            self._last_place_called = True
        return mapping, self._verified.session.call_tool(mapping.contract.name, arguments)

    def _rows_for_account(
        self, payload: Any, mapping: ProfileTool, what: str
    ) -> list[Mapping[str, Any]]:
        rows = dig(payload, mapping.result["items"])
        if not isinstance(rows, Sequence) or isinstance(rows, str | bytes):
            raise ToolResultUnreadable(
                f"the broker's {mapping.contract.name} answer carries no list at "
                f"{mapping.result['items']!r}"
            )
        kept: list[Mapping[str, Any]] = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise ToolResultUnreadable(
                    f"the broker's {mapping.contract.name} answer has a "
                    f"{type(row).__name__} where Tick expected one {what} row"
                )
            if dig(row, mapping.result["account"]) == self.account_id:
                kept.append(row)
        return kept

    def _positions(self) -> dict[str, Position]:
        mapping, payload = self._call(
            Category.READ_POSITIONS,
            {"account_id": self.account_id},
            require_proof=False,
        )
        positions: dict[str, Position] = {}
        for row in self._rows_for_account(payload, mapping, "positions"):
            symbol = text_at(row, mapping.result["symbol"], "position symbol")
            if isinstance(symbol, Unavailable):
                raise ToolResultUnreadable(str(symbol))
            qty = whole_at(row, mapping.result["quantity"], f"quantity of {symbol}")
            if isinstance(qty, Unavailable):
                raise ToolResultUnreadable(str(qty))
            if qty == 0:
                continue
            if qty < 0:
                raise ToolResultUnreadable(
                    f"the broker reports {qty} {symbol}, a short position. Tick is "
                    "long-only and will not run against this account."
                )
            average = decimal_at(row, mapping.result["average_cost"], f"average cost of {symbol}")
            if isinstance(average, Unavailable):
                raise ToolResultUnreadable(str(average))
            positions[symbol] = Position(symbol=symbol, qty=qty, avg_cost=average)
        return positions

    def _cash(self) -> Decimal:
        mapping, payload = self._call(
            Category.READ_BALANCES,
            {"account_id": self.account_id},
            require_proof=False,
        )
        rows = self._rows_for_account(payload, mapping, "balances")
        if len(rows) != 1:
            raise ToolResultUnreadable(
                f"the broker returned {len(rows)} balance rows for account {self.account_id}. "
                "Tick needs exactly one; inspect the mapping and account before continuing."
            )
        cash = decimal_at(rows[0], mapping.result["cash"], f"cash in {self.account_id}")
        if isinstance(cash, Unavailable):
            raise ToolResultUnreadable(str(cash))
        if cash < 0:
            raise ToolResultUnreadable(
                f"the broker reports {cash} cash in {self.account_id}. Tick runs a cash "
                "account and will not place an order."
            )
        return cash

    def _refuse_sell(self, intent: OrderIntent) -> Rejected | None:
        try:
            held = self._positions().get(intent.symbol)
        except (CapabilityUnmapped, ToolResultUnreadable) as exc:
            return Rejected(
                code=RejectCode.ACCOUNT_UNAVAILABLE,
                reason=(
                    f"{intent.describe()} was not placed because positions could not be "
                    f"verified ({exc}). Restore and prove read.positions before selling."
                ),
            )
        if held is None:
            return Rejected(
                code=RejectCode.NO_POSITION_TO_SELL,
                reason=(
                    f"{intent.describe()} was not placed: account {self.account_id} holds "
                    f"no {intent.symbol}. Tick is long-only and never opens a short."
                ),
            )
        if intent.qty > held.qty:
            return Rejected(
                code=RejectCode.SELL_EXCEEDS_POSITION,
                reason=(
                    f"{intent.describe()} exceeds the {held.qty} {intent.symbol} held. "
                    "The order is refused whole; reduce it to what the account holds."
                ),
            )
        return None

    def _fill_from(self, payload: Any, mapping: ProfileTool, intent: OrderIntent) -> PlaceResult:
        order_id = text_at(payload, mapping.result["order_id"], "order id")
        qty = whole_at(payload, mapping.result["quantity"], "filled quantity")
        price = decimal_at(payload, mapping.result["price"], "fill price")
        at = timestamp_at(payload, mapping.result["filled_at"], "fill time")
        missing = [value for value in (order_id, qty, price, at) if isinstance(value, Unavailable)]
        if missing:
            if isinstance(order_id, str):
                return OrderOutcomeUnknown(
                    code=RejectCode.ORDER_OUTCOME_UNKNOWN,
                    broker_order_id=order_id,
                    reason=(
                        f"order {order_id} for {intent.describe()} is not a readable fill "
                        f"({'; '.join(item.reason for item in missing)}). It will be "
                        "reconciled through read.orders before the next tick; THE ORDER MAY "
                        "HAVE BEEN ACCEPTED, so inspect it at "
                        "the broker now if you need an immediate answer."
                    ),
                )
            return Rejected(
                code=RejectCode.ORDER_OUTCOME_UNKNOWN,
                reason=(
                    f"the fill for {intent.describe()} could not be read "
                    f"({'; '.join(item.reason for item in missing)}). THE ORDER MAY HAVE "
                    "BEEN ACCEPTED; inspect it at the broker before acting."
                ),
            )
        assert isinstance(order_id, str) and isinstance(qty, int) and isinstance(price, Decimal)
        if qty < 1 or price <= 0:
            return OrderOutcomeUnknown(
                code=RejectCode.ORDER_OUTCOME_UNKNOWN,
                broker_order_id=order_id,
                reason=(
                    f"the broker returned order {order_id} with {qty} shares at {price}, "
                    "which is not a fill. It will be reconciled before the next tick; "
                    "inspect it at the broker now if you need an immediate answer."
                ),
            )
        return Fill(
            order_id=order_id,
            symbol=intent.symbol,
            side=intent.side,
            qty=qty,
            price=price,
            ts=at,  # type: ignore[arg-type]
        )
