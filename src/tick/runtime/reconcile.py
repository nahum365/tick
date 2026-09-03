"""Resolve placements whose broker outcome was not terminal when returned."""

from __future__ import annotations

from datetime import datetime

from tick.broker import BrokerPort, Fill, OrderState, RejectCode
from tick.records import DataSource, RecordKind, read

from .errors import RuntimeStateError
from .state import AgentRun

__all__ = ["reconcile_unknown_orders", "unknown_order_ids"]


def unknown_order_ids(agent: AgentRun) -> tuple[str, ...]:
    """Return unresolved broker ids in ledger order without asking the broker."""
    unresolved: dict[str, None] = {}
    for record in read(agent.ledger_path):
        payload = record.payload
        if record.kind is RecordKind.REJECTED:
            rejected = payload.get("rejected")
            if (
                isinstance(rejected, dict)
                and rejected.get("code") == RejectCode.ORDER_OUTCOME_UNKNOWN.value
                and isinstance(rejected.get("broker_order_id"), str)
            ):
                unresolved[rejected["broker_order_id"]] = None
        if record.kind is RecordKind.FILL:
            fill = payload.get("fill")
            if isinstance(fill, dict):
                unresolved.pop(str(fill.get("order_id")), None)
        if record.kind is RecordKind.NOTE and payload.get("event") == "order_cancelled":
            unresolved.pop(str(payload.get("order_id")), None)
    return tuple(unresolved)


def reconcile_unknown_orders(agent: AgentRun, broker: BrokerPort, *, now: datetime) -> int:
    """Append terminal/status evidence, or halt when a broker id is unmatched."""
    wanted = unknown_order_ids(agent)
    if not wanted:
        return 0
    try:
        observed = {order.order_id: order for order in broker.orders()}
    except Exception as exc:  # noqa: BLE001 - an unreadable list cannot authorize another tick
        raise RuntimeStateError(
            f"unknown broker orders {', '.join(wanted)} could not be reconciled "
            f"({type(exc).__name__}: {exc}). No next tick may run; inspect those order ids "
            "at the broker and restore read.orders."
        ) from exc
    missing = [order_id for order_id in wanted if order_id not in observed]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeStateError(
            f"unknown broker order {joined} was not returned by read.orders for this account. "
            "No next tick may run; check that order id at the broker and confirm the account "
            "and read.orders mapping."
        )
    ledger = agent.ledger(clock=lambda: now)
    appended = 0
    for order_id in wanted:
        order = observed[order_id]
        if order.state is OrderState.FILLED:
            assert None not in {order.symbol, order.side, order.qty, order.price, order.at}
            fill = Fill(
                order_id=order.order_id,
                symbol=order.symbol,  # type: ignore[arg-type]
                side=order.side,  # type: ignore[arg-type]
                qty=order.qty,  # type: ignore[arg-type]
                price=order.price,  # type: ignore[arg-type]
                ts=order.at,  # type: ignore[arg-type]
            )
            ledger.append(
                RecordKind.FILL,
                {"fill": fill, "reconciled": True, "at": now},
                source=DataSource.ROBINHOOD,
            )
        elif order.state is OrderState.CANCELLED:
            ledger.append(
                RecordKind.NOTE,
                {"event": "order_cancelled", "order_id": order_id, "at": now},
                source=DataSource.ROBINHOOD,
            )
        else:
            ledger.append(
                RecordKind.NOTE,
                {
                    "event": "order_still_working",
                    "order_id": order_id,
                    "reason": (
                        f"order {order_id} is still working. Inspect or cancel it at the "
                        "broker; Tick will reconcile it again before another tick."
                    ),
                    "at": now,
                },
                source=DataSource.ROBINHOOD,
            )
        appended += 1
    return appended
