"""Unknown broker outcomes are settled before another live tick can proceed."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from tick.broker import BrokerOrder, OrderState
from tick.records import DataSource, RecordKind, read
from tick.runtime import RuntimeStateError, reconcile_unknown_orders, unknown_order_ids

AT = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)


class Orders:
    def __init__(self, values: list[BrokerOrder]) -> None:
        self.values = values

    def orders(self) -> list[BrokerOrder]:
        return self.values


def unknown(agent, order_id: str) -> None:
    agent.ledger(clock=lambda: AT).append(
        RecordKind.REJECTED,
        {
            "rejected": {
                "code": "order_outcome_unknown",
                "broker_order_id": order_id,
                "reason": "the order may be working; check it at the broker.",
            }
        },
        source=DataSource.ROBINHOOD,
    )


def test_reconciliation_records_a_fill_and_resolves_the_unknown(agent):
    unknown(agent, "order-7")
    broker = Orders(
        [
            BrokerOrder(
                order_id="order-7",
                state=OrderState.FILLED,
                symbol="XYZ",
                side="buy",
                qty=2,
                price=Decimal("17.25"),
                at=AT,
            )
        ]
    )

    assert reconcile_unknown_orders(agent, broker, now=AT) == 1
    assert unknown_order_ids(agent) == ()
    result = list(read(agent.ledger_path))[-1]
    assert result.kind is RecordKind.FILL
    assert result.payload["reconciled"] is True


def test_reconciliation_records_cancelled_and_working_states(agent):
    unknown(agent, "order-8")
    unknown(agent, "order-9")
    broker = Orders(
        [
            BrokerOrder(
                order_id="order-8",
                state=OrderState.CANCELLED,
                symbol=None,
                side=None,
                qty=None,
                price=None,
                at=None,
            ),
            BrokerOrder(
                order_id="order-9",
                state=OrderState.WORKING,
                symbol=None,
                side=None,
                qty=None,
                price=None,
                at=None,
            ),
        ]
    )

    reconcile_unknown_orders(agent, broker, now=AT)
    notes = [row.payload for row in read(agent.ledger_path) if row.kind is RecordKind.NOTE]
    assert [note["event"] for note in notes] == ["order_cancelled", "order_still_working"]
    assert "Inspect or cancel" in notes[-1]["reason"]
    assert unknown_order_ids(agent) == ("order-9",)


def test_an_unmatched_unknown_halts_with_the_order_id_and_next_action(agent):
    unknown(agent, "order-missing")

    with pytest.raises(RuntimeStateError, match="order-missing.*check that order id"):
        reconcile_unknown_orders(agent, Orders([]), now=AT)
