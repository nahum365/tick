"""A stand-in for a brokerage MCP, so the adapter can be exercised offline.

Robinhood's real tool names and schemas are **unverified** (CLAUDE.md
invariant 7). This server is therefore not a model of theirs and must never be
read as one: it is a *plausible* brokerage that answers plausible tools, built
so the discovery, mapping, scoping and refusal machinery can be tested without
a socket, a token or an account. Every symbol in it is a placeholder — Tick
authors no strategies and names no securities anywhere.

Two of its behaviours are deliberately awkward, because the adapter has to
survive them:

- **It over-answers.** `get_positions` and `list_orders` return every account
  the grant can see, *whatever* account id they were asked for — which is what
  Robinhood's own disclosure describes, since the grant reads all of a user's
  accounts. That is what makes the adapter's read-scoping filter testable: a
  mock that helpfully returned only the requested account would prove nothing.
- **It confines trading.** `place_order` refuses an account that is not the
  Agentic one, the way the real broker does, so the adapter's own refusal is
  not the only thing standing between a spec and someone's IRA. Its refusals
  are `ToolError`s, which the SDK serves to the caller with their words intact;
  an ordinary exception would reach the adapter as "Error executing tool X",
  and a refusal nobody can read is not a refusal.

Money crosses the wire as decimal *strings*. That is the honest encoding for
money in JSON and the one the adapter accepts: a JSON float is a binary
approximation, and `spec/base.py` has refused those since slice 01.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

__all__ = [
    "MockBrokerage",
    "MockHolding",
    "build_mock_server",
]


@dataclass
class MockHolding:
    """A position in one of the mock's accounts."""

    account: str
    symbol: str
    quantity: int
    average_cost: Decimal


@dataclass
class MockBrokerage:
    """The mock's state, and the log of what was asked of it.

    `calls` is the point of the class as much as the state is: the read-scoping
    tests assert on which account ids ever appeared in an argument, and that is
    a claim about the requests, not about the answers.
    """

    agentic_account: str
    other_account: str
    cash: dict[str, Decimal]
    holdings: list[MockHolding]
    quotes: dict[str, Decimal]
    quoted_at: datetime = datetime(2026, 9, 1, 15, 30, tzinfo=UTC)
    orders: list[dict[str, Any]] = field(default_factory=list)
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    next_order: int = 1

    def record(self, tool: str, arguments: dict[str, Any]) -> None:
        self.calls.append((tool, arguments))

    def argument_text(self) -> str:
        """Every argument value ever sent, as one string a test can search."""
        return " ".join(
            f"{key}={value}" for _, arguments in self.calls for key, value in arguments.items()
        )

    def held(self, account: str, symbol: str) -> int:
        return sum(h.quantity for h in self.holdings if h.account == account and h.symbol == symbol)

    def order_id(self) -> str:
        identifier = f"mock-order-{self.next_order:04d}"
        self.next_order += 1
        return identifier


def default_brokerage() -> MockBrokerage:
    """A two-account brokerage on placeholder symbols. Test material only."""
    return MockBrokerage(
        agentic_account="agentic-0001",
        other_account="brokerage-0002",
        cash={"agentic-0001": Decimal("5000.00"), "brokerage-0002": Decimal("91000.00")},
        holdings=[
            MockHolding("agentic-0001", "XYZ", 12, Decimal("170.00")),
            MockHolding("agentic-0001", "ABCD", 4, Decimal("52.25")),
            MockHolding("brokerage-0002", "WXY", 800, Decimal("31.10")),
        ],
        quotes={"XYZ": Decimal("184.20"), "ABCD": Decimal("55.00"), "WXY": Decimal("29.90")},
    )


def build_mock_server(state: MockBrokerage) -> MCPServer:
    """An `MCPServer` exposing plausible brokerage tools over `state`."""
    server = MCPServer(name="mock-brokerage", version="0.0.1")

    @server.tool(description="Last trade price for one symbol.", structured_output=True)
    def get_quote(symbol: str) -> dict[str, Any]:
        state.record("get_quote", {"symbol": symbol})
        price = state.quotes.get(symbol)
        if price is None:
            return {"symbol": symbol, "last_price": None, "quoted_at": None}
        return {
            "symbol": symbol,
            "last_price": str(price),
            "quoted_at": state.quoted_at.isoformat(),
        }

    @server.tool(
        description="Positions. The grant reads every account, so every account is returned.",
        structured_output=True,
    )
    def get_positions(account_id: str) -> dict[str, Any]:
        state.record("get_positions", {"account_id": account_id})
        return {
            "positions": [
                {
                    "account": holding.account,
                    "symbol": holding.symbol,
                    "quantity": str(holding.quantity),
                    "average_cost": str(holding.average_cost),
                }
                for holding in state.holdings
            ]
        }

    @server.tool(description="Accounts and their cash balances.", structured_output=True)
    def get_accounts() -> dict[str, Any]:
        state.record("get_accounts", {})
        return {
            "accounts": [
                {
                    "account_id": account,
                    "kind": "agentic" if account == state.agentic_account else "brokerage",
                    "cash": str(balance),
                }
                for account, balance in state.cash.items()
            ]
        }

    @server.tool(description="Place a market order in the Agentic account.", structured_output=True)
    def place_order(account_id: str, symbol: str, side: str, quantity: str) -> dict[str, Any]:
        state.record(
            "place_order",
            {"account_id": account_id, "symbol": symbol, "side": side, "quantity": quantity},
        )
        if account_id != state.agentic_account:
            raise ToolError(f"account {account_id} is not agentic; orders are confined to it")
        price = state.quotes.get(symbol)
        if price is None:
            raise ToolError(f"no price for {symbol}")
        shares = int(quantity)
        if side == "sell" and shares > state.held(account_id, symbol):
            raise ToolError(f"sell of {shares} {symbol} exceeds the position")
        _apply(state, account_id, symbol, side, shares, price)
        order = {
            "order_id": state.order_id(),
            "status": "filled",
            "symbol": symbol,
            "side": side,
            "account": account_id,
            "filled_quantity": str(shares),
            "filled_price": str(price),
            "filled_at": state.quoted_at.isoformat(),
        }
        state.orders.append(order)
        return order

    @server.tool(description="Cancel a working order.", structured_output=True)
    def cancel_order(order_id: str) -> dict[str, Any]:
        state.record("cancel_order", {"order_id": order_id})
        for order in state.orders:
            if order["order_id"] == order_id:
                order["status"] = "cancelled"
                return {
                    "order_id": order_id,
                    "status": "cancelled",
                    "cancelled_at": state.quoted_at.isoformat(),
                }
        raise ToolError(f"no order {order_id}")

    @server.tool(
        description="Orders. Like positions, the grant sees every account.",
        structured_output=True,
    )
    def list_orders(account_id: str) -> dict[str, Any]:
        state.record("list_orders", {"account_id": account_id})
        seen = [dict(order) for order in state.orders]
        seen.append(
            {
                "order_id": "mock-order-other",
                "status": "filled",
                "symbol": "WXY",
                "side": "buy",
                "account": state.other_account,
                "filled_quantity": "800",
                "filled_price": "31.10",
                "filled_at": state.quoted_at.isoformat(),
            }
        )
        return {"orders": seen}

    return server


def _apply(
    state: MockBrokerage,
    account: str,
    symbol: str,
    side: str,
    shares: int,
    price: Decimal,
) -> None:
    """Move the mock's cash and holdings the way a fill would."""
    signed = shares if side == "buy" else -shares
    state.cash[account] -= Decimal(signed) * price
    for holding in state.holdings:
        if holding.account == account and holding.symbol == symbol:
            holding.quantity += signed
            if holding.quantity == 0:
                state.holdings.remove(holding)
            return
    if signed > 0:
        state.holdings.append(MockHolding(account, symbol, signed, price))
