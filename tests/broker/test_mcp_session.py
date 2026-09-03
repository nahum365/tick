"""The synchronous MCP seam: it discovers, it calls, and it stops on failure.

The failure tests are the point of the file. CLAUDE.md invariant 8 and the
2026-08-31 audit both say the runtime STOPS on a broker failure and never
blind-retries an order — so what is asserted here is not only that a dropped
session raises, but that it stays raised: the second call refuses with the
first call's reason and sends nothing.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from mcp.server.mcpserver import MCPServer

from tick.broker.errors import BrokerUnavailable, ToolResultUnreadable
from tick.broker.mcp_session import MCPSession
from tick.broker.mock_mcp import MockBrokerage

from .conftest import TIMEOUT, memory_opener


def session_over(server: MCPServer) -> MCPSession:
    return MCPSession(memory_opener(server), timeout_seconds=TIMEOUT)


def test_discovery_reads_the_tools_the_server_declares(mock_server: MCPServer):
    """Invariant 7: names and shapes come from `tools/list`, never from a guess."""
    with session_over(mock_server) as session:
        tools = session.list_tools()

    names = sorted(tool.name for tool in tools)
    assert names == [
        "cancel_order",
        "get_accounts",
        "get_positions",
        "get_quote",
        "list_orders",
        "place_order",
    ]
    schemas = {tool.name: tool.input_schema for tool in tools}
    assert "symbol" in schemas["get_quote"]["properties"]


def test_a_tools_list_cursor_loop_invalidates_the_whole_session():
    class Listing:
        async def list_tools(self, *, params):
            return SimpleNamespace(tools=[], next_cursor="same-cursor")

    session = MCPSession(memory_opener(MCPServer(name="unused")), timeout_seconds=TIMEOUT)
    session._submit = lambda operation: asyncio.run(operation(Listing()))
    with pytest.raises(BrokerUnavailable, match="cursor loop.*whole session"):
        session.list_tools()


def test_duplicate_names_across_inventory_pages_invalidate_the_session():
    item = SimpleNamespace(
        name="get_quote",
        title=None,
        description="Price in dollars.",
        input_schema={"type": "object", "properties": {}},
        output_schema=None,
        annotations=None,
        execution=None,
    )

    class Listing:
        calls = 0

        async def list_tools(self, *, params):
            self.calls += 1
            cursor = "second-page" if self.calls == 1 else None
            return SimpleNamespace(tools=[item], next_cursor=cursor)

    listing = Listing()
    session = MCPSession(memory_opener(MCPServer(name="unused")), timeout_seconds=TIMEOUT)
    session._submit = lambda operation: asyncio.run(operation(listing))
    with pytest.raises(BrokerUnavailable, match="duplicate tool names.*whole session"):
        session.list_tools()


def test_a_call_returns_the_structured_payload(mock_server: MCPServer):
    with session_over(mock_server) as session:
        payload = session.call_tool("get_quote", {"symbol": "XYZ"})

    assert payload["last_price"] == "184.20"


def test_the_server_names_itself_at_the_handshake(mock_server: MCPServer):
    with session_over(mock_server) as session:
        assert session.server_name == "mock-brokerage"


def test_a_tool_error_is_a_readable_refusal_not_a_number(mock_server: MCPServer):
    """A tool that failed produced no data; it must not read as one that produced zero."""
    with session_over(mock_server) as session:
        with pytest.raises(ToolResultUnreadable) as caught:
            session.call_tool("cancel_order", {"order_id": "no-such-order"})

    assert "cancel_order" in str(caught.value)


def test_a_session_that_never_opened_refuses_every_call():
    """Fail safe: no session, no call — and the message says what to run."""
    session = MCPSession(memory_opener(MCPServer(name="unused")), timeout_seconds=TIMEOUT)

    with pytest.raises(BrokerUnavailable) as caught:
        session.call_tool("get_quote", {"symbol": "XYZ"})

    assert "tick connect robinhood" in str(caught.value)


def test_an_opener_that_fails_reports_why_and_places_nothing():
    @asynccontextmanager
    async def broken() -> AsyncIterator[None]:
        raise ConnectionRefusedError("the broker refused the connection")
        yield  # pragma: no cover - unreachable, keeps this an async generator

    session = MCPSession(broken, timeout_seconds=TIMEOUT)

    with pytest.raises(BrokerUnavailable) as caught:
        session.open()

    assert "refused the connection" in str(caught.value)
    assert "nothing was retried" in str(caught.value).lower()


def test_a_failed_session_stays_failed_and_never_reconnects(mock_server: MCPServer):
    """One failure, one stop. The adapter above it must not get a second chance."""
    session = session_over(mock_server)
    session.open()
    session.close()
    # Simulate the transport dropping under us rather than a clean close.
    session._failure = "the broker's MCP session failed: dropped"

    for _ in range(2):
        with pytest.raises(BrokerUnavailable) as caught:
            session.call_tool("get_quote", {"symbol": "XYZ"})
        assert "dropped" in str(caught.value)


def test_a_call_that_hangs_times_out_rather_than_waiting_on_the_market():
    """A broker that never answers is a stop, not a pause before an order."""
    server = MCPServer(name="slow-brokerage")

    @server.tool(description="Never answers.", structured_output=True)
    async def get_quote(symbol: str) -> dict[str, str]:
        await asyncio.sleep(30)
        return {"symbol": symbol}  # pragma: no cover - the timeout fires first

    session = MCPSession(memory_opener(server), timeout_seconds=0.5)
    session.open()

    with pytest.raises(BrokerUnavailable) as caught:
        session.call_tool("get_quote", {"symbol": "XYZ"})

    assert "must not be sent twice" in str(caught.value)
    assert session.failure is not None


def test_opening_twice_is_refused(mock_server: MCPServer):
    session = session_over(mock_server)
    session.open()
    try:
        with pytest.raises(BrokerUnavailable):
            session.open()
    finally:
        session.close()


def test_a_non_positive_timeout_is_refused(mock_server: MCPServer):
    with pytest.raises(ValueError):
        MCPSession(memory_opener(mock_server), timeout_seconds=0)


def test_the_mock_over_answers_the_way_the_real_grant_does(
    mock_server: MCPServer, brokerage: MockBrokerage
):
    """The mock returns every account, which is what makes the filter testable."""
    with session_over(mock_server) as session:
        payload = session.call_tool("get_positions", {"account_id": brokerage.agentic_account})

    accounts = {row["account"] for row in payload["positions"]}
    assert accounts == {brokerage.agentic_account, brokerage.other_account}
