"""Fixtures for the broker tests — a brokerage in memory, and no socket anywhere.

The Robinhood adapter is exercised against `tick.broker.mock_mcp` over the MCP
SDK's in-memory transport: a real `MCPServer`, a real `ClientSession`, a real
`tools/list` handshake, and no network stack between them. Nothing in this
package resolves a name, opens a port, or holds a token.

`TICK_HOME` is redirected for the whole package, autouse: the tool map is a
file on disk, and a test that wrote one into the developer's own `~/.tick`
would change what their runtime does the next time they ran it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from mcp import ClientSession
from mcp.client._memory import InMemoryTransport
from mcp.server.mcpserver import MCPServer

from tick.auth import loopback as loopback_module
from tick.broker.mcp_session import SessionOpener
from tick.broker.mock_mcp import MockBrokerage, build_mock_server, default_brokerage
from tick.records import TICK_HOME_ENV

#: How long the tests let an in-memory call take before calling it a failure.
TIMEOUT = 10.0


class _NoSocketCallbackServer:
    """Enough listener state for CLI ceremony tests, without binding a port."""

    captured = None

    def __init__(self, address, handler) -> None:
        host, port = address
        self.server_address = (host, port or 48123)
        self.timeout = None

    def server_close(self) -> None:
        return None


@pytest.fixture(autouse=True)
def no_real_tick_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    home = tmp_path / "tick-home"
    monkeypatch.setenv(TICK_HOME_ENV, str(home))
    monkeypatch.setattr(loopback_module, "_CallbackServer", _NoSocketCallbackServer)
    yield home


@pytest.fixture
def home(no_real_tick_home: Path) -> Path:
    return no_real_tick_home


@pytest.fixture
def brokerage() -> MockBrokerage:
    return default_brokerage()


@pytest.fixture
def mock_server(brokerage: MockBrokerage) -> MCPServer:
    return build_mock_server(brokerage)


def memory_opener(server: MCPServer) -> SessionOpener:
    """An opener that speaks to `server` in this process, over memory streams."""

    @asynccontextmanager
    async def opener() -> AsyncIterator[ClientSession]:
        async with InMemoryTransport(server) as (read, write):
            async with ClientSession(read, write) as session:
                yield session

    return opener


@pytest.fixture
def opener(mock_server: MCPServer) -> SessionOpener:
    return memory_opener(mock_server)
