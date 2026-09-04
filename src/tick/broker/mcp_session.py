"""One MCP session, driven from synchronous code, and stopped on first failure.

The runtime is synchronous — a scheduler, a tick, a broker call — and MCP is
asynchronous. This module is the seam: a session is opened once on a private
event loop in a private thread, and `list_tools` / `call_tool` hand work to
that loop and wait. One long-lived session rather than one per call is the
cadence rule from `engine/cadence.py` applied to the connection itself:
Robinhood may terminate MCP connectivity for usage it judges excessive, and a
handshake per quote is exactly that.

**Failure stops the session; it never retries.** The 2026-08-31 audit and
CLAUDE.md invariant 8 both land here. When the transport drops, times out, or
raises, the session is marked failed, closed, and every later call refuses
immediately with the original reason. A broker adapter that reconnected and
re-sent an order whose outcome it never saw is how one intent becomes two
fills — so nothing in this file retries anything, and the runtime above stops
and notifies instead.

The two openers are separated for the same reason `MarketDataPort` is a
protocol: `streamable_http_session` is the real transport and names the only
remote host in the product, and a test supplies an in-memory opener over the
mock server in `mock_mcp.py`. Nothing in the test path touches a socket.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from types import TracebackType
from typing import Any

import httpx
from mcp import ClientSession
from mcp.client.auth import OAuthClientProvider
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client
from mcp.types import (
    CallToolResult,
    PaginatedRequestParams,
    TextContent,
    ToolListChangedNotification,
)

from .errors import BrokerUnavailable, ToolResultUnreadable
from .toolmap import DiscoveredTool

__all__ = [
    "MCPSession",
    "SessionOpener",
    "host_guard",
    "payload_of",
    "same_site",
    "streamable_http_session",
]

#: A factory for one MCP session. Called at most once per `MCPSession`.
SessionOpener = Callable[[], AbstractAsyncContextManager[ClientSession]]


def same_site(host: str | None, pinned_host: str) -> bool:
    """True when `host` is the pinned host or another host of its registrable domain.

    The MCP endpoint (agent.robinhood.com) and the OAuth authorization server
    that issues its tokens (api.robinhood.com) are different hosts of one
    operator. Credentials may travel to either; a host outside that domain is
    refused whether it arrives by redirect or by a direct request.
    """
    if not host:
        return False
    if host == pinned_host:
        return True
    labels = pinned_host.split(".")
    if len(labels) < 2:
        return False
    return host.endswith("." + ".".join(labels[-2:]))


def host_guard(pinned_host: str) -> Callable[[httpx.Response], Awaitable[None]]:
    """An httpx response hook that refuses hosts outside the pinned site.

    Every response the authorised client receives passes through here: the MCP
    handshake, token requests, and any redirect target named in Location.
    """

    async def enforce(response: httpx.Response) -> None:
        observed = response.url.host
        location = response.headers.get("location")
        redirected = response.url.join(location).host if location is not None else None
        if not same_site(observed, pinned_host):
            raise BrokerUnavailable(
                f"the broker session reached {observed}, outside pinned site {pinned_host}. "
                "The session is refused before anything is sent there; connect to the "
                "intended host explicitly."
            )
        if redirected is not None and not same_site(redirected, pinned_host):
            raise BrokerUnavailable(
                f"the broker redirected from {observed} to {redirected}, outside pinned site "
                f"{pinned_host}. The session is refused before following that redirect; "
                "connect to the intended host explicitly."
            )

    return enforce


def streamable_http_session(url: str, auth: OAuthClientProvider) -> SessionOpener:
    """An opener for a streamable-HTTP MCP server, authorised by `auth`.

    `auth` is the `OAuthClientProvider` built in `tick.auth`. It is a required
    argument: an opener that could be built without one would connect
    unauthenticated and fail deep inside the handshake, and the token is the
    whole of invariant 1 — it belongs where a reader can see it.
    """

    @asynccontextmanager
    async def opener() -> AsyncIterator[ClientSession]:
        async with create_mcp_http_client(auth=auth) as client:
            client.event_hooks["response"].append(host_guard(httpx.URL(url).host))
            async with streamable_http_client(url, http_client=client) as (read, write):
                async with ClientSession(read, write) as session:
                    yield session

    return opener


def payload_of(result: CallToolResult, tool: str) -> Any:
    """The data a tool returned, or a refusal — never a half-understood shape.

    Structured output is preferred where the server declares an output schema.
    Where it does not, the convention is one text block of JSON, so that is
    parsed. Anything else refuses: invariant 7 says an unrecognised shape is
    not guessed at, and a tool result nobody can read is not a number.
    """
    if result.is_error:
        raise ToolResultUnreadable(
            f"the broker's {tool} tool answered with an error: {_first_text(result)}"
        )
    if isinstance(result.structured_content, dict):
        return result.structured_content
    text = _first_text(result)
    if text is None:
        raise ToolResultUnreadable(
            f"the broker's {tool} tool returned no structured output and no text. "
            f"Tick maps discovered tools by their declared shapes and does not guess "
            f"at an empty one."
        )
    try:
        return json.loads(text, parse_float=str)
    except json.JSONDecodeError as exc:
        raise ToolResultUnreadable(
            f"the broker's {tool} tool returned text that is not JSON ({exc}). Run "
            f"`tick broker tools` to see what it actually declares, and map the "
            f"capability to a tool whose result Tick can read."
        ) from exc


def _first_text(result: CallToolResult) -> str | None:
    for block in result.content:
        if isinstance(block, TextContent):
            return block.text
    return None


def describe_failure(exc: BaseException) -> str:
    """Name the leaf exceptions, not the group that carried them.

    anyio task groups raise ExceptionGroup("unhandled errors in a TaskGroup"),
    which says nothing about the broker; the person needs the leaf sentence
    (an HTTP status from the token endpoint, a refused redirect, a timeout).
    """
    if isinstance(exc, BaseExceptionGroup):
        leaves = [describe_failure(sub) for sub in exc.exceptions]
        return "; ".join(leaves) if leaves else f"{type(exc).__name__}: {exc}"
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


class MCPSession:
    """A synchronous handle on one MCP session, closed for good on first failure.

    `timeout_seconds` is required. There is no default: how long the runtime
    waits on a brokerage before declaring it unavailable is a decision about
    the user's money, and a limit nobody chose is not a limit.
    """

    def __init__(self, opener: SessionOpener, *, timeout_seconds: float) -> None:
        if timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds ({timeout_seconds}) must be > 0")
        self._opener = opener
        self._timeout = timeout_seconds
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._session: ClientSession | None = None
        self._stop: asyncio.Event | None = None
        self._ready = threading.Event()
        self._failure: str | None = None
        self._server_name: str | None = None
        self._tools_changed_callbacks: list[Callable[[str], None]] = []

    # ------------------------------------------------------------------
    # Lifetime
    # ------------------------------------------------------------------

    def __enter__(self) -> MCPSession:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def server_name(self) -> str | None:
        """What the server called itself at initialize, once connected."""
        return self._server_name

    @property
    def failure(self) -> str | None:
        """Why this session stopped, or `None` while it is healthy."""
        return self._failure

    def open(self) -> None:
        """Start the session thread and complete the MCP handshake."""
        if self._thread is not None:
            raise BrokerUnavailable("this session has already been opened; build a new one.")
        self._thread = threading.Thread(target=self._run, name="tick-mcp-session", daemon=True)
        self._thread.start()
        if not self._ready.wait(self._timeout):
            self._fail(
                f"the broker did not complete the MCP handshake within "
                f"{self._timeout:.0f}s. Nothing was placed."
            )
        if self._failure is not None:
            raise BrokerUnavailable(self._failure)

    def close(self) -> None:
        """Close the session and stop its thread. Safe to call more than once."""
        loop, stop = self._loop, self._stop
        if loop is not None and stop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(stop.set)
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=self._timeout)
        self._session = None

    # ------------------------------------------------------------------
    # The two calls the adapter makes
    # ------------------------------------------------------------------

    def list_tools(self) -> tuple[DiscoveredTool, ...]:
        """Every tool the server declares — invariant 7's `tools/list`.

        Translated into Tick's own `DiscoveredTool` on the way out, so this file
        stays the only one in the adapter that imports the MCP SDK.
        """
        discovered: list[DiscoveredTool] = []
        cursor: str | None = None
        seen_cursors: set[str] = set()
        while True:
            params = PaginatedRequestParams(cursor=cursor) if cursor is not None else None
            result = self._submit(lambda session, params=params: session.list_tools(params=params))
            discovered.extend(
                DiscoveredTool(
                    name=tool.name,
                    title=tool.title,
                    description=tool.description,
                    input_schema=tool.input_schema or {},
                    output_schema=tool.output_schema,
                    annotations=(
                        tool.annotations.model_dump(mode="json", by_alias=True, exclude_none=True)
                        if tool.annotations is not None
                        else None
                    ),
                    execution=(
                        tool.execution.model_dump(mode="json", by_alias=True, exclude_none=True)
                        if tool.execution is not None
                        else None
                    ),
                )
                for tool in result.tools
            )
            next_cursor = result.next_cursor
            if next_cursor is None:
                break
            if not next_cursor or next_cursor in seen_cursors:
                self._fail(
                    "the broker returned an incomplete tools/list pagination cursor loop. "
                    "The whole session is invalid and no tool can be called."
                )
            seen_cursors.add(next_cursor)
            cursor = next_cursor
        names = [tool.name for tool in discovered]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            self._fail(
                f"the broker advertised duplicate tool names {duplicates}. The whole "
                "session is invalid and no tool can be called."
            )
        return tuple(discovered)

    def on_tools_changed(self, callback: Callable[[str], None]) -> None:
        """Revoke consumers when ``notifications/tools/list_changed`` arrives.

        The real MCP notification hook and the in-memory tests both enter
        through ``notify_tools_changed``; keeping registration on the session
        means no adapter can forget which binding owns the callback.
        """
        self._tools_changed_callbacks.append(callback)

    def notify_tools_changed(self) -> None:
        """Deliver the MCP tools-list change signal to every bound profile."""
        reason = "the broker announced notifications/tools/list_changed"
        for callback in tuple(self._tools_changed_callbacks):
            callback(reason)

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Any:
        """Call one discovered tool and return the data it answered with."""
        result = self._submit(lambda session: session.call_tool(name, dict(arguments)))
        if not isinstance(result, CallToolResult):
            raise ToolResultUnreadable(
                f"the broker's {name} tool asked for something Tick does not answer "
                f"({type(result).__name__}). Nothing was retried and nothing was placed."
            )
        return payload_of(result, name)

    # ------------------------------------------------------------------
    # The thread, the loop, and the one-way trip to failed
    # ------------------------------------------------------------------

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        try:
            loop.run_until_complete(self._lifetime())
        finally:
            self._session = None
            self._ready.set()
            loop.close()

    async def _lifetime(self) -> None:
        self._stop = asyncio.Event()
        try:
            async with self._opener() as session:
                original_handler = session._message_handler

                async def message_handler(message: Any) -> None:
                    if isinstance(message, ToolListChangedNotification):
                        self.notify_tools_changed()
                    await original_handler(message)

                session._message_handler = message_handler
                initialized = await session.initialize()
                self._server_name = initialized.server_info.name
                self._session = session
                self._ready.set()
                await self._stop.wait()
        except BaseException as exc:  # noqa: BLE001 - the reason is reported, not swallowed
            self._failure = self._failure or (
                f"the broker's MCP session failed: {describe_failure(exc)}. "
                f"Tick stops rather than reconnecting; nothing was retried."
            )

    def _submit(self, work: Callable[[ClientSession], Any]) -> Any:
        if self._failure is not None:
            raise BrokerUnavailable(self._failure)
        session, loop = self._session, self._loop
        if session is None or loop is None or loop.is_closed():
            raise BrokerUnavailable(
                "the broker's MCP session is not open. Run `tick connect robinhood` "
                "and start the runtime again; nothing was placed."
            )
        future = asyncio.run_coroutine_threadsafe(work(session), loop)
        try:
            return future.result(timeout=self._timeout)
        except TimeoutError:
            future.cancel()
            self._fail(
                f"the broker did not answer within {self._timeout:.0f}s. Tick stops "
                f"rather than asking again: an order whose outcome is unknown must "
                f"not be sent twice."
            )
        except BaseException as exc:  # noqa: BLE001 - re-raised as a broker failure
            self._fail(
                f"the broker's MCP session failed mid-call: {describe_failure(exc)}. "
                f"Nothing was retried."
            )
        raise AssertionError("unreachable: _fail always raises")  # pragma: no cover

    def _fail(self, reason: str) -> None:
        """Record why the session stopped, close it, and refuse from here on."""
        self._failure = self._failure or reason
        self.close()
        raise BrokerUnavailable(self._failure)
