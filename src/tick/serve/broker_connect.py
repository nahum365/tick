"""Supervise one Robinhood OAuth ceremony for a paired phone or box browser."""

from __future__ import annotations

import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from tick.auth import (
    ROBINHOOD_MCP_URL,
    CallbackError,
    FileTokenStorage,
    LoopbackAuthorization,
    build_oauth_provider,
    disclosure_text,
)
from tick.broker import MCPSession, streamable_http_session
from tick.broker.profile import sanction_for

__all__ = ["BrokerConnectError", "BrokerConnectManager"]


class BrokerConnectError(Exception):
    def __init__(self, code: str, reason: str) -> None:
        self.code = code
        self.reason = reason
        super().__init__(f"{code}: {reason}")


@dataclass(slots=True)
class _Connect:
    loopback: LoopbackAuthorization
    state: str
    reason: str | None
    tools: int | None


class BrokerConnectManager:
    """Keep PKCE/state in memory until either completion path finishes."""

    def __init__(
        self,
        *,
        home,
        timeout_seconds: float,
        announce_wait_seconds: float,
        session_factory: Callable[
            [str, FileTokenStorage, LoopbackAuthorization, float], MCPSession
        ],
    ) -> None:
        self._home = home
        self._timeout = timeout_seconds
        self._announce_wait = announce_wait_seconds
        self._session_factory = session_factory
        self._connects: dict[str, _Connect] = {}

    @classmethod
    def for_environment(cls, *, home) -> BrokerConnectManager:
        def session_factory(server, storage, loopback, timeout):
            provider = build_oauth_provider(server_url=server, storage=storage, loopback=loopback)
            return MCPSession(streamable_http_session(server, provider), timeout_seconds=timeout)

        return cls(
            home=home,
            timeout_seconds=300.0,
            announce_wait_seconds=10.0,
            session_factory=session_factory,
        )

    def start(self, server_url: str | None, redirect_scheme: str | None) -> dict[str, object]:
        """`redirect_scheme` is the phone app's own URL scheme; the redirect registered
        with the broker becomes `<scheme>://broker/callback` so the in-app browser can
        intercept it and post it back. None keeps the loopback redirect."""
        server = server_url or ROBINHOOD_MCP_URL
        if sanction_for(server) != "official":
            raise BrokerConnectError(
                "BROKER_SERVER_UNSANCTIONED",
                "the app route connects only to the official broker host. Use the local CLI "
                "with --unsanctioned if you deliberately chose a community server.",
            )
        loopback = LoopbackAuthorization(
            port=0,
            timeout_seconds=self._timeout,
            open_browser=False,
            announce=lambda _line: None,
            redirect_uri_override=(
                f"{redirect_scheme}://broker/callback" if redirect_scheme else None
            ),
        )
        loopback.__enter__()
        connect_id = secrets.token_hex(10)
        item = _Connect(loopback=loopback, state="pending", reason=None, tools=None)
        self._connects[connect_id] = item
        session = self._session_factory(
            server, FileTokenStorage(self._home), loopback, self._timeout
        )

        def work() -> None:
            try:
                session.open()
                item.tools = len(session.list_tools())
                item.state = "succeeded"
            except Exception as exc:  # noqa: BLE001 - provider sentence becomes state
                item.reason = str(exc)
                item.state = "failed"
            finally:
                session.close()
                loopback.__exit__(None, None, None)

        threading.Thread(target=work, name=f"tick-connect-{connect_id}", daemon=True).start()
        deadline = time.monotonic() + self._announce_wait
        while (
            loopback.authorization_url is None
            and item.state == "pending"
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        if loopback.authorization_url is None:
            raise BrokerConnectError(
                "BROKER_AUTHORIZATION_URL_UNAVAILABLE",
                (item.reason or "the broker did not issue an authorization URL")
                + ". Start the connection again after checking the broker service.",
            )
        return {
            "authorization_url": loopback.authorization_url,
            "connect_id": connect_id,
            "redirect_uri": loopback.redirect_uri,
            "disclosure": disclosure_text(),
        }

    def complete(self, connect_id: str, redirect_url: str) -> dict[str, str]:
        item = self._connects.get(connect_id)
        if item is None:
            raise BrokerConnectError(
                "BROKER_CONNECT_NOT_FOUND",
                f"connection {connect_id} is not active. Start the broker connection again.",
            )
        try:
            item.loopback.complete_redirect_url(redirect_url)
        except CallbackError as exc:
            raise BrokerConnectError("BROKER_REDIRECT_REFUSED", str(exc)) from exc
        return {"connect_id": connect_id, "state": item.state}

    def status(self, connect_id: str) -> dict[str, object]:
        item = self._connects.get(connect_id)
        if item is None:
            raise BrokerConnectError(
                "BROKER_CONNECT_NOT_FOUND",
                f"connection {connect_id} is not active. Start the broker connection again.",
            )
        return {
            "connect_id": connect_id,
            "state": item.state,
            "reason": item.reason,
            "tools_discovered": item.tools,
        }
