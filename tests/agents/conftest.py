"""Fixtures for the model-agent tests: a fake provider, and no network at all.

`no_network` is autouse for the whole package — `socket.socket` raises — so a
test that reached for a provider fails loudly instead of quietly billing
somebody's account. A model agent has a live path in the product and none here.

`FakeModelClient` implements `ModelClient` and answers with replies built in
the SDK's shape, read through the ADAPTER's own `read_model_reply`. A fake with
its own parser would be testing itself; this one exercises the real one on
every call, and keeps the request it was handed so a test can read apart
exactly what would have gone up.

Every symbol here is a placeholder (`XYZ`, `ABCD`, `WXY`). Tick names no real
security, in tests as anywhere else.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tick.agents import (
    EMIT_TOOL_NAME,
    ModelAgent,
    ModelAgentSpec,
    ModelReply,
    ModelRequest,
    parse_agent_spec,
    read_model_reply,
)
from tick.engine import FixtureMarketData, PortfolioState, Position
from tick.records import TICK_HOME_ENV

MARKET_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "market"

#: A Tuesday inside the 2026 calendar, mid-session, in UTC.
NOW = datetime(2026, 9, 1, 15, 0, tzinfo=UTC)

#: What the user wrote. Test material, and deliberately not a strategy: the
#: point of these tests is that Tick passes the user's words through unchanged,
#: not that any particular words work.
INSTRUCTIONS = "These are my own instructions.\nHold XYZ. Nothing else.\n"


class NetworkUsedInATest(RuntimeError):
    """A test tried to open a socket. Nothing in Tick's tests may."""


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise NetworkUsedInATest(
            "a test opened a socket. Model-agent tests answer from a fake client; "
            "nothing here may reach a provider."
        )

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


@pytest.fixture(autouse=True)
def tick_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    home = tmp_path / "tick-home"
    monkeypatch.setenv(TICK_HOME_ENV, str(home))
    yield home


def model_spec_document(
    *,
    universe: list[str] | None = None,
    provider: str = "anthropic",
    model: str = "claude-opus-5",
    cadence: dict[str, Any] | None = None,
    cage: dict[str, Any] | None = None,
    name: str = "Model agent under test",
) -> dict[str, Any]:
    """A model agent's document, so a test can bend one field."""
    return {
        "kind": "model_agent",
        "name": name,
        "version": 1,
        "universe": universe if universe is not None else ["XYZ", "ABCD"],
        "cadence": cadence if cadence is not None else {"kind": "daily_close"},
        "provider": provider,
        "model": model,
        "cage": cage
        if cage is not None
        else {
            "max_position_pct": "100.00",
            "max_positions": 5,
            "max_order_notional": "1000000.00",
            "max_daily_drawdown_pct": "50.00",
            "allowed_session": "regular_hours",
        },
    }


def build_model_spec(**kwargs: Any) -> ModelAgentSpec:
    spec = parse_agent_spec(model_spec_document(**kwargs))
    assert isinstance(spec, ModelAgentSpec)
    return spec


def reply_object(
    *,
    intents: list[Any] | None = None,
    model: str = "claude-opus-5-20260401",
    tool: str | None = EMIT_TOOL_NAME,
    stop_reason: str = "tool_use",
    text: str | None = None,
    stop_details: dict[str, str] | None = None,
    payload: dict[str, Any] | None = None,
) -> Any:
    """A provider reply in the SDK's shape, for `read_model_reply` to read."""
    blocks: list[Any] = []
    if text is not None:
        blocks.append(SimpleNamespace(type="text", text=text))
    if tool is not None:
        blocks.append(
            SimpleNamespace(
                type="tool_use",
                name=tool,
                input=payload if payload is not None else {"intents": intents or []},
            )
        )
    return SimpleNamespace(
        model=model,
        stop_reason=stop_reason,
        stop_details=SimpleNamespace(**stop_details) if stop_details else None,
        content=blocks,
    )


class FakeModelClient:
    """A `ModelClient` that answers with a prepared reply and keeps the request."""

    def __init__(self, reply: Any) -> None:
        self._reply = reply
        self.requests: list[ModelRequest] = []

    @classmethod
    def answering(cls, intents: list[Any], **kwargs: Any) -> FakeModelClient:
        return cls(reply_object(intents=intents, **kwargs))

    @property
    def request(self) -> ModelRequest:
        assert len(self.requests) == 1, f"expected exactly one call, got {len(self.requests)}"
        return self.requests[0]

    def propose(self, request: ModelRequest) -> ModelReply:
        self.requests.append(request)
        return read_model_reply(self._reply)


class ExplodingModelClient:
    """A client that raises. Used to pin that a tick stops rather than retries."""

    def __init__(self, error: Exception) -> None:
        self._error = error
        self.calls = 0

    def propose(self, request: ModelRequest) -> ModelReply:
        self.calls += 1
        raise self._error


def market(now: datetime = NOW) -> FixtureMarketData:
    """The placeholder market series, seen from `now`."""
    return FixtureMarketData.from_directory(MARKET_FIXTURES, now=now)


def portfolio(**held: int) -> PortfolioState:
    """An account holding whole shares of the placeholder symbols."""
    return PortfolioState(
        cash=Decimal("10000.00"),
        positions={
            symbol: Position(symbol=symbol, qty=qty, avg_cost=Decimal("100.00"))
            for symbol, qty in held.items()
        },
    )


def agent_for(client: Any, *, spec: ModelAgentSpec | None = None, instructions: str = INSTRUCTIONS):
    return ModelAgent(spec or build_model_spec(), client=client, instructions=instructions)
