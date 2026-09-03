"""Adapter-level drift refusal: no raw profile and no call before authorization."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from tick.broker import (
    Category,
    DiscoveredTool,
    ProfileBroker,
    ProfileState,
    ProfileTool,
    contract_for,
    inventory_hash,
    verify_session_profile,
)
from tick.broker.errors import CapabilityUnmapped
from tick.broker.profile import (
    CANONICALIZER_VERSION,
    CATEGORIZER_VERSION,
    CATEGORY_REGISTRY_VERSION,
    PROFILE_FORMAT_VERSION,
    build_profile,
    mapping_hash,
)
from tick.engine import Unavailable

AT = datetime(2026, 9, 2, 12, 0, tzinfo=UTC)
SERVER = "https://agent.robinhood.com/mcp/trading"


def discovered(name, description="Declared in dollars."):
    return DiscoveredTool(
        name=name,
        title=None,
        description=description,
        input_schema={
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
            "additionalProperties": False,
        },
        output_schema=None,
        annotations=None,
        execution=None,
    )


def mapped(tool, category, arguments, result):
    contract = contract_for(tool)
    return ProfileTool(
        category=category,
        contract=contract,
        arguments=arguments,
        result=result,
        confirmed_contract_hash=contract.contract_hash,
        mapping_hash=mapping_hash(category, arguments, result),
        confirmed_at=AT,
        confirmed_by="terminal",
        categorizer_version=CATEGORIZER_VERSION,
        proved_contract_hash=None,
        proved_mapping_hash=None,
        proved_at=None,
        proof=None,
    )


def profile(*tools):
    return build_profile(
        server=SERVER,
        account_id="agentic-0001",
        tools={tool.contract.name: tool for tool in tools},
        inventory_hash=inventory_hash(tuple(tool.contract for tool in tools)),
        data_class="display_only",
        sanction="official",
        profile_format_version=PROFILE_FORMAT_VERSION,
        canonicalizer_version=CANONICALIZER_VERSION,
        category_registry_version=CATEGORY_REGISTRY_VERSION,
        state=ProfileState.CONFIRMED,
        observed_inventory_hash=None,
        drift=(),
    )


class Session:
    def __init__(self, tools):
        self.tools = tuple(tools)
        self.calls = []
        self.callback = None

    def list_tools(self):
        return self.tools

    def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return {"price": "184.20", "at": AT.isoformat()}

    def on_tools_changed(self, callback):
        self.callback = callback


def broker_for(stored, session):
    verified = verify_session_profile(
        stored,
        session,
        server=SERVER,
        account_id="agentic-0001",
        confirmation_recorded=True,
    )
    return ProfileBroker(
        verified, max_cancels=1, kill_switch=lambda: False, approval_mode="standing"
    )


def test_profile_broker_rejects_a_raw_profile():
    with pytest.raises(TypeError, match="VerifiedSessionProfile"):
        ProfileBroker(profile(), max_cancels=1, kill_switch=lambda: False, approval_mode="standing")


def test_description_drifted_quote_returns_unavailable_before_tools_call():
    quote = discovered("get_quote")
    stored = profile(
        mapped(
            quote,
            Category.READ_QUOTE,
            {"symbol": "{symbol}"},
            {"price": "price", "asof": "at"},
        )
    )
    session = Session([discovered("get_quote", "Declared in cents.")])
    result = broker_for(stored, session).quote("XYZ")
    assert isinstance(result, Unavailable)
    assert "drifted" in result.reason
    assert session.calls == []


def test_an_unchanged_quote_works_when_an_unrelated_new_tool_appears():
    quote = discovered("get_quote")
    stored = profile(
        mapped(
            quote,
            Category.READ_QUOTE,
            {"symbol": "{symbol}"},
            {"price": "price", "asof": "at"},
        )
    )
    session = Session([quote, discovered("new_tool")])
    result = broker_for(stored, session).quote("XYZ")
    assert result.price.as_tuple() == (0, (1, 8, 4, 2, 0), -2)
    assert result.source == "agent.robinhood.com"
    assert result.price_source == "agent.robinhood.com"
    assert result.data_class == "display_only"
    assert session.calls == [("get_quote", {"symbol": "XYZ"})]


def test_an_unchanged_quote_works_when_an_optional_sibling_drifts():
    quote = discovered("get_quote")
    history = discovered("get_history")
    stored = profile(
        mapped(
            quote,
            Category.READ_QUOTE,
            {"symbol": "{symbol}"},
            {"price": "price", "asof": "at"},
        ),
        mapped(
            history,
            Category.READ_HISTORY,
            {"symbol": "{symbol}"},
            {
                "items": "bars",
                "timestamp": "at",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            },
        ),
    )
    changed_history = discovered("get_history", "Changed units.")
    session = Session([quote, changed_history])
    result = broker_for(stored, session).quote("XYZ")
    assert result.price.as_tuple() == (0, (1, 8, 4, 2, 0), -2)
    assert session.calls == [("get_quote", {"symbol": "XYZ"})]


def test_bars_refuses_without_a_history_tool():
    session = Session([])
    result = broker_for(profile(), session).bars("XYZ", 2)
    assert isinstance(result, Unavailable)
    assert result.reason == (
        "no history tool is mapped in this profile; confirm one or use paper data"
    )
    assert session.calls == []


def test_bars_are_read_only_through_mapped_history_with_profile_provenance():
    history = DiscoveredTool(
        name="get_history",
        title=None,
        description="OHLC prices are declared in dollars; volume is whole shares.",
        input_schema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "count": {"type": "integer", "minimum": 1},
            },
            "required": ["symbol", "count"],
            "additionalProperties": False,
        },
        output_schema=None,
        annotations=None,
        execution=None,
    )
    stored = profile(
        mapped(
            history,
            Category.READ_HISTORY,
            {"symbol": "{symbol}", "count": "{count}"},
            {
                "items": "bars",
                "timestamp": "at",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            },
        )
    )

    class HistorySession(Session):
        def call_tool(self, name, arguments):
            self.calls.append((name, arguments))
            return {
                "bars": [
                    {
                        "at": AT.isoformat(),
                        "open": "181.00",
                        "high": "185.00",
                        "low": "180.00",
                        "close": "184.20",
                        "volume": 120,
                    }
                ]
            }

    session = HistorySession([history])
    result = broker_for(stored, session).bars("XYZ", 1)
    assert not isinstance(result, Unavailable)
    assert result[0].close.as_tuple() == (0, (1, 8, 4, 2, 0), -2)
    assert result[0].price_source == "agent.robinhood.com"
    assert result[0].data_class == "display_only"
    assert session.calls == [("get_history", {"symbol": "XYZ", "count": 1})]


def test_raw_name_call_can_never_reach_a_denied_or_unknown_tool():
    session = Session([])
    broker = broker_for(profile(), session)
    with pytest.raises(CapabilityUnmapped, match="positive allowlist"):
        broker.call_named("transfer_funds", {})
    assert session.calls == []


def test_tools_list_changed_revokes_the_binding_before_another_read():
    quote = discovered("get_quote")
    stored = profile(
        mapped(
            quote,
            Category.READ_QUOTE,
            {"symbol": "{symbol}"},
            {"price": "price", "asof": "at"},
        )
    )
    session = Session([quote])
    broker = broker_for(stored, session)
    assert session.callback is not None
    session.callback("tools/list changed")
    result = broker.quote("XYZ")
    assert isinstance(result, Unavailable)
    assert "revoked" in result.reason
    assert session.calls == []
