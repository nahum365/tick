"""The tool map: discovered, proposed, confirmed — and refusing where it is not.

Invariant 7 is the subject of this file. The proposal is a suggestion and is
asserted to write nothing; the adopted map is a document; and a capability
nobody mapped raises rather than falling through to a plausible-looking tool.
Invariant 5 is the other half: every reader below answers `Unavailable` rather
than a zero when the number is missing, unparsable, or a binary float.
"""

from __future__ import annotations

import json
import stat
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from mcp.server.mcpserver import MCPServer

from tick.broker.errors import CapabilityUnmapped, ToolResultUnreadable
from tick.broker.mcp_session import MCPSession
from tick.broker.toolmap import (
    Capability,
    CapabilityMapping,
    DiscoveredTool,
    ToolMap,
    decimal_at,
    dig,
    load_tool_map,
    propose,
    save_tool_map,
    text_at,
    timestamp_at,
    toolmap_path,
    whole_at,
)
from tick.engine import Unavailable

from .conftest import TIMEOUT, memory_opener

AT = datetime(2026, 9, 1, 15, 30, tzinfo=UTC)
ACCOUNT = "agentic-0001"


def discovered(server: MCPServer) -> tuple[DiscoveredTool, ...]:
    with MCPSession(memory_opener(server), timeout_seconds=TIMEOUT) as session:
        return session.list_tools()


def a_mapping(**overrides) -> CapabilityMapping:
    base = {
        "tool": "get_quote",
        "arguments": {"symbol": "{symbol}"},
        "result": {"price": "last_price", "asof": "quoted_at"},
    }
    return CapabilityMapping(**{**base, **overrides})


def a_map(capabilities=None, **overrides) -> ToolMap:
    base = {
        "account_id": ACCOUNT,
        "server_name": "mock-brokerage",
        "discovered_at": AT,
        "capabilities": capabilities
        if capabilities is not None
        else {Capability.QUOTE: a_mapping()},
    }
    return ToolMap(**{**base, **overrides})


# ----------------------------------------------------------------------
# Reading numbers out of a mapped answer
# ----------------------------------------------------------------------


def test_a_dotted_path_reaches_into_lists_and_objects():
    payload = {"positions": [{"symbol": "XYZ", "quantity": "12"}]}

    assert dig(payload, "positions.0.symbol") == "XYZ"
    assert dig(payload, "positions.1.symbol") is None
    assert dig(payload, "positions.symbol") is None
    assert dig(payload, "nothing.here") is None


def test_a_decimal_string_is_read_exactly():
    assert decimal_at({"price": "184.20"}, "price", "quote") == Decimal("184.20")
    assert decimal_at({"qty": 12}, "qty", "quantity") == Decimal(12)


def test_a_missing_number_is_unavailable_and_never_zero():
    """Invariant 5. A zero here would report an empty account as a funded one."""
    result = decimal_at({"price": None}, "price", "quote for XYZ")

    assert isinstance(result, Unavailable)
    assert result.what == "quote for XYZ"
    assert "no 'price'" in result.reason


def test_a_json_float_is_refused_rather_than_rounded():
    """A binary float is an approximation of money; Tick does not adopt one."""
    result = decimal_at({"price": 184.2}, "price", "quote")

    assert isinstance(result, Unavailable)
    assert "binary approximation" in result.reason


def test_a_boolean_is_not_a_number():
    result = decimal_at({"price": True}, "price", "quote")

    assert isinstance(result, Unavailable)
    assert "true/false" in result.reason


def test_text_that_is_not_a_number_is_unavailable():
    result = decimal_at({"price": "unavailable"}, "price", "quote")

    assert isinstance(result, Unavailable)
    assert "not a number" in result.reason


def test_a_fractional_share_count_is_refused():
    result = whole_at({"quantity": "1.5"}, "quantity", "position in XYZ")

    assert isinstance(result, Unavailable)
    assert "whole shares only" in result.reason
    assert whole_at({"quantity": "12"}, "quantity", "position") == 12


def test_a_timestamp_without_a_zone_is_refused():
    """Never date a price you were not given a zone for."""
    naive = timestamp_at({"at": "2026-09-01T15:30:00"}, "at", "quote time")

    assert isinstance(naive, Unavailable)
    assert "no timezone" in naive.reason
    assert timestamp_at({"at": "2026-09-01T15:30:00Z"}, "at", "quote time") == AT


def test_missing_text_is_unavailable():
    assert isinstance(text_at({"id": ""}, "id", "order id"), Unavailable)
    assert text_at({"id": " o-1 "}, "id", "order id") == "o-1"


# ----------------------------------------------------------------------
# The map itself
# ----------------------------------------------------------------------


def test_an_unmapped_capability_refuses_and_says_how_to_map_it():
    """Invariant 7: refuse, never fall through to a plausible-looking tool."""
    tool_map = a_map()

    with pytest.raises(CapabilityUnmapped) as caught:
        tool_map.mapping_for(Capability.PLACE_ORDER)

    assert "place_order" in str(caught.value)
    assert "tick broker propose" in str(caught.value)
    assert Capability.PLACE_ORDER in tool_map.unmapped()


def test_a_map_must_name_the_account_its_reads_are_scoped_to():
    with pytest.raises(ValueError, match="Agentic account"):
        a_map(account_id="  ")


def test_a_mapping_may_not_ask_tick_for_a_value_it_should_not_send():
    with pytest.raises(ValueError, match="no business supplying"):
        a_map(capabilities={Capability.QUOTE: a_mapping(arguments={"symbol": "{account_id}"})})


def test_an_order_mapping_missing_a_placeholder_is_refused():
    """An order without a side, a size or an account is a different order."""
    with pytest.raises(ValueError, match="different order"):
        a_map(
            capabilities={
                Capability.PLACE_ORDER: CapabilityMapping(
                    tool="place_order",
                    arguments={"account_id": "{account_id}", "symbol": "{symbol}"},
                    result={
                        "order_id": "order_id",
                        "quantity": "filled_quantity",
                        "price": "filled_price",
                        "filled_at": "filled_at",
                    },
                )
            }
        )


def test_a_mapping_that_says_nothing_about_a_needed_number_is_refused():
    with pytest.raises(ValueError, match="never by guesswork"):
        a_map(capabilities={Capability.QUOTE: a_mapping(result={"price": "last_price"})})


def test_rendering_arguments_fills_only_what_the_call_supplies():
    mapping = a_mapping()

    assert mapping.render({"symbol": "XYZ"}) == {"symbol": "XYZ"}
    with pytest.raises(CapabilityUnmapped, match="has no such value"):
        mapping.render({})


# ----------------------------------------------------------------------
# Proposing a mapping from what the broker declared
# ----------------------------------------------------------------------


def test_the_proposal_maps_every_capability_the_mock_serves(mock_server: MCPServer):
    proposal = propose(
        discovered(mock_server), account_id=ACCOUNT, server_name="mock-brokerage", discovered_at=AT
    )

    bound = {capability: mapping.tool for capability, mapping in proposal.capabilities.items()}
    assert bound == {
        Capability.QUOTE: "get_quote",
        Capability.POSITIONS: "get_positions",
        Capability.ACCOUNT: "get_accounts",
        Capability.PLACE_ORDER: "place_order",
        Capability.CANCEL_ORDER: "cancel_order",
        Capability.LIST_ORDERS: "list_orders",
    }
    assert proposal.unmapped == {}
    assert proposal.to_tool_map().account_id == ACCOUNT


def test_the_proposal_says_which_result_paths_it_could_not_check(mock_server: MCPServer):
    """A convention offered as one. The user confirms it; Tick does not adopt it alone."""
    proposal = propose(
        discovered(mock_server), account_id=ACCOUNT, server_name=None, discovered_at=AT
    )

    assert Capability.QUOTE in proposal.notes
    assert "declares no output schema" in proposal.notes[Capability.QUOTE]


def test_the_proposal_scopes_every_account_taking_read_to_the_configured_account(
    mock_server: MCPServer,
):
    proposal = propose(
        discovered(mock_server), account_id=ACCOUNT, server_name=None, discovered_at=AT
    )

    for capability in (Capability.POSITIONS, Capability.LIST_ORDERS, Capability.PLACE_ORDER):
        arguments = proposal.capabilities[capability].arguments
        assert arguments["account_id"] == "{account_id}"


def test_a_capability_no_tool_resembles_is_left_unmapped():
    server = MCPServer(name="thin-brokerage")

    @server.tool(description="Only a quote.", structured_output=True)
    def get_quote(symbol: str) -> dict[str, str]:
        return {"symbol": symbol}

    proposal = propose(discovered(server), account_id=ACCOUNT, server_name=None, discovered_at=AT)

    assert set(proposal.capabilities) == {Capability.QUOTE}
    assert Capability.PLACE_ORDER in proposal.unmapped
    assert "refuses until you map it" in proposal.unmapped[Capability.PLACE_ORDER]


def test_a_tool_needing_an_argument_tick_cannot_fill_is_left_unmapped():
    """Tick will not invent an argument to send a broker."""
    server = MCPServer(name="odd-brokerage")

    @server.tool(description="Wants a venue.", structured_output=True)
    def get_quote(symbol: str, venue: str) -> dict[str, str]:
        return {"symbol": symbol, "venue": venue}

    proposal = propose(discovered(server), account_id=ACCOUNT, server_name=None, discovered_at=AT)

    assert Capability.QUOTE not in proposal.capabilities
    assert "'venue'" in proposal.unmapped[Capability.QUOTE]


def test_a_proposal_needs_the_account_it_will_scope_reads_to():
    with pytest.raises(ValueError, match="Agentic account id"):
        propose([], account_id="", server_name=None, discovered_at=AT)


# ----------------------------------------------------------------------
# The file on disk
# ----------------------------------------------------------------------


def test_nothing_is_written_until_the_map_is_saved(home: Path, mock_server: MCPServer):
    """Proposing is not adopting: only explicit legacy persistence writes."""
    propose(discovered(mock_server), account_id=ACCOUNT, server_name=None, discovered_at=AT)

    assert load_tool_map(home) is None
    assert not toolmap_path(home).exists()


def test_an_adopted_map_round_trips_and_is_private(home: Path, mock_server: MCPServer):
    """The map carries the user's account number, so it is 0600 like the token."""
    proposal = propose(
        discovered(mock_server), account_id=ACCOUNT, server_name="mock-brokerage", discovered_at=AT
    )

    path = save_tool_map(home, proposal.to_tool_map())
    loaded = load_tool_map(home)

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert loaded is not None
    assert loaded.account_id == ACCOUNT
    assert loaded.mapping_for(Capability.PLACE_ORDER).tool == "place_order"


def test_an_unreadable_map_refuses_every_capability(home: Path):
    toolmap_path(home).parent.mkdir(parents=True, exist_ok=True)
    toolmap_path(home).write_text(json.dumps({"account_id": ""}), encoding="utf-8")

    with pytest.raises(ToolResultUnreadable) as caught:
        load_tool_map(home)

    assert "every broker capability refuses" in str(caught.value)
