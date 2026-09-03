"""The Robinhood adapter, against a mock brokerage over in-memory MCP.

The audit constraints are the subject here, one test each: reads are scoped to
the configured Agentic account and every other account's rows are dropped
inside the adapter; a sell may never exceed what that account holds; an
unmapped capability refuses instead of guessing; money is `Decimal`; a number
that is not there is `Unavailable`; and a transport failure STOPS rather than
becoming a rejection that would tell the caller nothing was placed.

No socket is opened anywhere in this file.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from mcp.server.mcpserver import MCPServer

from tick.broker import (
    BrokerPort,
    Cancelled,
    Category,
    Fill,
    ProfileBroker,
    ProfileState,
    ProfileTool,
    RejectCode,
    Rejected,
    contract_for,
    inventory_hash,
    verify_session_profile,
)
from tick.broker.errors import BrokerUnavailable, ToolResultUnreadable
from tick.broker.mcp_session import MCPSession
from tick.broker.mock_mcp import MockBrokerage
from tick.broker.profile import (
    CANONICALIZER_VERSION,
    CATEGORIZER_VERSION,
    CATEGORY_REGISTRY_VERSION,
    PROFILE_FORMAT_VERSION,
    ProofResult,
    build_profile,
    mapping_hash,
)
from tick.broker.robinhood import RobinhoodMCPBroker
from tick.broker.toolmap import Capability, CapabilityMapping, ToolMap, propose
from tick.engine import OrderIntent, Unavailable
from tick.spec import Side

from .conftest import TIMEOUT, memory_opener

AT = datetime(2026, 9, 1, 15, 30, tzinfo=UTC)
SERVER = "https://agent.robinhood.com/mcp/trading"

PROFILE_CATEGORY = {
    Capability.QUOTE: Category.READ_QUOTE,
    Capability.POSITIONS: Category.READ_POSITIONS,
    Capability.ACCOUNT: Category.READ_BALANCES,
    Capability.PLACE_ORDER: Category.ORDER_PLACE,
    Capability.CANCEL_ORDER: Category.ORDER_CANCEL,
    Capability.LIST_ORDERS: Category.READ_ORDERS,
}


def an_intent(symbol: str = "XYZ", side: Side = Side.BUY, qty: int = 3) -> OrderIntent:
    return OrderIntent(
        source="rule:placeholder",
        symbol=symbol,
        side=side,
        qty=qty,
        est_price=Decimal("184.20"),
        est_notional=Decimal("184.20") * qty,
        price_asof=AT,
        price_source="fixture",
        reason="the placeholder rule fired",
    )


def a_map(session: MCPSession, account_id: str) -> ToolMap:
    return propose(
        session.list_tools(),
        account_id=account_id,
        server_name=session.server_name,
        discovered_at=AT,
    ).to_tool_map()


def without(tool_map: ToolMap, capability: Capability) -> ToolMap:
    """The same map with one capability unmapped — what a partial discovery leaves."""
    return ToolMap(
        account_id=tool_map.account_id,
        server_name=tool_map.server_name,
        discovered_at=tool_map.discovered_at,
        capabilities={k: v for k, v in tool_map.capabilities.items() if k is not capability},
    )


def remapped(tool_map: ToolMap, capability: Capability, result: dict[str, str]) -> ToolMap:
    """The same map with one capability's result paths replaced."""
    existing = tool_map.capabilities[capability]
    return ToolMap(
        account_id=tool_map.account_id,
        server_name=tool_map.server_name,
        discovered_at=tool_map.discovered_at,
        capabilities={
            **tool_map.capabilities,
            capability: CapabilityMapping(
                tool=existing.tool, arguments=existing.arguments, result=result
            ),
        },
    )


def adapted_broker(session: MCPSession, tool_map: ToolMap, *, max_cancels: int) -> ProfileBroker:
    """Exercise the replacement adapter with mappings from the prototype fixture."""
    discovered = session.list_tools()
    contracts = {tool.name: contract_for(tool) for tool in discovered}
    tools: dict[str, ProfileTool] = {}
    for capability, legacy in tool_map.capabilities.items():
        category = PROFILE_CATEGORY[capability]
        contract = contracts[legacy.tool]
        digest = mapping_hash(category, legacy.arguments, legacy.result)
        tools[legacy.tool] = ProfileTool(
            category=category,
            contract=contract,
            arguments=legacy.arguments,
            result=legacy.result,
            confirmed_contract_hash=contract.contract_hash,
            mapping_hash=digest,
            confirmed_at=AT,
            confirmed_by="terminal",
            categorizer_version=CATEGORIZER_VERSION,
            proved_contract_hash=contract.contract_hash,
            proved_mapping_hash=digest,
            proved_at=AT,
            proof=ProofResult(
                success=True,
                resolved=tuple(legacy.result),
                unresolved={},
                detail="fixture proof of the exact contract and mapping",
            ),
        )
    profile = build_profile(
        server=SERVER,
        account_id=tool_map.account_id,
        tools=tools,
        inventory_hash=inventory_hash(tuple(contracts.values())),
        data_class="display_only",
        sanction="official",
        profile_format_version=PROFILE_FORMAT_VERSION,
        canonicalizer_version=CANONICALIZER_VERSION,
        category_registry_version=CATEGORY_REGISTRY_VERSION,
        state=ProfileState.CONFIRMED,
        observed_inventory_hash=None,
        drift=(),
    )
    verified = verify_session_profile(
        profile,
        session,
        server=SERVER,
        account_id=tool_map.account_id,
        confirmation_recorded=True,
    )
    return ProfileBroker(
        verified,
        max_cancels=max_cancels,
        kill_switch=lambda: False,
        approval_mode="standing",
    )


@pytest.fixture
def session(mock_server: MCPServer):
    handle = MCPSession(memory_opener(mock_server), timeout_seconds=TIMEOUT)
    handle.open()
    yield handle
    handle.close()


@pytest.fixture
def tool_map(session: MCPSession, brokerage: MockBrokerage) -> ToolMap:
    return a_map(session, brokerage.agentic_account)


@pytest.fixture
def broker(session: MCPSession, tool_map: ToolMap) -> ProfileBroker:
    return adapted_broker(session, tool_map, max_cancels=2)


# ----------------------------------------------------------------------
# It is a broker
# ----------------------------------------------------------------------


def test_it_satisfies_the_broker_port(broker: ProfileBroker):
    assert isinstance(broker, BrokerPort)


def test_the_old_public_name_keeps_the_verified_constructor_boundary():
    assert issubclass(RobinhoodMCPBroker, ProfileBroker)


def test_a_negative_cancel_limit_is_refused(session: MCPSession, tool_map: ToolMap):
    with pytest.raises(ValueError):
        adapted_broker(session, tool_map, max_cancels=-1)


# ----------------------------------------------------------------------
# Quotes
# ----------------------------------------------------------------------


def test_a_quote_is_read_as_an_exact_decimal_with_its_provenance(broker: ProfileBroker):
    quote = broker.quote("XYZ")

    assert not isinstance(quote, Unavailable)
    assert quote.price == Decimal("184.20")
    assert isinstance(quote.price, Decimal)
    assert quote.asof == AT
    assert quote.source == "agent.robinhood.com"


def test_a_quote_the_broker_has_no_price_for_is_unavailable(broker: ProfileBroker):
    """Invariant 5: not zero, not the last price seen, not a guess."""
    quote = broker.quote("NOPE")

    assert isinstance(quote, Unavailable)
    assert "no 'last_price'" in quote.reason


def test_an_undated_quote_is_refused_rather_than_stamped_locally(
    session: MCPSession, tool_map: ToolMap
):
    """Never date a price the broker did not date."""
    broken = remapped(tool_map, Capability.QUOTE, {"price": "last_price", "asof": "not_a_field"})
    broker = adapted_broker(session, broken, max_cancels=1)

    quote = broker.quote("XYZ")

    assert isinstance(quote, Unavailable)
    assert "no usable 'not_a_field'" in quote.reason


def test_an_unmapped_quote_capability_is_unavailable_not_an_exception(
    session: MCPSession, tool_map: ToolMap
):
    broker = adapted_broker(session, without(tool_map, Capability.QUOTE), max_cancels=1)

    quote = broker.quote("XYZ")

    assert isinstance(quote, Unavailable)
    assert "tick broker propose" in quote.reason


def test_the_trading_connection_supplies_no_history(broker: ProfileBroker):
    """A short series would silently change what every indicator means."""
    bars = broker.bars("XYZ", 20)

    assert isinstance(bars, Unavailable)
    assert "no history tool is mapped" in bars.reason


# ----------------------------------------------------------------------
# Read scoping — the audit's central constraint
# ----------------------------------------------------------------------


def test_state_holds_only_the_configured_accounts_positions_and_cash(
    broker: ProfileBroker, brokerage: MockBrokerage
):
    state = broker.state()

    assert state.cash == Decimal("5000.00")
    assert sorted(state.positions) == ["ABCD", "XYZ"]
    assert state.positions["XYZ"].qty == 12
    assert state.positions["XYZ"].avg_cost == Decimal("170.00")
    assert "WXY" not in state.positions


def test_no_other_account_id_is_ever_requested_retained_or_recorded(
    broker: ProfileBroker, brokerage: MockBrokerage
):
    """The grant reads every account. Tick asks about one and keeps one."""
    state = broker.state()
    broker.quote("XYZ")
    order_ids = broker.order_ids()

    asked = brokerage.argument_text()
    assert brokerage.agentic_account in asked
    assert brokerage.other_account not in asked

    kept = state.model_dump_json() + repr(order_ids)
    assert brokerage.other_account not in kept
    assert "WXY" not in kept


def test_orders_are_listed_for_the_configured_account_only(
    broker: ProfileBroker, brokerage: MockBrokerage
):
    """The mock returns another account's order; the adapter drops it."""
    broker.place(an_intent())

    assert broker.order_ids() == ("mock-order-0001",)
    assert "mock-order-other" not in repr(broker.order_ids())


def test_a_missing_average_cost_refuses_rather_than_reading_as_zero(
    session: MCPSession, tool_map: ToolMap
):
    """A zero basis would report the whole holding as profit (inherited from investy)."""
    broken = remapped(
        tool_map,
        Capability.POSITIONS,
        {
            "items": "positions",
            "account": "account",
            "symbol": "symbol",
            "quantity": "quantity",
            "average_cost": "not_a_field",
        },
    )
    broker = adapted_broker(session, broken, max_cancels=1)

    with pytest.raises(ToolResultUnreadable) as caught:
        broker.state()

    assert "average cost of" in str(caught.value)


def test_cash_for_an_account_the_broker_does_not_return_refuses(
    session: MCPSession, brokerage: MockBrokerage
):
    """A scoped read that matches nothing is a stop, not an empty account."""
    tool_map = a_map(session, "agentic-not-yours")
    broker = adapted_broker(session, tool_map, max_cancels=1)

    with pytest.raises(ToolResultUnreadable) as caught:
        broker.state()

    assert "returned 0 balance rows" in str(caught.value)


# ----------------------------------------------------------------------
# Placing, and long-only
# ----------------------------------------------------------------------


def test_a_buy_fills_through_the_mapped_tool_in_the_configured_account(
    broker: ProfileBroker, brokerage: MockBrokerage
):
    result = broker.place(an_intent())

    assert isinstance(result, Fill)
    assert result.price == Decimal("184.20")
    assert result.qty == 3
    assert result.side is Side.BUY
    assert result.ts == AT
    placed = [args for tool, args in brokerage.calls if tool == "place_order"]
    assert placed == [
        {
            "account_id": brokerage.agentic_account,
            "symbol": "XYZ",
            "side": "buy",
            "quantity": "3",
        }
    ]


def test_a_sell_within_the_position_closes_part_of_it(
    broker: ProfileBroker, brokerage: MockBrokerage
):
    result = broker.place(an_intent(side=Side.SELL, qty=5))

    assert isinstance(result, Fill)
    assert result.side is Side.SELL
    assert brokerage.held(brokerage.agentic_account, "XYZ") == 7


def test_a_sell_larger_than_the_position_is_refused_whole_and_never_sent(
    broker: ProfileBroker, brokerage: MockBrokerage
):
    """Long only: refuse, never truncate to a smaller sell, never go short."""
    result = broker.place(an_intent(side=Side.SELL, qty=13))

    assert isinstance(result, Rejected)
    assert result.code is RejectCode.SELL_EXCEEDS_POSITION
    assert "order is refused whole" in result.reason
    assert [tool for tool, _ in brokerage.calls if tool == "place_order"] == []
    assert brokerage.held(brokerage.agentic_account, "XYZ") == 12


def test_a_sell_of_something_not_held_is_refused(broker: ProfileBroker):
    result = broker.place(an_intent(symbol="WXY", side=Side.SELL, qty=1))

    assert isinstance(result, Rejected)
    assert result.code is RejectCode.NO_POSITION_TO_SELL
    assert "never opens a short" in result.reason


def test_a_sell_is_refused_when_the_position_cannot_be_read(
    session: MCPSession, tool_map: ToolMap, brokerage: MockBrokerage
):
    """An unverifiable long-only check is not a check."""
    broker = adapted_broker(session, without(tool_map, Capability.POSITIONS), max_cancels=1)

    result = broker.place(an_intent(side=Side.SELL, qty=1))

    assert isinstance(result, Rejected)
    assert result.code is RejectCode.CAPABILITY_UNMAPPED
    assert "no confirmed tool is mapped to read.positions" in result.reason
    assert [tool for tool, _ in brokerage.calls if tool == "place_order"] == []


def test_an_unmapped_place_order_refuses_rather_than_guessing_a_tool(
    session: MCPSession, tool_map: ToolMap, brokerage: MockBrokerage
):
    """Invariant 7 with money on it."""
    broker = adapted_broker(session, without(tool_map, Capability.PLACE_ORDER), max_cancels=1)

    result = broker.place(an_intent())

    assert isinstance(result, Rejected)
    assert result.code is RejectCode.CAPABILITY_UNMAPPED
    assert brokerage.calls == []


def test_an_order_the_broker_refuses_is_reported_with_its_reason(
    session: MCPSession, brokerage: MockBrokerage
):
    """The mock confines trading to the Agentic account, as the real broker does."""
    tool_map = a_map(session, brokerage.other_account)
    broker = adapted_broker(session, tool_map, max_cancels=1)

    result = broker.place(an_intent())

    assert isinstance(result, Rejected)
    assert result.code is RejectCode.BROKER_REFUSED
    assert "not agentic" in result.reason


def test_an_unreadable_fill_says_the_order_may_exist(session: MCPSession, tool_map: ToolMap):
    """A rejection means nothing was placed; this is not that, and says so."""
    broken = remapped(
        tool_map,
        Capability.PLACE_ORDER,
        {
            "order_id": "order_id",
            "quantity": "not_a_field",
            "price": "filled_price",
            "filled_at": "filled_at",
        },
    )
    broker = adapted_broker(session, broken, max_cancels=1)

    result = broker.place(an_intent())

    assert isinstance(result, Rejected)
    assert result.code is RejectCode.ORDER_OUTCOME_UNKNOWN
    assert "MAY HAVE BEEN ACCEPTED" in result.reason


# ----------------------------------------------------------------------
# Cancelling, and the cancel-ratio guard
# ----------------------------------------------------------------------


def test_a_cancel_goes_through_the_mapped_tool(broker: ProfileBroker):
    broker.place(an_intent())

    result = broker.cancel("mock-order-0001")

    assert isinstance(result, Cancelled)
    assert result.order_id == "mock-order-0001"
    assert result.ts == AT


def test_an_unknown_order_is_a_readable_refusal(broker: ProfileBroker):
    result = broker.cancel("no-such-order")

    assert isinstance(result, Rejected)
    assert result.code is RejectCode.BROKER_REFUSED


def test_cancels_are_bounded_so_a_loop_cannot_get_the_account_terminated(
    broker: ProfileBroker,
):
    broker.cancel("a")
    broker.cancel("b")

    result = broker.cancel("c")

    assert isinstance(result, Rejected)
    assert result.code is RejectCode.CANCEL_LIMIT_REACHED
    assert broker.cancels == 2


def test_an_unmapped_cancel_refuses(session: MCPSession, tool_map: ToolMap):
    broker = adapted_broker(session, without(tool_map, Capability.CANCEL_ORDER), max_cancels=1)

    result = broker.cancel("mock-order-0001")

    assert isinstance(result, Rejected)
    assert result.code is RejectCode.CAPABILITY_UNMAPPED


# ----------------------------------------------------------------------
# Fail safe
# ----------------------------------------------------------------------


def test_a_dropped_session_stops_the_runtime_and_is_not_a_rejection(
    session: MCPSession, tool_map: ToolMap
):
    """A rejection claims nothing was placed. A dropped connection cannot claim that."""
    broker = adapted_broker(session, tool_map, max_cancels=1)
    session.close()
    session._failure = "the broker's MCP session failed: dropped"

    with pytest.raises(BrokerUnavailable):
        broker.place(an_intent())
    with pytest.raises(BrokerUnavailable):
        broker.state()
    with pytest.raises(BrokerUnavailable):
        broker.quote("XYZ")
