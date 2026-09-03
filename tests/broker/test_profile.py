"""Broker-profile content hashes and the per-tool authorization state table."""

from __future__ import annotations

import json
import stat
from datetime import UTC, datetime

import pytest

from tick.broker import (
    Category,
    DiscoveredTool,
    ProfileState,
    ProfileTool,
    ToolState,
    contract_for,
    inventory_hash,
    load_profile,
    propose_profile,
    verify_session_profile,
)
from tick.broker.errors import CapabilityUnmapped, ToolResultUnreadable
from tick.broker.profile import (
    CANONICALIZER_VERSION,
    CATEGORIZER_VERSION,
    CATEGORY_REGISTRY_VERSION,
    PROFILE_FORMAT_VERSION,
    ProofResult,
    build_profile,
    mapping_hash,
    profile_path,
)
from tick.records import write_private_file

AT = datetime(2026, 9, 2, 12, 0, tzinfo=UTC)
SERVER = "https://agent.robinhood.com/mcp/trading"
ACCOUNT = "agentic-0001"


def tool(
    name: str,
    *,
    description: str = "A declared operation.",
    input_schema=None,
    output_schema=None,
    annotations=None,
) -> DiscoveredTool:
    return DiscoveredTool(
        name=name,
        title=None,
        description=description,
        input_schema=(
            {"type": "object", "properties": {}} if input_schema is None else input_schema
        ),
        output_schema=output_schema,
        annotations=annotations,
        execution=None,
    )


def confirmed(
    discovered: DiscoveredTool,
    category: Category,
    *,
    arguments,
    result,
    proved: bool = False,
) -> ProfileTool:
    contract = contract_for(discovered)
    proof = ProofResult(success=True, resolved=tuple(result), unresolved={}, detail="proved")
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
        proved_contract_hash=contract.contract_hash if proved else None,
        proved_mapping_hash=mapping_hash(category, arguments, result) if proved else None,
        proved_at=AT if proved else None,
        proof=proof if proved else None,
    )


def profile(*mappings: ProfileTool):
    return build_profile(
        server=SERVER,
        account_id=ACCOUNT,
        tools={mapping.contract.name: mapping for mapping in mappings},
        inventory_hash=inventory_hash(tuple(mapping.contract for mapping in mappings)),
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

    def list_tools(self):
        return self.tools

    def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return {}


def test_category_registry_is_closed_and_unknown_is_not_a_denial():
    assert len(Category) == 16
    proposal = propose_profile(
        [tool("do_something")], server=SERVER, account_id=ACCOUNT, proposed_at=AT
    )
    assert proposal.tools["do_something"].category is None


def test_inventory_hash_is_order_independent_but_contract_content_is_not():
    quote = tool("get_quote", description="Price in dollars.")
    positions = tool("get_positions", description="Positions.")
    assert inventory_hash([quote, positions]) == inventory_hash([positions, quote])

    changed_description = quote.model_copy(update={"description": "Price in cents."})
    assert contract_for(quote).shape_hash == contract_for(changed_description).shape_hash
    assert contract_for(quote).contract_hash != contract_for(changed_description).contract_hash


def test_profile_hash_covers_approval_but_not_runtime_drift_state():
    quote = tool("get_quote", description="Price in dollars.")
    mapping = confirmed(
        quote,
        Category.READ_QUOTE,
        arguments={},
        result={"price": "price", "asof": "at"},
    )
    stored = profile(mapping)
    observed = build_profile(
        server=stored.server,
        account_id=stored.account_id,
        tools=stored.tools,
        inventory_hash=stored.inventory_hash,
        data_class=stored.data_class,
        sanction=stored.sanction,
        profile_format_version=stored.profile_format_version,
        canonicalizer_version=stored.canonicalizer_version,
        category_registry_version=stored.category_registry_version,
        state=ProfileState.DRIFTED,
        observed_inventory_hash="sha256:" + "0" * 64,
        drift=(),
    )
    assert observed.profile_hash == stored.profile_hash

    changed_mapping = confirmed(
        quote,
        Category.READ_QUOTE,
        arguments={},
        result={"price": "last_price", "asof": "at"},
    )
    assert profile(changed_mapping).profile_hash != stored.profile_hash


def test_annotations_are_hashed_but_never_authorize_a_tool():
    unknown = tool("do_something", annotations={"readOnlyHint": True})
    proposal = propose_profile([unknown], server=SERVER, account_id=ACCOUNT, proposed_at=AT)
    assert proposal.tools[unknown.name].category is None
    changed = unknown.model_copy(update={"annotations": {"readOnlyHint": False}})
    assert contract_for(unknown).contract_hash != contract_for(changed).contract_hash


def test_a_changed_denied_tool_stays_denied_without_confirmation():
    transfer = tool("transfer_funds", description="Transfer funds between accounts.")
    contract = contract_for(transfer)
    denied = ProfileTool(
        category=Category.DENIED_TRANSFERS,
        contract=contract,
        arguments={},
        result={},
        confirmed_contract_hash=None,
        mapping_hash=mapping_hash(Category.DENIED_TRANSFERS, {}, {}),
        confirmed_at=None,
        confirmed_by=None,
        categorizer_version=CATEGORIZER_VERSION,
        proved_contract_hash=None,
        proved_mapping_hash=None,
        proved_at=None,
        proof=None,
    )
    changed = transfer.model_copy(update={"description": "Changed transfer description."})
    verified = verify_session_profile(
        profile(denied),
        Session([changed]),
        server=SERVER,
        account_id=ACCOUNT,
        confirmation_recorded=True,
    )
    assert verified.states[transfer.name] is ToolState.DENIED


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "number", "maximum": float("inf")},
        {"$ref": "https://schemas.invalid/tool.json"},
        {"$ref": "#/$defs/missing", "$defs": {}},
    ],
)
def test_an_inventory_that_cannot_be_canonicalized_refuses_the_session(schema):
    with pytest.raises(ValueError, match="non-finite|remote|unresolved"):
        contract_for(tool("get_quote", input_schema=schema))


def test_duplicate_tool_names_invalidate_the_whole_session():
    duplicate = tool("get_quote")
    with pytest.raises(ToolResultUnreadable, match="whole session"):
        inventory_hash([duplicate, duplicate])


def test_order_mapping_requires_the_old_placeholders_and_result_roles():
    place = tool(
        "place_order",
        input_schema={
            "type": "object",
            "properties": {
                "account": {"type": "string"},
                "symbol": {"type": "string"},
                "side": {"type": "string"},
                "quantity": {"type": "string"},
            },
            "required": ["account", "symbol", "side", "quantity"],
        },
    )
    with pytest.raises(ValueError, match="different order"):
        confirmed(
            place,
            Category.ORDER_PLACE,
            arguments={"account": "{account_id}", "symbol": "{symbol}"},
            result={
                "order_id": "id",
                "quantity": "qty",
                "price": "price",
                "filled_at": "at",
            },
        )


def test_state_table_keeps_new_tools_unmapped_and_exact_tools_callable():
    quote = tool("get_quote")
    mapping = confirmed(
        quote,
        Category.READ_QUOTE,
        arguments={},
        result={"price": "price", "asof": "at"},
    )
    stored = profile(mapping)
    session = Session([quote, tool("new_unrelated")])
    verified = verify_session_profile(
        stored,
        session,
        server=SERVER,
        account_id=ACCOUNT,
        confirmation_recorded=True,
    )
    assert verified.states == {
        "get_quote": ToolState.CONFIRMED,
        "new_unrelated": ToolState.UNMAPPED,
    }
    assert verified.mapping_for(Category.READ_QUOTE, require_proof=False) is mapping


def test_changed_and_removed_mapped_tools_are_uncallable():
    quote = tool("get_quote", description="Price in dollars.")
    mapping = confirmed(
        quote,
        Category.READ_QUOTE,
        arguments={},
        result={"price": "price", "asof": "at"},
    )
    stored = profile(mapping)
    for live, state in (
        ([quote.model_copy(update={"description": "Price in cents."})], ToolState.DRIFTED),
        ([], ToolState.DRIFTED),
    ):
        verified = verify_session_profile(
            stored,
            Session(live),
            server=SERVER,
            account_id=ACCOUNT,
            confirmation_recorded=True,
        )
        assert verified.states["get_quote"] is state
        with pytest.raises(CapabilityUnmapped, match=state.value):
            verified.mapping_for(Category.READ_QUOTE, require_proof=False)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        (
            "input_schema",
            {"type": "object", "properties": {"symbol": {"type": "string"}}},
        ),
        (
            "output_schema",
            {"type": "object", "properties": {"price": {"type": "string"}}},
        ),
    ],
)
def test_any_complete_schema_change_drifts_only_that_mapped_tool(field, replacement):
    quote = tool("get_quote")
    stored = profile(
        confirmed(
            quote,
            Category.READ_QUOTE,
            arguments={},
            result={"price": "price", "asof": "at"},
        )
    )
    changed = quote.model_copy(update={field: replacement})
    verified = verify_session_profile(
        stored,
        Session([changed]),
        server=SERVER,
        account_id=ACCOUNT,
        confirmation_recorded=True,
    )
    assert verified.states["get_quote"] is ToolState.DRIFTED


def test_server_and_account_mismatch_invalidate_the_whole_session():
    stored = profile()
    with pytest.raises(ToolResultUnreadable, match="whole session"):
        verify_session_profile(
            stored,
            Session([]),
            server="https://community.invalid/mcp",
            account_id=ACCOUNT,
            confirmation_recorded=True,
        )
    with pytest.raises(ToolResultUnreadable, match="whole session"):
        verify_session_profile(
            stored,
            Session([]),
            server=SERVER,
            account_id="another-account",
            confirmation_recorded=True,
        )


def test_no_confirmation_ledger_note_means_no_tool_is_callable():
    quote = tool("get_quote")
    stored = profile(
        confirmed(
            quote,
            Category.READ_QUOTE,
            arguments={},
            result={"price": "price", "asof": "at"},
        )
    )
    verified = verify_session_profile(
        stored,
        Session([quote]),
        server=SERVER,
        account_id=ACCOUNT,
        confirmation_recorded=False,
    )
    with pytest.raises(CapabilityUnmapped, match="profile_confirmed"):
        verified.mapping_for(Category.READ_QUOTE, require_proof=False)


def test_an_input_schema_change_to_order_place_blocks_that_exact_tool():
    place = tool(
        "place_order",
        input_schema={
            "type": "object",
            "properties": {
                "account": {"type": "string"},
                "symbol": {"type": "string"},
                "side": {"type": "string"},
                "quantity": {"type": "string"},
            },
            "required": ["account", "symbol", "side", "quantity"],
        },
    )
    mapping = confirmed(
        place,
        Category.ORDER_PLACE,
        arguments={
            "account": "{account_id}",
            "symbol": "{symbol}",
            "side": "{side}",
            "quantity": "{qty}",
        },
        result={"order_id": "id", "quantity": "qty", "price": "price", "filled_at": "at"},
    )
    changed = place.model_copy(
        update={
            "input_schema": {
                **place.input_schema,
                "properties": {
                    **place.input_schema["properties"],
                    "time_in_force": {"type": "string"},
                },
            }
        }
    )
    verified = verify_session_profile(
        profile(mapping),
        Session([changed]),
        server=SERVER,
        account_id=ACCOUNT,
        confirmation_recorded=True,
    )
    assert verified.states["place_order"] is ToolState.DRIFTED
    with pytest.raises(CapabilityUnmapped, match="drifted"):
        verified.mapping_for(Category.ORDER_PLACE, require_proof=False)


def test_mutating_refresh_also_invalidates_a_changed_dependency():
    place = tool(
        "place_order",
        input_schema={
            "type": "object",
            "properties": {
                "account": {"type": "string"},
                "symbol": {"type": "string"},
                "side": {"type": "string"},
                "quantity": {"type": "string"},
            },
            "required": ["account", "symbol", "side", "quantity"],
        },
    )
    quote = tool("get_quote", description="Price in dollars.")
    stored = profile(
        confirmed(
            place,
            Category.ORDER_PLACE,
            arguments={
                "account": "{account_id}",
                "symbol": "{symbol}",
                "side": "{side}",
                "quantity": "{qty}",
            },
            result={
                "order_id": "id",
                "quantity": "qty",
                "price": "price",
                "filled_at": "at",
            },
        ),
        confirmed(
            quote,
            Category.READ_QUOTE,
            arguments={},
            result={"price": "price", "asof": "at"},
        ),
    )
    session = Session([place, quote])
    verified = verify_session_profile(
        stored,
        session,
        server=SERVER,
        account_id=ACCOUNT,
        confirmation_recorded=True,
    )
    session.tools = (place, quote.model_copy(update={"description": "Price in cents."}))
    verified.refresh_tool("place_order")
    with pytest.raises(CapabilityUnmapped, match="drifted"):
        verified.mapping_for(Category.READ_QUOTE, require_proof=False)


def test_prototype_toolmap_migrates_once_to_a_private_drifted_profile(tmp_path):
    old = tmp_path / "broker" / "toolmap.json"
    write_private_file(
        old,
        json.dumps(
            {
                "account_id": ACCOUNT,
                "server_name": "prototype",
                "discovered_at": AT.isoformat(),
                "capabilities": {
                    "quote": {
                        "tool": "get_quote",
                        "arguments": {"symbol": "{symbol}"},
                        "result": {"price": "price", "asof": "at"},
                    }
                },
            }
        ),
    )
    migrated = load_profile(tmp_path)
    assert migrated is not None
    assert migrated.state is ProfileState.DRIFTED
    assert "requires confirmation" in migrated.drift[0].changes
    assert stat.S_IMODE(profile_path(tmp_path).stat().st_mode) == 0o600

    old.write_text("not json anymore", encoding="utf-8")
    assert load_profile(tmp_path) == migrated
