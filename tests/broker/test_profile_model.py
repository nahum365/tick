"""The user's provider proposes; warnings and people never override denial."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from tick.agents import StructuredReply
from tick.broker import DiscoveredTool
from tick.broker.profile import (
    CATEGORIZER_VERSION,
    Category,
    ProposalReply,
    ProposalReplyTool,
    contract_for,
    edit_proposal,
    propose_profile,
)
from tick.broker.profile_model import MODEL_PROPOSAL_TOOL, ModelCategorizer, check_proposal

FIXTURES = Path(__file__).with_name("fixtures")
AT = datetime(2026, 9, 4, 8, 0, tzinfo=UTC)
SERVER = "https://agent.robinhood.com/mcp/trading"


def inventory() -> list[DiscoveredTool]:
    raw = json.loads((FIXTURES / "robinhood_tools_list_2026-09-04.json").read_text())
    return [DiscoveredTool.model_validate(item) for item in raw["tools"]]


class RecordedModel:
    def __init__(self, reply: dict | None = None) -> None:
        self.requests = []
        self.reply = reply or json.loads(
            (FIXTURES / "robinhood_profile_reply_2026-09-04.json").read_text()
        )

    def propose(self, request):
        self.requests.append(request)
        return StructuredReply(
            model=self.reply["model"],
            tool_name=self.reply["tool_name"],
            payload=self.reply["payload"],
        )


def test_recorded_inventory_is_proposed_in_one_structured_model_request():
    tools = inventory()
    client = RecordedModel()
    proposal = propose_profile(
        tools,
        server=SERVER,
        account_id=None,
        proposed_at=AT,
        categorizer=ModelCategorizer(client=client, model="requested-model"),
    )

    assert len(client.requests) == 1
    request = client.requests[0]
    assert request.tools[0]["name"] == MODEL_PROPOSAL_TOOL
    prompt = request.composed_text()
    assert "idempotency_key" in prompt
    assert "annotations_untrusted" in prompt
    assert "Options, crypto, futures" in prompt
    assert proposal.account_id is None
    assert proposal.categorizer_version == "model-v1:provider-model-fixture"
    assert all(not row.warnings for row in proposal.tools.values())
    expected = {
        "get_accounts": Category.READ_ACCOUNTS,
        "get_equity_positions": Category.READ_POSITIONS,
        "get_portfolio": Category.READ_BALANCES,
        "get_equity_orders": Category.READ_ORDERS,
        "get_equity_quotes": Category.READ_QUOTE,
        "get_equity_historicals": Category.READ_HISTORY,
        "review_equity_order": Category.ORDER_PREFLIGHT,
        "place_equity_order": Category.ORDER_PLACE,
        "cancel_equity_order": Category.ORDER_CANCEL,
    }
    assert {
        name: row.category for name, row in proposal.tools.items() if row.category is not None
    } == expected


def simple_tool(
    name: str = "inspect",
    *,
    annotations=None,
) -> DiscoveredTool:
    return DiscoveredTool(
        name=name,
        title=None,
        description="Inspect one value.",
        input_schema={
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
        output_schema={
            "type": "object",
            "properties": {"data": {"type": "object", "properties": {"price": {"type": "string"}}}},
        },
        annotations=annotations,
        execution=None,
    )


def test_light_validation_adds_every_warning_without_changing_the_rows():
    first = simple_tool("inspect_one", annotations={"destructiveHint": True})
    second = simple_tool("inspect_two")
    reply = ProposalReply(
        model="reported-model",
        tools=(
            ProposalReplyTool(
                name=first.name,
                category=Category.READ_QUOTE,
                arguments={"unknown": "{query}"},
                result={"price": "data.missing", "asof": "data.at"},
                reason="This looks like a quote read.",
            ),
            ProposalReplyTool(
                name=second.name,
                category=Category.READ_QUOTE,
                arguments={},
                result={"price": "data.price", "asof": "data.at"},
                reason="This also looks like a quote read.",
            ),
        ),
    )
    checked = check_proposal(reply, [contract_for(first), contract_for(second)])
    one, two = checked.tools
    text = " ".join(one.warnings)
    assert "argument 'unknown' is not declared" in text
    assert "required input 'symbol' has no proposed binding" in text
    assert "placeholders ['query'] are outside the grammar" in text
    assert "does not resolve in the declared output schema" in text
    assert "also claimed" in text
    assert "destructive or not read-only" in text
    assert "also claimed" in " ".join(two.warnings)
    assert one.category is reply.tools[0].category
    assert one.arguments == reply.tools[0].arguments


def test_denial_registry_overrides_a_model_read_mapping():
    transfer = simple_tool("transfer_assets")
    raw = {
        "model": "reported-model",
        "tool_name": MODEL_PROPOSAL_TOOL,
        "payload": {
            "tools": [
                {
                    "name": transfer.name,
                    "category": "read.quote",
                    "arguments": {"symbol": "{symbol}"},
                    "result": {"price": "data.price", "asof": "data.at"},
                    "reason": "This looks like a read.",
                }
            ]
        },
    }
    proposal = propose_profile(
        [transfer],
        server=SERVER,
        account_id=None,
        proposed_at=AT,
        categorizer=ModelCategorizer(client=RecordedModel(raw), model="requested-model"),
    )
    row = proposal.tools[transfer.name]
    assert row.category is Category.DENIED_TRANSFERS
    assert row.arguments == {}
    assert row.original.category is Category.READ_QUOTE


def test_no_provider_uses_named_deterministic_fallback_and_says_so():
    proposal = propose_profile(
        [simple_tool("unknown_contract")],
        server=SERVER,
        account_id=None,
        proposed_at=AT,
        categorizer=None,
    )
    row = proposal.tools["unknown_contract"]
    assert proposal.categorizer_version == CATEGORIZER_VERSION
    assert row.category is None
    assert "connect a provider" in row.reason


def test_person_edit_records_old_to_new_and_retains_the_model_original():
    discovered = simple_tool("inspect_quote")
    raw = {
        "model": "reported-model",
        "tool_name": MODEL_PROPOSAL_TOOL,
        "payload": {
            "tools": [
                {
                    "name": discovered.name,
                    "category": None,
                    "arguments": {},
                    "result": {},
                    "reason": "I did not map this tool.",
                }
            ]
        },
    }
    proposal = propose_profile(
        [discovered],
        server=SERVER,
        account_id=None,
        proposed_at=AT,
        categorizer=ModelCategorizer(client=RecordedModel(raw), model="requested-model"),
    )
    edited = edit_proposal(
        proposal,
        discovered.name,
        {
            "category": "read.quote",
            "arguments": {"symbol": "{symbol}"},
            "result": {"price": "data.price", "asof": "data.observed_at"},
        },
        who="api",
        at=AT,
    ).tools[discovered.name]
    assert edited.original.category is None
    assert edited.category is Category.READ_QUOTE
    assert [(change.field, change.old, change.new) for change in edited.edits] == [
        ("category", None, Category.READ_QUOTE),
        ("arguments", {}, {"symbol": "{symbol}"}),
        ("result", {}, {"price": "data.price", "asof": "data.observed_at"}),
    ]
    assert all(change.who == "api" and change.at == AT for change in edited.edits)


def test_the_reply_schema_is_strict_and_the_pair_form_folds_back():
    """Live 2026-09-04: codex refused the schema ("'oneOf' is not permitted")."""
    import json

    from tick.broker.profile_model import _normalize_payload, _reply_schema

    text = json.dumps(_reply_schema()["input_schema"])
    assert "oneOf" not in text

    def strict(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False
                assert set(node.get("required", [])) == set(node.get("properties", {}))
            for value in node.values():
                strict(value)
        elif isinstance(node, list):
            for value in node:
                strict(value)

    strict(_reply_schema()["input_schema"])

    folded = _normalize_payload(
        {
            "tools": [
                {
                    "name": "get_equity_quotes",
                    "category": "read.quote",
                    "bindings": [{"input": "symbols", "value": ["{symbol}"]}],
                    "paths": [{"role": "price", "path": "data.results.0.quote.last_trade_price"}],
                    "reason": "quotes",
                }
            ]
        }
    )
    assert folded["tools"][0]["arguments"] == {"symbols": ["{symbol}"]}
    assert folded["tools"][0]["result"] == {"price": "data.results.0.quote.last_trade_price"}
    assert "bindings" not in folded["tools"][0]
    recorded = _normalize_payload({"tools": [{"name": "x", "arguments": {"a": 1}, "result": {}}]})
    assert recorded["tools"][0]["arguments"] == {"a": 1}


def test_root_anchored_row_paths_are_rewritten_relative_to_the_items_row():
    """Live 2026-09-04: the model wrote data.positions.0.symbol for a per-row role."""
    from tick.broker.profile_model import _normalize_payload

    folded = _normalize_payload(
        {
            "tools": [
                {
                    "name": "get_equity_positions",
                    "category": "read.positions",
                    "arguments": {"account_number": "{account_id}"},
                    "result": {
                        "items": "data.positions",
                        "account": "{account_id}",
                        "symbol": "data.positions.0.symbol",
                        "quantity": "data.positions.0.quantity",
                        "average_cost": "average_buy_price",
                    },
                    "reason": "positions",
                }
            ]
        }
    )
    assert folded["tools"][0]["result"] == {
        "items": "data.positions",
        "account": "{account_id}",
        "symbol": "symbol",
        "quantity": "quantity",
        "average_cost": "average_buy_price",
    }
