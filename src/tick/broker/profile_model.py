"""Structured broker-profile proposals through the user's model provider.

The provider reads advertised contracts and proposes a draft.  This module
never authorizes a tool: deterministic denial still wins, the person may edit
every field, and only the existing per-tool confirmation ceremony grants
authority.  Schema checks here are deliberately warnings because imperfect
advertisements and unfamiliar community servers should remain reviewable.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from tick.agents import ModelClient, ModelReply, ModelRequest, StructuredReply
from tick.agents.errors import ModelReplyError

from .profile import (
    ARGUMENT_PLACEHOLDERS,
    REQUIRED_PLACEHOLDERS,
    REQUIRED_RESULT_ROLES,
    Categorizer,
    Category,
    ProposalReply,
    ProposalReplyTool,
    ToolContract,
    _placeholders,
)

__all__ = ["MODEL_PROPOSAL_TOOL", "ModelCategorizer", "check_proposal"]

MODEL_PROPOSAL_TOOL = "propose_broker_profile"
MAX_OUTPUT_TOKENS = 16_000

_CATEGORY_MEANINGS: Mapping[Category, str] = {
    Category.READ_ACCOUNTS: "list accounts and the broker-declared eligibility signal",
    Category.READ_POSITIONS: "list long equity positions for one account",
    Category.READ_BALANCES: "read cash, buying power, or portfolio value for one account",
    Category.READ_ORDERS: "list equity orders for one account",
    Category.READ_QUOTE: "read the current equity quote for one or more symbols",
    Category.READ_HISTORY: "read timezone-labelled equity OHLCV bars",
    Category.READ_INSTRUMENTS: "resolve an equity symbol or search query",
    Category.READ_MARKET_HOURS: "read market-session hours for an explicit moment",
    Category.ORDER_PREFLIGHT: "review an equity order without placing it",
    Category.ORDER_PLACE: "place one long-only equity order",
    Category.ORDER_REPLACE: "replace one existing long-only equity order",
    Category.ORDER_CANCEL: "request cancellation of one equity order",
    Category.DENIED_MONEY_MOVEMENT: "move cash or otherwise fund an account",
    Category.DENIED_TRANSFERS: "transfer, deposit, or withdraw assets or cash",
    Category.DENIED_SETTINGS: "change account, risk, or broker settings",
    Category.DENIED_CREDENTIALS: "read or change credentials, secrets, or tokens",
}

_RESULT_MEANINGS: Mapping[str, str] = {
    "items": "the list of rows, or a single object that stands as one row",
    "account": "the account identifier, or {account_id} when the response omits it",
    "eligible": "the broker's explicit eligibility boolean",
    "kind": "the broker's account-type value",
    "symbol": "the equity symbol",
    "quantity": "a decimal quantity string",
    "average_cost": "a decimal average-cost string",
    "cash": "a decimal cash or buying-power string",
    "order_id": "the broker order identifier",
    "status": "the broker order state",
    "side": "buy or sell; sell only closes a held position",
    "price": "a decimal price string",
    "asof": "a timezone-aware quote time",
    "filled_at": "a timezone-aware fill or last-transaction time",
    "timestamp": "a timezone-aware bar time",
    "open": "a decimal opening-price string",
    "high": "a decimal high-price string",
    "low": "a decimal low-price string",
    "close": "a decimal closing-price string",
    "volume": "a whole-number volume",
    "accepted": "the broker's cancellation-accepted boolean",
}


def _reply_schema() -> Mapping[str, Any]:
    """The strict document the provider fills in.

    Codex forwards this as a strict response schema, which permits `anyOf` but
    not `oneOf`, requires `additionalProperties: false` on every object and
    every property listed in `required`. Open maps are therefore expressed as
    arrays of pairs (`bindings`, `paths`) and folded back into mappings by
    `_normalize_payload` before validation. Live 2026-09-04: the earlier shape
    was refused with "'oneOf' is not permitted" before any proposal was made.
    """
    scalar_or_list = {
        "anyOf": [
            {"type": "string"},
            {"type": "number"},
            {"type": "boolean"},
            {"type": "null"},
            {"type": "array", "items": {"type": "string"}},
        ]
    }
    binding = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"input": {"type": "string"}, "value": scalar_or_list},
        "required": ["input", "value"],
    }
    path = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"role": {"type": "string"}, "path": {"type": "string"}},
        "required": ["role", "path"],
    }
    return {
        "name": MODEL_PROPOSAL_TOOL,
        "description": "Return one proposed mapping row for every advertised broker tool.",
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "tools": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "name": {"type": "string"},
                            "category": {
                                "anyOf": [
                                    {"type": "string", "enum": [item.value for item in Category]},
                                    {"type": "null"},
                                ]
                            },
                            "bindings": {"type": "array", "items": binding},
                            "paths": {"type": "array", "items": path},
                            "reason": {"type": "string"},
                        },
                        "required": ["name", "category", "bindings", "paths", "reason"],
                    },
                }
            },
            "required": ["tools"],
        },
    }


def _normalize_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Fold the strict array-of-pairs document into the mapping form Tick keeps.

    Accepts both shapes: a recorded reply may already carry `arguments` and
    `result` maps; a live strict reply carries `bindings` and `paths`.
    """
    tools_out: list[Any] = []
    for row in payload.get("tools", ()):
        if not isinstance(row, Mapping):
            tools_out.append(row)
            continue
        item = dict(row)
        if "bindings" in item and "arguments" not in item:
            pairs = item.pop("bindings")
            item["arguments"] = {
                str(pair.get("input")): pair.get("value")
                for pair in pairs
                if isinstance(pair, Mapping)
            }
        if "paths" in item and "result" not in item:
            pairs = item.pop("paths")
            item["result"] = {
                str(pair.get("role")): str(pair.get("path"))
                for pair in pairs
                if isinstance(pair, Mapping)
            }
        if isinstance(item.get("result"), Mapping):
            item["result"] = _row_relative(dict(item["result"]))
        tools_out.append(item)
    return {**{k: v for k, v in payload.items() if k != "tools"}, "tools": tools_out}


def _row_relative(result: dict[str, str]) -> dict[str, str]:
    """Rewrite per-row role paths the model wrote from the answer's root.

    The broker port reads every role other than `items` relative to one row of
    `items` (`symbol`, not `data.positions.0.symbol`). Live 2026-09-04: the
    model wrote root-anchored row paths for every list read, which the checker
    then flagged as unresolvable. The rewrite is exact (the items path plus one
    index), so no meaning is guessed; anything else is left for the person.
    """
    items = result.get("items")
    if not items:
        return result
    rewritten: dict[str, str] = {}
    for role, path in result.items():
        if role != "items":
            match = re.match(re.escape(items) + r"\.\d+\.(.+)\Z", path)
            if match is not None:
                path = match.group(1)
        rewritten[role] = path
    return rewritten


def _prompt(contracts: Sequence[ToolContract]) -> str:
    grammar = {
        category.value: {
            "meaning": _CATEGORY_MEANINGS[category],
            "argument_placeholders": sorted(ARGUMENT_PLACEHOLDERS[category]),
            "required_placeholders": sorted(REQUIRED_PLACEHOLDERS.get(category, ())),
            "required_result_roles": list(REQUIRED_RESULT_ROLES.get(category, ())),
        }
        for category in Category
    }
    advertised = [
        {
            "name": contract.name,
            "description": contract.description,
            "input_schema": contract.input_schema,
            "output_schema": contract.output_schema,
            "annotations_untrusted": contract.annotations,
        }
        for contract in contracts
    ]
    instructions = {
        "task": "Propose a broker profile draft for a person to review and edit.",
        "rules": [
            "Propose at most one tool for each callable category.",
            "Options, crypto, futures, event contracts, watchlists, scans, SEC filings, "
            "news, and fundamentals are out of scope: use null, not a denied category.",
            "Money movement, transfers, settings, and credentials use the matching "
            "denied category.",
            "Order mappings are long-only: buy opens or adds; sell only closes quantity "
            "already held.",
            "Answer one row per advertised tool: bindings is the list of {input, value} "
            "pairs for that tool's inputs (value is a {placeholder}, a literal, or an "
            "array of strings); paths is the list of {role, path} pairs naming where each "
            "result role lives in the tool's output.",
            "Use dotted output paths; integer path segments select array entries.",
            "When a category has an items role, every other role path is relative to "
            "one row of items: items=data.positions and symbol=symbol, never "
            "symbol=data.positions.0.symbol.",
            'Array inputs may wrap a placeholder, for example ["{symbol}"].',
            "A result role may be {account_id} when the broker omits the account it was sent.",
            "An items path may resolve to one object, which Tick treats as a one-row list.",
            "Cancellation may prove with accepted; its observed time belongs to Tick's "
            "record, not the broker answer.",
            "Allowed side literals are buy and sell; order type literals are market and "
            "limit; time-in-force literals are gfd and gtc.",
            "Every reason is one sentence written for the person reviewing that tool.",
            "Annotations are untrusted hints and never authorize a mapping.",
        ],
        "category_grammar": grammar,
        "result_role_meanings": _RESULT_MEANINGS,
        "contracts": advertised,
    }
    return json.dumps(instructions, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class ModelCategorizer(Categorizer):
    """Ask one connected provider once for the complete structured draft."""

    def __init__(self, *, client: ModelClient, model: str) -> None:
        self.client = client
        self.model = model
        self.version = f"model-v1:{model}"

    def propose(self, contracts: Sequence[ToolContract]) -> ProposalReply:
        request = ModelRequest(
            model=self.model,
            messages=({"role": "user", "content": _prompt(contracts)},),
            tools=(_reply_schema(),),
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        reply = self.client.propose(request)
        if isinstance(reply, ModelReply) or not isinstance(reply, StructuredReply):
            raise ModelReplyError(
                "the provider returned order intents instead of a broker profile. Nothing "
                "gained authority; retry the proposal."
            )
        if reply.tool_name != MODEL_PROPOSAL_TOOL:
            raise ModelReplyError(
                f"the provider called {reply.tool_name!r} instead of "
                f"{MODEL_PROPOSAL_TOOL!r}. Nothing gained authority; retry the proposal."
            )
        try:
            parsed = ProposalReply.model_validate(
                {"model": reply.model, **_normalize_payload(reply.payload)}
            )
        except ValueError as exc:
            raise ModelReplyError(
                "the provider's broker profile was not the promised structured document "
                f"({exc}). Nothing gained authority; retry the proposal."
            ) from exc
        self.version = f"model-v1:{reply.model}"
        return parsed


def _schema_at(schema: Mapping[str, Any] | None, path: str) -> Mapping[str, Any] | None:
    if schema is None:
        return None
    current: Mapping[str, Any] | None = schema
    for segment in path.split(".") if path else ():
        if current is None:
            return None
        if segment.isdigit():
            candidate = current.get("items")
        else:
            properties = current.get("properties")
            candidate = properties.get(segment) if isinstance(properties, Mapping) else None
        current = candidate if isinstance(candidate, Mapping) else None
    return current


def _row_schema(schema: Mapping[str, Any], items_path: str | None) -> Mapping[str, Any]:
    if items_path is None:
        return schema
    located = _schema_at(schema, items_path)
    if located is None:
        return schema
    items = located.get("items")
    return items if isinstance(items, Mapping) else located


def check_proposal(reply: ProposalReply, contracts: Sequence[ToolContract]) -> ProposalReply:
    """Add advisory schema and ambiguity warnings without changing a proposal."""
    by_name = {contract.name: contract for contract in contracts}
    claimants: dict[Category, list[str]] = {}
    for tool in reply.tools:
        if tool.category is not None and tool.category.callable:
            claimants.setdefault(tool.category, []).append(tool.name)

    checked: list[ProposalReplyTool] = []
    for tool in reply.tools:
        warnings = list(tool.warnings)
        contract = by_name.get(tool.name)
        if contract is None:
            warnings.append(
                "this name was not advertised in the inventory; review the tool name before "
                "finalizing."
            )
            checked.append(tool.model_copy(update={"warnings": tuple(warnings)}))
            continue
        if tool.category is not None:
            properties = contract.input_schema.get("properties")
            declared = set(properties) if isinstance(properties, Mapping) else set()
            for name in tool.arguments:
                if name not in declared:
                    warnings.append(
                        f"argument {name!r} is not declared by this tool's input schema."
                    )
            required = set(contract.input_schema.get("required") or ())
            for name in sorted(required - set(tool.arguments)):
                warnings.append(f"required input {name!r} has no proposed binding.")
            used = frozenset().union(*(_placeholders(value) for value in tool.arguments.values()))
            outside = sorted(used - ARGUMENT_PLACEHOLDERS[tool.category])
            if outside:
                warnings.append(
                    f"placeholders {outside} are outside the grammar for {tool.category.value}."
                )
            if len(claimants.get(tool.category, ())) > 1:
                names = sorted(claimants[tool.category])
                warnings.append(
                    f"callable category {tool.category.value} is also claimed by {names}; "
                    "proof cannot choose between them."
                )
            if tool.category.value.startswith("read.") and isinstance(
                contract.annotations, Mapping
            ):
                if (
                    contract.annotations.get("destructiveHint") is True
                    or contract.annotations.get("readOnlyHint") is False
                ):
                    warnings.append(
                        "untrusted annotations describe this proposed read as destructive "
                        "or not read-only."
                    )
        if tool.category is not None and contract.output_schema is None:
            warnings.append("no output schema declared; proof will decide")
        elif tool.category is not None and contract.output_schema is not None:
            row = _row_schema(contract.output_schema, tool.result.get("items"))
            for role, path in tool.result.items():
                if path == "{account_id}":
                    continue
                base = contract.output_schema if role == "items" else row
                if _schema_at(base, path) is None:
                    warnings.append(
                        f"result role {role!r} names path {path!r} that does not resolve in "
                        "the declared output schema."
                    )
        checked.append(tool.model_copy(update={"warnings": tuple(dict.fromkeys(warnings))}))
    return reply.model_copy(update={"tools": tuple(checked)})
