"""The tool map: what Tick needs, bound to what the broker turned out to have.

CLAUDE.md invariant 7 says broker tool schemas are **discovered, never
assumed**, and this file is where that discovery becomes usable. Nothing here
knows a Robinhood tool name. A `ToolMap` is a document on the user's machine
that says, for each capability the runtime needs — a quote, positions, cash, an
order placed, an order cancelled, orders listed — which discovered tool serves
it, with what arguments, and where in its answer each number lives.

Three rules make this an honest mapping rather than a dressed-up guess:

- **Proposal and adoption were different acts.** `propose` reads `tools/list`
  and suggests a mapping by name and schema heuristics. This prototype format
  is now migration input only; the profile ceremony is `broker propose`,
  `confirm`, then `prove`.
- **An unmapped capability refuses.** `ToolMap.mapping_for` raises rather than
  falling back to a likely-looking tool. A runtime that guessed which tool
  places an order would be guessing with the user's money.
- **A number that is not there is `Unavailable`, never zero** (invariant 5).
  The readers below return `Unavailable` for a missing path, an unparsable
  value, or a JSON float — a binary float is an approximation of money, and
  `spec/base.py` has refused those since slice 01.

Read scoping lives here too, because this is where the account id is recorded.
The map carries the ONE Agentic account Tick is configured for; the adapter
sends it as an argument wherever a tool takes one, and filters every returned
row that belongs to another account. Robinhood's grant reads all of a user's
accounts (see `auth/disclosure.py`); this file is where Tick declines to.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import AwareDatetime, BaseModel, ConfigDict, ValidationError, model_validator

from tick.engine import Unavailable
from tick.records import write_private_file

from .errors import CapabilityUnmapped, ToolResultUnreadable

__all__ = [
    "ARGUMENT_PLACEHOLDERS",
    "Capability",
    "CapabilityMapping",
    "DiscoveredTool",
    "MAP_FILE",
    "Proposal",
    "REQUIRED_RESULT_ROLES",
    "ToolMap",
    "decimal_at",
    "dig",
    "load_tool_map",
    "propose",
    "rows_at",
    "save_tool_map",
    "text_at",
    "timestamp_at",
    "toolmap_path",
    "whole_at",
]

#: Where the adopted map lives, beside the grant it is used with.
MAP_FILE = "toolmap.json"


class Capability(StrEnum):
    """What the runtime needs a brokerage to do. Not what any broker calls it."""

    QUOTE = "quote"
    POSITIONS = "positions"
    ACCOUNT = "account"
    PLACE_ORDER = "place_order"
    CANCEL_ORDER = "cancel_order"
    LIST_ORDERS = "list_orders"


#: What each capability's arguments may be filled from. A template naming
#: anything else is refused: the map cannot ask Tick for a value it has no
#: business supplying to a broker.
ARGUMENT_PLACEHOLDERS: Mapping[Capability, frozenset[str]] = {
    Capability.QUOTE: frozenset({"symbol"}),
    Capability.POSITIONS: frozenset({"account_id"}),
    Capability.ACCOUNT: frozenset({"account_id"}),
    Capability.PLACE_ORDER: frozenset({"account_id", "symbol", "side", "qty"}),
    Capability.CANCEL_ORDER: frozenset({"order_id"}),
    Capability.LIST_ORDERS: frozenset({"account_id"}),
}

#: Every placeholder an order must carry. An order missing one of these is not
#: under-specified, it is a different order.
REQUIRED_PLACEHOLDERS: Mapping[Capability, frozenset[str]] = {
    Capability.PLACE_ORDER: frozenset({"account_id", "symbol", "side", "qty"}),
    Capability.CANCEL_ORDER: frozenset({"order_id"}),
}

#: Where each capability's numbers live in the answer. `items` is a path to a
#: list; the roles after it are read relative to each element of that list.
#: `account` is required on every list-shaped read — without it the adapter
#: cannot tell whose row it is holding, and read scoping would be a claim
#: rather than a filter.
REQUIRED_RESULT_ROLES: Mapping[Capability, tuple[str, ...]] = {
    Capability.QUOTE: ("price", "asof"),
    Capability.POSITIONS: ("items", "account", "symbol", "quantity", "average_cost"),
    Capability.ACCOUNT: ("items", "account", "cash"),
    Capability.PLACE_ORDER: ("order_id", "quantity", "price", "filled_at"),
    Capability.CANCEL_ORDER: ("order_id", "cancelled_at"),
    Capability.LIST_ORDERS: ("items", "account", "order_id"),
}

_PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")


class MapModel(BaseModel):
    """Frozen and closed, like every other document Tick keeps."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class DiscoveredTool(MapModel):
    """One tool a broker declared, in Tick's own vocabulary.

    `tools/list` answers in the MCP SDK's types; this is what the rest of the
    adapter reads instead. The translation is one line in `mcp_session.py`, and
    it buys something worth a line: everything downstream of discovery — the
    mapping, the proposal, the CLI's listing — is a plain Tick value that can be
    built in a test without an MCP server, and no file but the transport seam
    imports the SDK.
    """

    name: str
    title: str | None = None
    description: str | None
    input_schema: Mapping[str, Any]
    output_schema: Mapping[str, Any] | None
    annotations: Mapping[str, Any] | None = None
    execution: Mapping[str, Any] | None = None

    @model_validator(mode="after")
    def _check(self) -> DiscoveredTool:
        if not self.name.strip():
            raise ValueError("a discovered tool must have a name")
        return self

    def required_inputs(self) -> tuple[str, ...]:
        return tuple(self.input_schema.get("required") or ())

    def input_properties(self) -> tuple[str, ...]:
        return tuple((self.input_schema.get("properties") or {}).keys())

    def output_properties(self) -> tuple[str, ...]:
        return tuple(((self.output_schema or {}).get("properties") or {}).keys())


class CapabilityMapping(MapModel):
    """One capability bound to one discovered tool.

    `arguments` maps the tool's own argument names to templates Tick fills
    (`{"account_id": "{account_id}", "symbol": "{symbol}"}`). `result` maps
    Tick's roles to dotted paths into the tool's answer.
    """

    tool: str
    arguments: Mapping[str, str]
    result: Mapping[str, str]

    @model_validator(mode="after")
    def _check(self) -> CapabilityMapping:
        if not self.tool.strip():
            raise ValueError("a mapping must name the tool it binds")
        for role, path in self.result.items():
            if not path.strip():
                raise ValueError(f"result role {role!r} maps to an empty path")
        return self

    def placeholders(self) -> frozenset[str]:
        """Every value this mapping asks Tick to supply."""
        return frozenset(
            name for template in self.arguments.values() for name in _PLACEHOLDER.findall(template)
        )

    def render(self, values: Mapping[str, Any]) -> dict[str, Any]:
        """The concrete arguments for one call, or a refusal naming what is missing."""
        rendered: dict[str, Any] = {}
        for name, template in self.arguments.items():
            wanted = _PLACEHOLDER.findall(template)
            missing = [key for key in wanted if key not in values]
            if missing:
                raise CapabilityUnmapped(
                    f"the mapping for {self.tool} asks for {missing} to fill its "
                    f"{name!r} argument, and this call has no such value."
                )
            rendered[name] = _PLACEHOLDER.sub(lambda m: str(values[m.group(1)]), template)
        return rendered


class ToolMap(MapModel):
    """The adopted mapping: one broker, one account, one set of bindings."""

    account_id: str
    server_name: str | None
    discovered_at: AwareDatetime
    capabilities: Mapping[Capability, CapabilityMapping]

    @model_validator(mode="after")
    def _check(self) -> ToolMap:
        if not self.account_id.strip():
            raise ValueError(
                "a tool map must name the Agentic account it is scoped to; every read "
                "is issued for that account and every other account is filtered out"
            )
        for capability, mapping in self.capabilities.items():
            permitted = ARGUMENT_PLACEHOLDERS[capability]
            used = mapping.placeholders()
            if not used <= permitted:
                raise ValueError(
                    f"{capability.value} asks Tick for {sorted(used - permitted)}, which "
                    f"it has no business supplying to that tool"
                )
            required = REQUIRED_PLACEHOLDERS.get(capability, frozenset())
            if not required <= used:
                raise ValueError(
                    f"{capability.value} does not carry {sorted(required - used)}; an "
                    f"order missing one of those is a different order"
                )
            missing = [
                role for role in REQUIRED_RESULT_ROLES[capability] if role not in mapping.result
            ]
            if missing:
                raise ValueError(
                    f"{capability.value} maps {mapping.tool} but says nothing about "
                    f"{missing} in its answer; Tick reads numbers by mapped path and "
                    f"never by guesswork"
                )
        return self

    def mapping_for(self, capability: Capability) -> CapabilityMapping:
        """The mapping for `capability`, or a refusal that says how to fix it."""
        mapping = self.capabilities.get(capability)
        if mapping is None:
            raise CapabilityUnmapped(
                f"no discovered tool is mapped to {capability.value}. Tick refuses the "
                f"capability rather than guessing which tool means it: run "
                f"`tick broker tools` to see what the broker declares, then "
                f"`tick broker propose`, then confirm that exact tool."
            )
        return mapping

    def mapped(self) -> tuple[Capability, ...]:
        return tuple(sorted(self.capabilities, key=lambda capability: capability.value))

    def unmapped(self) -> tuple[Capability, ...]:
        return tuple(
            c for c in sorted(Capability, key=lambda c: c.value) if c not in self.capabilities
        )


class Proposal(MapModel):
    """The prototype proposal retained only for one-time map migration.

    `notes` carries what could not be checked — most often that a tool declares
    no output schema, so the proposed result paths are a convention Tick will
    only find out about when it calls the tool. They are shown to the user
    because that is the difference between adopting a mapping and inheriting a
    guess.
    """

    account_id: str
    server_name: str | None
    discovered_at: AwareDatetime
    capabilities: Mapping[Capability, CapabilityMapping]
    notes: Mapping[Capability, str]
    unmapped: Mapping[Capability, str]

    def to_tool_map(self) -> ToolMap:
        return ToolMap(
            account_id=self.account_id,
            server_name=self.server_name,
            discovered_at=self.discovered_at,
            capabilities=dict(self.capabilities),
        )


# ----------------------------------------------------------------------
# Reading a mapped answer: paths in, Decimal or Unavailable out
# ----------------------------------------------------------------------


def dig(payload: Any, path: str) -> Any:
    """Follow a dotted path into a decoded tool answer; `None` where it stops.

    Integer segments index a list, so `orders.0.id` is expressible. A path that
    runs off the end returns `None` rather than raising: a missing field is a
    number that is not there, which is `Unavailable`'s business and not an
    exception's.
    """
    current = payload
    for segment in path.split("."):
        if isinstance(current, Mapping):
            current = current.get(segment)
        elif isinstance(current, Sequence) and not isinstance(current, str | bytes):
            if not segment.isdigit() or int(segment) >= len(current):
                return None
            current = current[int(segment)]
        else:
            return None
        if current is None:
            return None
    return current


def decimal_at(payload: Any, path: str, what: str) -> Decimal | Unavailable:
    """The exact decimal at `path`, or why there is not one.

    A JSON **float** is refused rather than converted. `0.1` in JSON is not the
    number 0.1, and money that has been through a binary float is money whose
    last digits Tick made up — invariant 5's exact case.
    """
    value = dig(payload, path)
    if value is None:
        return Unavailable(what=what, reason=f"the broker's answer carries no {path!r}")
    if isinstance(value, bool):
        return Unavailable(what=what, reason=f"{path!r} is a true/false, not a number")
    if isinstance(value, float):
        return Unavailable(
            what=what,
            reason=(
                f"{path!r} arrived as a JSON float ({value!r}), which is a binary "
                f"approximation. Tick reads money as an exact decimal string or a whole "
                f"number and will not round one into a record"
            ),
        )
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, str):
        try:
            return Decimal(value.strip())
        except InvalidOperation:
            return Unavailable(what=what, reason=f"{path!r} is {value!r}, which is not a number")
    return Unavailable(what=what, reason=f"{path!r} is a {type(value).__name__}, not a number")


def whole_at(payload: Any, path: str, what: str) -> int | Unavailable:
    """A whole share count at `path`. A fractional quantity is not a whole share."""
    value = decimal_at(payload, path, what)
    if isinstance(value, Unavailable):
        return value
    if value != value.to_integral_value():
        return Unavailable(
            what=what,
            reason=f"{path!r} is {value}, and Tick trades whole shares only",
        )
    return int(value)


def text_at(payload: Any, path: str, what: str) -> str | Unavailable:
    """A non-empty string at `path`."""
    value = dig(payload, path)
    if not isinstance(value, str) or not value.strip():
        return Unavailable(what=what, reason=f"the broker's answer carries no usable {path!r}")
    return value.strip()


def timestamp_at(payload: Any, path: str, what: str) -> datetime | Unavailable:
    """A timezone-aware moment at `path`.

    A naive timestamp is refused. Market logic runs on Eastern sessions, and a
    moment with no offset is a time in an unstated zone — dating a price with
    one would be inventing the fact that is most load-bearing about it.
    """
    value = text_at(payload, path, what)
    if isinstance(value, Unavailable):
        return value
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return Unavailable(what=what, reason=f"{path!r} is {value!r}, which is not an ISO moment")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return Unavailable(
            what=what,
            reason=f"{path!r} is {value!r}, a moment with no timezone; Tick will not assume one",
        )
    return parsed


def rows_at(payload: Any, mapping: CapabilityMapping, what: str) -> list[Any] | Unavailable:
    """The list a list-shaped capability maps to."""
    value = dig(payload, mapping.result["items"])
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return Unavailable(
            what=what,
            reason=(
                f"the broker's {mapping.tool} answer carries no list at {mapping.result['items']!r}"
            ),
        )
    return list(value)


# ----------------------------------------------------------------------
# Proposing a mapping from what was discovered
# ----------------------------------------------------------------------

#: Words that suggest a tool serves a capability, most specific first.
_NAME_HINTS: Mapping[Capability, tuple[str, ...]] = {
    Capability.QUOTE: ("quote", "price", "last_trade"),
    Capability.POSITIONS: ("position", "holding"),
    Capability.ACCOUNT: ("account", "balance", "cash"),
    Capability.PLACE_ORDER: ("place_order", "create_order", "submit_order", "buy", "order"),
    Capability.CANCEL_ORDER: ("cancel",),
    Capability.LIST_ORDERS: ("list_orders", "get_orders", "orders"),
}

#: Argument names Tick knows how to fill, and what it fills them with.
_ARGUMENT_HINTS: Mapping[str, tuple[str, ...]] = {
    "symbol": ("symbol", "ticker", "instrument"),
    "account_id": ("account_id", "account", "account_number", "accountid"),
    "qty": ("quantity", "qty", "shares", "amount"),
    "side": ("side", "action", "direction"),
    "order_id": ("order_id", "orderid", "id"),
}

#: The result paths a proposal offers when the tool declares no output schema.
#: A convention, shown to the user as one — never adopted behind their back.
_RESULT_HINTS: Mapping[Capability, Mapping[str, tuple[str, ...]]] = {
    Capability.QUOTE: {
        "price": ("last_price", "price", "last", "last_trade_price"),
        "asof": ("quoted_at", "asof", "as_of", "timestamp", "updated_at"),
    },
    Capability.POSITIONS: {
        "items": ("positions", "holdings", "results"),
        "account": ("account", "account_id", "account_number"),
        "symbol": ("symbol", "ticker", "instrument"),
        "quantity": ("quantity", "qty", "shares"),
        "average_cost": ("average_cost", "avg_cost", "average_price", "cost_basis_per_share"),
    },
    Capability.ACCOUNT: {
        "items": ("accounts", "results"),
        "account": ("account_id", "account", "account_number", "id"),
        "cash": ("cash", "buying_power", "cash_balance", "available_cash"),
    },
    Capability.PLACE_ORDER: {
        "order_id": ("order_id", "id"),
        "quantity": ("filled_quantity", "quantity", "qty", "shares"),
        "price": ("filled_price", "average_price", "price"),
        "filled_at": ("filled_at", "executed_at", "timestamp", "updated_at"),
    },
    Capability.CANCEL_ORDER: {
        "order_id": ("order_id", "id"),
        "cancelled_at": ("cancelled_at", "canceled_at", "updated_at", "timestamp"),
    },
    Capability.LIST_ORDERS: {
        "items": ("orders", "results"),
        "account": ("account", "account_id", "account_number"),
        "order_id": ("order_id", "id"),
    },
}


def propose(
    tools: Sequence[DiscoveredTool],
    *,
    account_id: str,
    server_name: str | None,
    discovered_at: datetime,
) -> Proposal:
    """Suggest a mapping from discovered tools. Writes nothing, decides nothing.

    Every argument is required, `account_id` most of all: a proposal that
    invented the account to scope reads to would defeat the point of scoping
    them.
    """
    if not account_id.strip():
        raise ValueError("propose() needs the Agentic account id every read will be scoped to")
    mappings: dict[Capability, CapabilityMapping] = {}
    notes: dict[Capability, str] = {}
    unmapped: dict[Capability, str] = {}
    for capability in Capability:
        tool = _best_tool(tools, capability)
        if tool is None:
            unmapped[capability] = (
                f"no discovered tool's name suggests {capability.value} "
                f"({', '.join(_NAME_HINTS[capability])}). Tick leaves it unmapped, and "
                f"the capability refuses until you map it."
            )
            continue
        arguments = _propose_arguments(tool, capability)
        if isinstance(arguments, str):
            unmapped[capability] = arguments
            continue
        result, note = _propose_result(tool, capability)
        mappings[capability] = CapabilityMapping(tool=tool.name, arguments=arguments, result=result)
        if note is not None:
            notes[capability] = note
    return Proposal(
        account_id=account_id,
        server_name=server_name,
        discovered_at=discovered_at,
        capabilities=mappings,
        notes=notes,
        unmapped=unmapped,
    )


def _best_tool(tools: Sequence[DiscoveredTool], capability: Capability) -> DiscoveredTool | None:
    """The first tool whose name carries the most specific hint for `capability`."""
    for hint in _NAME_HINTS[capability]:
        for tool in tools:
            name = tool.name.lower()
            if hint in name and not _claimed_by_another(name, capability):
                return tool
    return None


def _claimed_by_another(name: str, capability: Capability) -> bool:
    """Keep `cancel_order` from being read as `place_order`, and so on."""
    if capability is Capability.PLACE_ORDER:
        return "cancel" in name or "list" in name or name.startswith("get_")
    if capability is Capability.LIST_ORDERS:
        return "cancel" in name or "place" in name or "create" in name
    return False


def _propose_arguments(tool: DiscoveredTool, capability: Capability) -> dict[str, str] | str:
    """Fill the tool's required inputs from Tick's vocabulary, or say why not."""
    properties = tool.input_properties()
    required = tool.required_inputs()
    permitted = ARGUMENT_PLACEHOLDERS[capability]
    arguments: dict[str, str] = {}
    for name in properties:
        placeholder = _placeholder_for(name)
        if placeholder is None or placeholder not in permitted:
            if name in required:
                return (
                    f"{tool.name} requires an input {name!r} that Tick has no value for. "
                    f"Mapping it would mean inventing an argument to send a broker, so "
                    f"{capability.value} is left unmapped."
                )
            continue
        arguments[name] = "{" + placeholder + "}"
    missing = [name for name in required if name not in arguments]
    if missing:
        return (
            f"{tool.name} requires {missing}, which Tick cannot fill. "
            f"{capability.value} is left unmapped rather than called with a guess."
        )
    return arguments


def _placeholder_for(argument_name: str) -> str | None:
    lowered = argument_name.lower()
    for placeholder, hints in _ARGUMENT_HINTS.items():
        if lowered in hints:
            return placeholder
    return None


def _propose_result(
    tool: DiscoveredTool, capability: Capability
) -> tuple[dict[str, str], str | None]:
    """Offer a result mapping, and say what about it could not be checked."""
    declared = tool.output_properties()
    hints = _RESULT_HINTS[capability]
    result = {role: candidates[0] for role, candidates in hints.items()}
    if declared:
        for role, candidates in hints.items():
            for candidate in candidates:
                if candidate in declared:
                    result[role] = candidate
                    break
    note = None
    if not declared:
        note = (
            f"{tool.name} declares no output schema, so the result paths below are a "
            f"convention rather than something Tick could check. If they are wrong the "
            f"capability reports the number as unavailable — it never invents one."
        )
    return result, note


# ----------------------------------------------------------------------
# The file on disk
# ----------------------------------------------------------------------


def toolmap_path(home: str | os.PathLike[str]) -> Path:
    """`<home>/robinhood/toolmap.json`, beside the grant it is used with."""
    return Path(home) / "robinhood" / MAP_FILE


def save_tool_map(home: str | os.PathLike[str], tool_map: ToolMap) -> Path:
    """Write the adopted map, private: it carries the user's account id."""
    return write_private_file(toolmap_path(home), tool_map.model_dump_json(indent=2))


def load_tool_map(home: str | os.PathLike[str]) -> ToolMap | None:
    """The adopted map, or `None` if nothing has been adopted on this machine."""
    path = toolmap_path(home)
    if not path.exists():
        return None
    try:
        return ToolMap.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ToolResultUnreadable(
            f"{path} is not a tool map Tick can read: {exc}. Delete it and run "
            f"`tick broker propose`, then confirm each callable tool; until then every "
            f"broker capability refuses."
        ) from exc
