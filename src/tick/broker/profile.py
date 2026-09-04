"""Broker profiles: confirmed contracts, per-tool drift, and positive routing.

A profile is approval evidence, not a cache of whatever an MCP server last
said.  Each callable tool pins its complete advertised contract and the
mapping the user approved.  On an open session Tick fetches the complete live
inventory and authorizes only exact per-tool matches.  A changed read is as
uncallable as a changed order tool; a new, unrelated tool stays unmapped and
does not disable unchanged tools.

Three hashes deliberately answer three different questions:

* ``shape_hash`` covers the exact name and complete input/output schemas.
* ``contract_hash`` adds mechanically normalized title, description,
  behavioral annotations, and execution semantics.  Descriptions are included
  because they are often the only declaration of units.  Annotations remain
  untrusted: they may veto a proposal, but never authorize one.
* ``inventory_hash`` covers the sorted set of names and contract hashes for
  audit and diffing.  It is never a profile-wide permission bit.

Canonicalization sorts object keys and tool names but preserves schema array
order.  It rejects duplicate names, non-finite numbers, and unresolved or
remote ``$ref`` values.  Response-envelope fields such as cursors, icons, and
``_meta`` never enter these models or hashes.

Residual threat-model limit: MCP has no atomic "call only if contract equals
X" precondition.  Tick promises to call only when the latest advertised
contract matches; it cannot prove that the server implements what it advertises.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import AwareDatetime, BaseModel, ConfigDict, ValidationError, model_validator

from tick.records import DataSource, Ledger, RecordKind, read, write_private_file

from .errors import CapabilityUnmapped, ToolResultUnreadable
from .toolmap import DiscoveredTool

__all__ = [
    "ARGUMENT_PLACEHOLDERS",
    "CANONICALIZER_VERSION",
    "CATEGORIZER_VERSION",
    "CATEGORY_REGISTRY_VERSION",
    "HOST_ALLOWLIST",
    "PROFILE_FORMAT_VERSION",
    "PROFILE_FILE",
    "Category",
    "Categorizer",
    "DriftDifference",
    "Profile",
    "ProfileProposal",
    "ProposalEdit",
    "ProposalReply",
    "ProposalReplyTool",
    "ProfileState",
    "ProfileTool",
    "ProofResult",
    "ProposedTool",
    "REQUIRED_RESULT_ROLES",
    "ToolContract",
    "ToolState",
    "ORDER_VALUES",
    "VerifiedSessionProfile",
    "canonical_json",
    "history_values",
    "categorize",
    "confirm_profile",
    "contract_for",
    "diff_profile",
    "edit_proposal",
    "has_confirmation_note",
    "inventory_hash",
    "load_profile",
    "load_proposal",
    "mapping_hash",
    "migrate_toolmap",
    "profile_ledger_path",
    "profile_path",
    "proposal_path",
    "propose_profile",
    "prove_profile",
    "prove_proposal",
    "save_profile",
    "save_proposal",
    "sanction_for",
    "verify_session_profile",
]

PROFILE_FILE = "profile.json"
PROPOSAL_FILE = "proposal.json"
PROFILE_FORMAT_VERSION = "1"
CANONICALIZER_VERSION = "1"
CATEGORY_REGISTRY_VERSION = "1"
CATEGORIZER_VERSION = "deterministic-v1"
HOST_ALLOWLIST = frozenset({"agent.robinhood.com"})
_HTTPS_SERVER = re.compile(r"\Ahttps\x3a//([^/:?#]+)(?::[0-9]+)?(?:[/?#]|\Z)", re.IGNORECASE)


class Category(StrEnum):
    """The closed capabilities Tick recognizes; unknown tools are unmapped."""

    READ_ACCOUNTS = "read.accounts"
    READ_POSITIONS = "read.positions"
    READ_BALANCES = "read.balances"
    READ_ORDERS = "read.orders"
    READ_QUOTE = "read.quote"
    READ_HISTORY = "read.history"
    READ_INSTRUMENTS = "read.instruments"
    READ_MARKET_HOURS = "read.market_hours"
    ORDER_PREFLIGHT = "order.preflight"
    ORDER_PLACE = "order.place"
    ORDER_REPLACE = "order.replace"
    ORDER_CANCEL = "order.cancel"
    DENIED_MONEY_MOVEMENT = "denied.money_movement"
    DENIED_TRANSFERS = "denied.transfers"
    DENIED_SETTINGS = "denied.settings"
    DENIED_CREDENTIALS = "denied.credentials"

    @property
    def callable(self) -> bool:
        return self.value.startswith(("read.", "order."))

    @property
    def denied(self) -> bool:
        return self.value.startswith("denied.")

    @property
    def mutating(self) -> bool:
        return self.value in {
            "order.place",
            "order.replace",
            "order.cancel",
        }


class ProfileState(StrEnum):
    """Persisted summary for status; authorization remains per tool."""

    CONFIRMED = "confirmed"
    DRIFTED = "drifted"
    NONE = "none"


class ToolState(StrEnum):
    """The authorization state of one advertised or previously mapped tool."""

    CONFIRMED = "confirmed"
    DRIFTED = "drifted"
    DENIED = "denied"
    UNMAPPED = "unmapped"


class ProfileModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _normalized_text(value: str | None) -> str | None:
    if value is None:
        return None
    return unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))


def _validate_json(value: Any, *, root: Any, where: str) -> None:
    if value is None or isinstance(value, str | bool | int):
        return
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise ValueError(f"{where} contains a non-finite number; the inventory is invalid")
        return
    if isinstance(value, Mapping):
        reference = value.get("$ref")
        if reference is not None:
            if not isinstance(reference, str) or not reference.startswith("#/"):
                raise ValueError(
                    f"{where} contains remote or unresolved $ref {reference!r}; "
                    "the complete schema closure must be advertised"
                )
            target = root
            for token in reference[2:].split("/"):
                key = token.replace("~1", "/").replace("~0", "~")
                if not isinstance(target, Mapping) or key not in target:
                    raise ValueError(
                        f"{where} contains unresolved $ref {reference!r}; "
                        "the complete schema closure must be advertised"
                    )
                target = target[key]
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{where} contains a non-string object key")
            _validate_json(item, root=root, where=f"{where}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for index, item in enumerate(value):
            _validate_json(item, root=root, where=f"{where}[{index}]")
        return
    raise ValueError(f"{where} contains {type(value).__name__}, which is not JSON")


def canonical_json(value: Any) -> str:
    """Canonical JSON for broker evidence, rejecting values JSON cannot pin."""
    _validate_json(value, root=value, where="contract")
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )


def _hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class ToolContract(ProfileModel):
    """The complete advertised content whose exact match authorizes one tool."""

    name: str
    title: str | None
    description: str | None
    input_schema: Mapping[str, Any]
    output_schema: Mapping[str, Any] | None
    annotations: Mapping[str, Any] | None
    execution: Mapping[str, Any] | None
    shape_hash: str
    contract_hash: str

    @model_validator(mode="after")
    def _check(self) -> ToolContract:
        if not self.name.strip():
            raise ValueError("a tool contract must name its exact tool")
        for label, schema in (
            ("input schema", self.input_schema),
            ("output schema", self.output_schema),
        ):
            if schema is not None:
                _validate_json(schema, root=schema, where=f"{self.name} {label}")
        expected_shape = _hash(
            {
                "name": self.name,
                "input_schema": self.input_schema,
                "output_schema": self.output_schema,
                "execution": self.execution,
            }
        )
        expected_contract = _hash(
            {
                "shape_hash": expected_shape,
                "title": _normalized_text(self.title),
                "description": _normalized_text(self.description),
                "annotations": self.annotations,
            }
        )
        if self.shape_hash != expected_shape or self.contract_hash != expected_contract:
            raise ValueError(
                f"{self.name} carries hashes that do not match its advertised contract"
            )
        return self


def contract_for(tool: DiscoveredTool) -> ToolContract:
    """Normalize one transport value into the exact contract Tick pins."""
    title = _normalized_text(getattr(tool, "title", None))
    description = _normalized_text(tool.description)
    annotations = getattr(tool, "annotations", None)
    execution = getattr(tool, "execution", None)
    if hasattr(annotations, "model_dump"):
        annotations = annotations.model_dump(mode="json", by_alias=True, exclude_none=True)
    if hasattr(execution, "model_dump"):
        execution = execution.model_dump(mode="json", by_alias=True, exclude_none=True)
    annotations = dict(annotations) if isinstance(annotations, Mapping) else None
    execution = dict(execution) if isinstance(execution, Mapping) else None
    shape = {
        "name": tool.name,
        "input_schema": dict(tool.input_schema),
        "output_schema": dict(tool.output_schema) if tool.output_schema is not None else None,
        "execution": execution,
    }
    shape_digest = _hash(shape)
    contract_digest = _hash(
        {
            "shape_hash": shape_digest,
            "title": title,
            "description": description,
            "annotations": annotations,
        }
    )
    return ToolContract(
        name=tool.name,
        title=title,
        description=description,
        input_schema=shape["input_schema"],
        output_schema=shape["output_schema"],
        annotations=annotations,
        execution=execution,
        shape_hash=shape_digest,
        contract_hash=contract_digest,
    )


def _contracts(tools: Sequence[DiscoveredTool]) -> tuple[ToolContract, ...]:
    contracts = tuple(sorted((contract_for(tool) for tool in tools), key=lambda item: item.name))
    names = [item.name for item in contracts]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ToolResultUnreadable(
            f"the broker advertised duplicate tool names {duplicates}; the whole session "
            "is invalid and no tool can be called. Fix the server listing and reconnect."
        )
    return contracts


def inventory_hash(tools: Sequence[DiscoveredTool] | Sequence[ToolContract]) -> str:
    """Audit hash over sorted ``{name, contract_hash}``, never an allow bit."""
    if tools and isinstance(tools[0], DiscoveredTool):
        contracts = _contracts(tools)  # type: ignore[arg-type]
    else:
        contracts = tuple(sorted(tools, key=lambda item: item.name))  # type: ignore[arg-type]
        names = [item.name for item in contracts]
        if len(names) != len(set(names)):
            raise ToolResultUnreadable(
                "the broker advertised duplicate tool names; the whole session is invalid."
            )
    return _hash([{"name": item.name, "contract_hash": item.contract_hash} for item in contracts])


ARGUMENT_PLACEHOLDERS: Mapping[Category, frozenset[str]] = {
    Category.READ_ACCOUNTS: frozenset(),
    Category.READ_POSITIONS: frozenset({"account_id"}),
    Category.READ_BALANCES: frozenset({"account_id"}),
    Category.READ_ORDERS: frozenset({"account_id"}),
    Category.READ_QUOTE: frozenset({"symbol"}),
    Category.READ_HISTORY: frozenset({"symbol", "count", "start_time", "end_time", "interval"}),
    Category.READ_INSTRUMENTS: frozenset({"symbol", "query"}),
    Category.READ_MARKET_HOURS: frozenset({"at"}),
    Category.ORDER_PREFLIGHT: frozenset(
        {
            "account_id",
            "symbol",
            "side",
            "qty",
            "order_type",
            "time_in_force",
            "extended_hours",
            "limit_price",
            "stop_price",
        }
    ),
    Category.ORDER_PLACE: frozenset(
        {
            "account_id",
            "symbol",
            "side",
            "qty",
            "order_type",
            "time_in_force",
            "extended_hours",
            "limit_price",
            "stop_price",
            "idempotency_key",
        }
    ),
    Category.ORDER_REPLACE: frozenset(
        {
            "account_id",
            "order_id",
            "qty",
            "order_type",
            "time_in_force",
            "extended_hours",
            "limit_price",
            "stop_price",
        }
    ),
    Category.ORDER_CANCEL: frozenset({"account_id", "order_id"}),
    Category.DENIED_MONEY_MOVEMENT: frozenset(),
    Category.DENIED_TRANSFERS: frozenset(),
    Category.DENIED_SETTINGS: frozenset(),
    Category.DENIED_CREDENTIALS: frozenset(),
}

REQUIRED_PLACEHOLDERS: Mapping[Category, frozenset[str]] = {
    Category.ORDER_PREFLIGHT: frozenset({"account_id", "symbol", "side", "qty"}),
    Category.ORDER_PLACE: frozenset({"account_id", "symbol", "side", "qty"}),
    Category.ORDER_REPLACE: frozenset({"order_id"}),
    Category.ORDER_CANCEL: frozenset({"order_id"}),
}

REQUIRED_RESULT_ROLES: Mapping[Category, tuple[str, ...]] = {
    Category.READ_ACCOUNTS: ("items", "account", "eligible", "kind"),
    Category.READ_POSITIONS: ("items", "account", "symbol", "quantity", "average_cost"),
    Category.READ_BALANCES: ("items", "account", "cash"),
    Category.READ_ORDERS: ("items", "account", "order_id"),
    Category.READ_QUOTE: ("price", "asof"),
    Category.READ_HISTORY: ("items", "timestamp", "open", "high", "low", "close", "volume"),
    Category.ORDER_PLACE: ("order_id", "quantity", "price", "filled_at"),
    Category.ORDER_CANCEL: (),
}

_PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")


def _placeholders(value: Any) -> frozenset[str]:
    """Collect placeholders recursively so array arguments cannot bypass the grammar."""
    if isinstance(value, str):
        return frozenset(_PLACEHOLDER.findall(value))
    if isinstance(value, Mapping):
        return frozenset().union(*(_placeholders(item) for item in value.values()))
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return frozenset().union(*(_placeholders(item) for item in value))
    return frozenset()


def _placeholders_in(value: Any) -> list[str]:
    """Every placeholder name inside a template, nested arrays and objects included."""
    if isinstance(value, str):
        return _PLACEHOLDER.findall(value)
    if isinstance(value, Mapping):
        return [key for item in value.values() for key in _placeholders_in(item)]
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [key for item in value for key in _placeholders_in(item)]
    return []


#: Values the runtime supplies for every history call besides the symbol and count:
#: a daily bar interval and a window wide enough to hold `count` sessions, so a
#: broker that takes a time range (Robinhood) and one that takes a count both work.
HISTORY_INTERVAL = "day"
HISTORY_CALENDAR_DAYS_PER_BAR = 2
HISTORY_WINDOW_SLACK_DAYS = 10


def history_values(symbol: str, count: int, *, now: datetime) -> dict[str, Any]:
    """The placeholder values for one `read.history` call, derived, never guessed.

    `start_time`/`end_time` are RFC 3339 UTC and bound the window; the caller keeps
    the most recent `count` bars of whatever the broker returns inside it.
    """
    if count < 1:
        raise ValueError("a history call needs at least one bar")
    start = now - timedelta(days=count * HISTORY_CALENDAR_DAYS_PER_BAR + HISTORY_WINDOW_SLACK_DAYS)
    return {
        "symbol": symbol,
        "count": count,
        "interval": HISTORY_INTERVAL,
        "start_time": start.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end_time": now.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


#: Tick places whole-share market orders good for the day, in regular hours; these
#: are the runtime's values for the order placeholders a mapping may bind. A broker
#: that spells them differently (gfd, regular_hours) takes a literal in the mapping.
ORDER_VALUES: Mapping[str, Any] = {
    "order_type": "market",
    "time_in_force": "day",
    "extended_hours": False,
}


def _render_template(value: Any, values: Mapping[str, Any], *, exact_strings: bool) -> Any:
    """Render nested templates without turning array elements into one JSON-looking string."""
    if isinstance(value, str):
        wanted = _PLACEHOLDER.findall(value)
        missing = [key for key in wanted if key not in values]
        if missing:
            raise KeyError(missing[0])
        exact = _PLACEHOLDER.fullmatch(value)
        if exact is not None and not exact_strings:
            return values[exact.group(1)]
        return _PLACEHOLDER.sub(lambda match: str(values[match.group(1)]), value)
    if isinstance(value, Mapping):
        return {
            str(key): _render_template(item, values, exact_strings=False)
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_render_template(item, values, exact_strings=False) for item in value]
    return value


def mapping_hash(
    category: Category, arguments: Mapping[str, Any], result: Mapping[str, str]
) -> str:
    """Hash category, templates, fixed literals, and result paths together."""
    return _hash({"category": category.value, "arguments": arguments, "result": result})


class ProofResult(ProfileModel):
    success: bool
    resolved: tuple[str, ...]
    unresolved: Mapping[str, str]
    detail: str

    @model_validator(mode="after")
    def _check(self) -> ProofResult:
        if not self.detail.strip():
            raise ValueError("a proof result must say what was established")
        if self.success and self.unresolved:
            raise ValueError("a successful proof has no unresolved result path")
        return self


class ProfileTool(ProfileModel):
    """One denied or explicitly confirmed tool and the content it binds."""

    category: Category
    contract: ToolContract
    arguments: Mapping[str, Any]
    result: Mapping[str, str]
    confirmed_contract_hash: str | None
    mapping_hash: str
    confirmed_at: AwareDatetime | None
    confirmed_by: Literal["terminal", "api"] | None
    categorizer_version: str
    proved_contract_hash: str | None
    proved_mapping_hash: str | None
    proved_at: AwareDatetime | None
    proof: ProofResult | None

    @model_validator(mode="after")
    def _check(self) -> ProfileTool:
        expected_mapping = mapping_hash(self.category, self.arguments, self.result)
        if self.mapping_hash != expected_mapping:
            raise ValueError(f"{self.contract.name} carries a mapping hash that does not match")
        if self.category.denied:
            if any(
                value is not None
                for value in (
                    self.confirmed_contract_hash,
                    self.confirmed_at,
                    self.confirmed_by,
                    self.proved_contract_hash,
                    self.proved_mapping_hash,
                    self.proved_at,
                    self.proof,
                )
            ):
                raise ValueError(
                    "denied tools carry no confirmation or proof; they are never called"
                )
            if self.arguments or self.result:
                raise ValueError("denied tools carry no call mapping; they are never called")
            return self
        if not self.category.callable:
            raise ValueError(f"{self.category.value} is not a callable category")
        if self.confirmed_contract_hash != self.contract.contract_hash:
            raise ValueError("a callable tool must confirm its exact contract hash")
        if self.confirmed_at is None or self.confirmed_by is None:
            raise ValueError("a mapped read or order tool requires who confirmed it and when")
        # Proposal checks are warnings by owner ruling.  Confirmation records the
        # person's exact draft; proof and the final JSON-schema check decide whether
        # it is usable, without silently repairing any missing value or path.
        proof_values = (
            self.proved_contract_hash,
            self.proved_mapping_hash,
            self.proved_at,
            self.proof,
        )
        if any(value is not None for value in proof_values) and not all(
            value is not None for value in proof_values
        ):
            raise ValueError("proof contract, mapping, time, and result are one evidence set")
        return self

    @property
    def proved(self) -> bool:
        return bool(
            self.proof
            and self.proof.success
            and self.proved_contract_hash == self.contract.contract_hash
            and self.proved_mapping_hash == self.mapping_hash
        )

    def placeholders_of(self, name: str) -> tuple[str, ...]:
        """The placeholder names one argument template refers to, in order."""
        return tuple(dict.fromkeys(_placeholders_in(self.arguments[name])))

    def missing_placeholders(self, values: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
        """Required inputs whose placeholders have no value in `values`, by argument.

        An optional input whose placeholders are all absent is simply omitted from
        the call (see `render`); a required one is reported here so the caller can
        say exactly which value to supply.
        """
        required = set(self.contract.input_schema.get("required") or ())
        missing: dict[str, tuple[str, ...]] = {}
        for name in self.arguments:
            absent = tuple(key for key in self.placeholders_of(name) if key not in values)
            if absent and name in required:
                missing[name] = absent
        return missing

    def render(self, values: Mapping[str, Any]) -> dict[str, Any]:
        """Render the call's arguments from `values`.

        An optional input whose placeholders all lack a value is left out rather
        than sent with a guess (a limit price on a market order, say). A required
        input with a missing placeholder refuses and names it.
        """
        required = set(self.contract.input_schema.get("required") or ())
        rendered: dict[str, Any] = {}
        for name, template in self.arguments.items():
            wanted = self.placeholders_of(name)
            absent = [key for key in wanted if key not in values]
            if absent and name not in required and len(absent) == len(wanted):
                continue
            expected_type = (
                self.contract.input_schema.get("properties", {}).get(name, {}).get("type")
            )
            try:
                rendered[name] = _render_template(
                    template,
                    values,
                    exact_strings=expected_type == "string",
                )
            except KeyError as exc:
                missing = [str(exc.args[0])]
                raise CapabilityUnmapped(
                    f"the confirmed mapping for {self.contract.name} needs {missing} for "
                    f"argument {name!r}, and this call has no such value. Fix the mapping "
                    "and confirm this tool again."
                ) from exc
        return rendered


class DriftDifference(ProfileModel):
    tool: str
    changes: tuple[str, ...]

    def sentence(self) -> str:
        return f"{self.tool}: {', '.join(self.changes)}"


class Profile(ProfileModel):
    """The user-approved broker contract; mutable runtime drift is excluded from its hash."""

    server: str
    account_id: str | None
    tools: Mapping[str, ProfileTool]
    inventory_hash: str
    data_class: Literal["display_only"]
    sanction: Literal["official", "community"]
    profile_format_version: str
    canonicalizer_version: str
    category_registry_version: str
    state: ProfileState
    observed_inventory_hash: str | None
    drift: tuple[DriftDifference, ...]
    profile_hash: str

    @model_validator(mode="after")
    def _check(self) -> Profile:
        if _HTTPS_SERVER.match(self.server) is None:
            raise ValueError("a broker profile server must be an https URL with a host")
        if self.account_id is not None and not self.account_id.strip():
            raise ValueError("a broker profile account is either absent or non-empty")
        if self.sanction != sanction_for(self.server):
            raise ValueError("profile sanction does not match the server host")
        if self.profile_format_version != PROFILE_FORMAT_VERSION:
            raise ValueError("the profile format version changed; confirm every tool again")
        if self.canonicalizer_version != CANONICALIZER_VERSION:
            raise ValueError("the canonicalizer version changed; confirm every tool again")
        if self.category_registry_version != CATEGORY_REGISTRY_VERSION:
            raise ValueError("the category registry changed; confirm every tool again")
        for name, tool in self.tools.items():
            if name != tool.contract.name:
                raise ValueError(f"profile key {name!r} does not match tool {tool.contract.name!r}")
        if self.account_id is None and any(
            tool.category.callable and tool.category is not Category.READ_ACCOUNTS
            for tool in self.tools.values()
        ):
            raise ValueError(
                "only read.accounts may be confirmed before the person selects an eligible account"
            )
        if self.profile_hash != profile_hash_for(self):
            raise ValueError("profile_hash does not match the content the user approved")
        return self

    def mapping_for(self, category: Category) -> ProfileTool:
        matches = [tool for tool in self.tools.values() if tool.category is category]
        if len(matches) > 1:
            raise CapabilityUnmapped(
                f"more than one confirmed tool claims {category.value}. Proof cannot choose "
                "between them; edit the draft and confirm one mapping."
            )
        if matches:
            return matches[0]
        raise CapabilityUnmapped(
            f"no confirmed tool is mapped to {category.value}. Tick refuses rather than "
            "guessing; run `tick broker propose`, then confirm that exact tool."
        )


def _profile_approved_body(profile: Profile | Mapping[str, Any]) -> dict[str, Any]:
    def plain(value: Any) -> Any:
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        if isinstance(value, Mapping):
            return {str(key): plain(item) for key, item in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
            return [plain(item) for item in value]
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, StrEnum):
            return value.value
        return value

    if isinstance(profile, Profile):
        body = profile.model_dump(mode="json")
    else:
        body = plain(profile)
    for key in ("profile_hash", "state", "observed_inventory_hash", "drift"):
        body.pop(key, None)
    for tool in body.get("tools", {}).values():
        for key in (
            "proved_contract_hash",
            "proved_mapping_hash",
            "proved_at",
            "proof",
        ):
            tool.pop(key, None)
    return body


def profile_hash_for(profile: Profile | Mapping[str, Any]) -> str:
    return _hash(_profile_approved_body(profile))


def build_profile(**values: Any) -> Profile:
    body = dict(values)
    body["profile_hash"] = profile_hash_for(body)
    return Profile.model_validate(body)


class ProposedTool(ProfileModel):
    contract: ToolContract
    category: Category | None
    arguments: Mapping[str, Any]
    result: Mapping[str, str]
    reason: str
    warnings: tuple[str, ...]
    original: ProposalReplyTool
    edits: tuple[ProposalEdit, ...]

    @property
    def note(self) -> str:
        """Compatibility label for terminal output; the draft now calls this a reason."""
        return self.reason


class ProposalReplyTool(ProfileModel):
    """One provider-authored row, before deterministic denial and warnings."""

    name: str
    category: Category | None
    arguments: Mapping[str, Any]
    result: Mapping[str, str]
    reason: str
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _sentence(self) -> ProposalReplyTool:
        if not self.name.strip():
            raise ValueError("a proposal row must name its advertised tool")
        if not self.reason.strip():
            raise ValueError("a proposal row must give the person a reason")
        return self


class ProposalReply(ProfileModel):
    model: str | None
    tools: tuple[ProposalReplyTool, ...]


class ProposalEdit(ProfileModel):
    field: str
    old: Any
    new: Any
    who: Literal["api", "terminal"]
    at: AwareDatetime


class ProfileProposal(ProfileModel):
    server: str
    account_id: str | None
    sanction: Literal["official", "community"]
    inventory_hash: str
    tools: Mapping[str, ProposedTool]
    categorizer_version: str
    proposed_at: AwareDatetime


@runtime_checkable
class Categorizer(Protocol):
    """One structured whole-inventory proposal through the model port."""

    version: str

    def propose(self, contracts: Sequence[ToolContract]) -> ProposalReply: ...


_DENIAL_HINTS: tuple[tuple[Category, tuple[str, ...]], ...] = (
    (Category.DENIED_CREDENTIALS, ("credential", "password", "secret", "token", "api_key")),
    (Category.DENIED_TRANSFERS, ("transfer", "wire", "withdraw", "deposit")),
    (Category.DENIED_MONEY_MOVEMENT, ("money_movement", "move_money", "fund_account")),
    (Category.DENIED_SETTINGS, ("setting", "preference", "profile_update")),
)

_CATEGORY_HINTS: tuple[tuple[Category, tuple[str, ...]], ...] = (
    (Category.ORDER_CANCEL, ("cancel",)),
    (Category.ORDER_REPLACE, ("replace", "modify_order", "update_order")),
    (Category.ORDER_PREFLIGHT, ("preflight", "preview_order", "validate_order")),
    (Category.ORDER_PLACE, ("place_order", "create_order", "submit_order")),
    (Category.READ_POSITIONS, ("position", "holding")),
    (Category.READ_BALANCES, ("balance", "cash")),
    (Category.READ_ACCOUNTS, ("account",)),
    (Category.READ_ORDERS, ("list_orders", "get_orders", "orders")),
    (Category.READ_HISTORY, ("history", "historical", "bars", "candles")),
    (Category.READ_QUOTE, ("quote", "last_price", "last_trade")),
    (Category.READ_INSTRUMENTS, ("instrument", "security_search", "symbol_search")),
    (Category.READ_MARKET_HOURS, ("market_hours", "market_clock", "trading_hours")),
)


def categorize(tool: DiscoveredTool) -> Category | None:
    """Deterministic name/schema proposal; annotations can veto, never authorize."""
    haystack = " ".join((tool.name, tool.description or "")).lower()
    for category, hints in _DENIAL_HINTS:
        if any(hint in haystack for hint in hints):
            return category
    name = tool.name.lower()
    proposed = next(
        (category for category, hints in _CATEGORY_HINTS if any(hint in name for hint in hints)),
        None,
    )
    description = (tool.description or "").lower()
    if proposed is Category.READ_ACCOUNTS and any(
        word in description for word in ("balance", "cash")
    ):
        proposed = Category.READ_BALANCES
    if proposed is None:
        proposed = next(
            (
                category
                for category, hints in _CATEGORY_HINTS
                if any(hint in description for hint in hints)
            ),
            None,
        )
    annotations = getattr(tool, "annotations", None)
    if hasattr(annotations, "model_dump"):
        annotations = annotations.model_dump(mode="json", by_alias=True, exclude_none=True)
    if proposed and proposed.value.startswith("read.") and isinstance(annotations, Mapping):
        if annotations.get("destructiveHint") is True or annotations.get("readOnlyHint") is False:
            return None
    return proposed


_ARGUMENT_HINTS: Mapping[str, tuple[str, ...]] = {
    "symbol": ("symbol", "ticker", "instrument"),
    "account_id": ("account_id", "account", "account_number", "accountid"),
    "qty": ("quantity", "qty", "shares", "amount"),
    "side": ("side", "action", "direction"),
    "order_id": ("order_id", "orderid", "id"),
    "count": ("count", "limit", "bars"),
    "order_type": ("order_type", "type"),
    "time_in_force": ("time_in_force", "tif"),
    "extended_hours": ("extended_hours", "allow_extended_hours"),
}

_RESULT_HINTS: Mapping[Category, Mapping[str, tuple[str, ...]]] = {
    Category.READ_QUOTE: {
        "price": ("last_price", "price", "last"),
        "asof": ("quoted_at", "asof", "timestamp"),
    },
    Category.READ_POSITIONS: {
        "items": ("positions", "holdings", "results"),
        "account": ("account", "account_id"),
        "symbol": ("symbol", "ticker"),
        "quantity": ("quantity", "qty", "shares"),
        "average_cost": ("average_cost", "avg_cost", "average_price"),
    },
    Category.READ_BALANCES: {
        "items": ("accounts", "balances", "results"),
        "account": ("account_id", "account"),
        "cash": ("cash", "cash_balance", "buying_power"),
    },
    Category.READ_ACCOUNTS: {
        "items": ("accounts", "results"),
        "account": ("account_id", "account", "id"),
        "eligible": ("agentic_allowed", "eligible"),
        "kind": ("brokerage_account_type", "kind", "type"),
    },
    Category.READ_ORDERS: {
        "items": ("orders", "results"),
        "account": ("account", "account_id"),
        "order_id": ("order_id", "id"),
        "status": ("status", "state"),
        "symbol": ("symbol", "ticker"),
        "side": ("side", "action"),
        "quantity": ("filled_quantity", "quantity"),
        "price": ("filled_price", "average_price", "price"),
        "filled_at": ("filled_at", "executed_at", "timestamp"),
    },
    Category.READ_HISTORY: {
        "items": ("bars", "history", "candles", "results"),
        "timestamp": ("timestamp", "ts", "at"),
        "open": ("open",),
        "high": ("high",),
        "low": ("low",),
        "close": ("close",),
        "volume": ("volume",),
    },
    Category.ORDER_PLACE: {
        "order_id": ("order_id", "id"),
        "quantity": ("filled_quantity", "quantity"),
        "price": ("filled_price", "average_price", "price"),
        "filled_at": ("filled_at", "executed_at", "timestamp"),
    },
    Category.ORDER_CANCEL: {
        "accepted": ("accepted",),
        "order_id": ("order_id", "id"),
        "cancelled_at": ("cancelled_at", "canceled_at", "timestamp"),
    },
}


def _argument_for(name: str) -> str | None:
    lowered = name.lower()
    for placeholder, aliases in _ARGUMENT_HINTS.items():
        if lowered in aliases:
            return placeholder
    return None


def _proposed_mapping(
    tool: DiscoveredTool, category: Category
) -> tuple[dict[str, Any], dict[str, str], str | None]:
    if category.denied:
        return {}, {}, "denied by the category registry; this tool can never be called"
    arguments: dict[str, Any] = {}
    required = set(tool.required_inputs())
    for name in tool.input_properties():
        placeholder = _argument_for(name)
        if placeholder is not None and placeholder in ARGUMENT_PLACEHOLDERS[category]:
            arguments[name] = "{" + placeholder + "}"
        elif name in required:
            return {}, {}, f"required input {name!r} has no value Tick can supply; left unmapped"
    result: dict[str, str] = {}
    declared = set(tool.output_properties())
    hints = _RESULT_HINTS.get(category, {})
    if category is Category.ORDER_CANCEL:
        hints = (
            {"accepted": hints["accepted"]}
            if "accepted" in declared
            else {
                "order_id": hints["order_id"],
                "cancelled_at": hints["cancelled_at"],
            }
        )
    for role, candidates in hints.items():
        result[role] = next(
            (candidate for candidate in candidates if candidate in declared), candidates[0]
        )
    note = (
        None if declared else "no output schema is declared; every proposed result path needs proof"
    )
    return arguments, result, note


def denial_for(tool: DiscoveredTool | ToolContract) -> Category | None:
    """The only proposal rule that overrides both model and person."""
    name = tool.name.lower()
    if any(word in name for word in ("credential", "password", "secret", "token", "api_key")):
        return Category.DENIED_CREDENTIALS
    if any(word in name for word in ("transfer", "withdraw", "deposit")):
        return Category.DENIED_TRANSFERS
    if any(word in name for word in ("move_money", "fund_account", "money_movement")):
        return Category.DENIED_MONEY_MOVEMENT
    if any(word in name for word in ("account_setting", "account_preference")):
        return Category.DENIED_SETTINGS
    return None


def _deterministic_reply(
    tools: Sequence[DiscoveredTool], contracts: Sequence[ToolContract]
) -> ProposalReply:
    by_name = {tool.name: tool for tool in tools}
    rows: list[ProposalReplyTool] = []
    claimed: set[Category] = set()
    for contract in contracts:
        tool = by_name[contract.name]
        category = denial_for(tool) or categorize(tool)
        arguments: dict[str, Any] = {}
        result: dict[str, str] = {}
        reason = (
            "the deterministic fallback found no category; connect a provider or map it by hand."
        )
        if category is not None:
            arguments, result, note = _proposed_mapping(tool, category)
            reason = note or f"the deterministic fallback matched {category.value}; review it."
            if (not category.denied and not arguments and tool.required_inputs()) or (
                category.callable and category in claimed
            ):
                category = None
                reason = note or (
                    "another tool already claimed this callable category; map it by hand if needed."
                )
            elif category.callable:
                claimed.add(category)
        rows.append(
            ProposalReplyTool(
                name=contract.name,
                category=category,
                arguments=arguments,
                result=result,
                reason=reason,
            )
        )
    return ProposalReply(model=None, tools=tuple(rows))


def propose_profile(
    tools: Sequence[DiscoveredTool],
    *,
    server: str,
    account_id: str | None,
    proposed_at: datetime,
    categorizer: Categorizer | None = None,
) -> ProfileProposal:
    """Build an editable draft; no row has authority until separately confirmed."""
    if account_id is not None and not account_id.strip():
        raise ValueError("an account id is either absent or non-empty")
    contracts = _contracts(tools)
    if categorizer is None:
        reply = _deterministic_reply(tools, contracts)
        version = CATEGORIZER_VERSION
    else:
        from .profile_model import check_proposal

        reply = check_proposal(categorizer.propose(contracts), contracts)
        version = categorizer.version
    reply_by_name = {row.name: row for row in reply.tools}
    proposed: dict[str, ProposedTool] = {}
    for contract in contracts:
        row = reply_by_name.get(contract.name)
        if row is None:
            row = ProposalReplyTool(
                name=contract.name,
                category=None,
                arguments={},
                result={},
                reason=(
                    "the provider returned no row for this tool; map it by hand if Tick needs it."
                ),
            )
        category = denial_for(contract) or row.category
        reason = row.reason
        if denial_for(contract) is not None:
            reason = "Tick's denial registry makes this capability permanently uncallable."
        proposed[contract.name] = ProposedTool(
            contract=contract,
            category=category,
            arguments={} if category is not None and category.denied else row.arguments,
            result={} if category is not None and category.denied else row.result,
            reason=reason,
            warnings=row.warnings,
            original=row,
            edits=(),
        )
    return ProfileProposal(
        server=server,
        account_id=account_id,
        sanction=sanction_for(server),
        inventory_hash=inventory_hash(contracts),
        tools=proposed,
        categorizer_version=version,
        proposed_at=proposed_at,
    )


def sanction_for(server: str) -> Literal["official", "community"]:
    match = _HTTPS_SERVER.match(server)
    host = match.group(1).lower() if match is not None else ""
    return "official" if host in HOST_ALLOWLIST else "community"


def server_host(server: str) -> str:
    """The validated HTTPS host without importing a network-capable package."""
    match = _HTTPS_SERVER.match(server)
    if match is None:
        raise ValueError("a broker server must be an https URL with a host")
    return match.group(1).lower()


def profile_path(home: str | os.PathLike[str]) -> Path:
    return Path(home) / "broker" / PROFILE_FILE


def proposal_path(home: str | os.PathLike[str]) -> Path:
    return Path(home) / "broker" / PROPOSAL_FILE


def profile_ledger_path(home: str | os.PathLike[str]) -> Path:
    return Path(home) / "broker" / "records.jsonl"


def save_profile(home: str | os.PathLike[str], profile: Profile) -> Path:
    return write_private_file(profile_path(home), profile.model_dump_json(indent=2))


def save_proposal(home: str | os.PathLike[str], proposal: ProfileProposal) -> Path:
    return write_private_file(proposal_path(home), proposal.model_dump_json(indent=2))


def load_proposal(home: str | os.PathLike[str]) -> ProfileProposal:
    path = proposal_path(home)
    try:
        return ProfileProposal.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError) as exc:
        raise ToolResultUnreadable(
            f"{path} is not an editable broker proposal ({exc}). Propose one again."
        ) from exc


def edit_proposal(
    proposal: ProfileProposal,
    tool_name: str,
    changes: Mapping[str, Any],
    *,
    who: Literal["api", "terminal"],
    at: datetime,
) -> ProfileProposal:
    """Record exact person-owned edits while retaining the provider's original row."""
    if set(changes) - {"category", "arguments", "result"} or not changes:
        raise ValueError(
            "an edit changes category, arguments, or result. Choose at least one field."
        )
    current = proposal.tools.get(tool_name)
    if current is None:
        raise ValueError(f"tool {tool_name!r} is not in this proposal. Refresh the draft.")
    if denial_for(current.contract) is not None:
        raise ValueError(
            f"tool {tool_name!r} is denied by Tick's registry and cannot be remapped. "
            "Leave it locked."
        )
    values = current.model_dump(mode="python")
    edits = list(current.edits)
    for field, raw in changes.items():
        if field == "category":
            new: Any = None if raw is None or raw == "not used" else Category(raw)
            if new is not None and new.denied:
                raise ValueError(
                    "people may choose a callable category or not used; denial comes only "
                    "from Tick's registry."
                )
        elif field == "arguments":
            if not isinstance(raw, Mapping) or any(not isinstance(key, str) for key in raw):
                raise ValueError("arguments must be an object keyed by declared input name.")
            new = dict(raw)
        else:
            if not isinstance(raw, Mapping) or any(
                not isinstance(key, str) or not isinstance(value, str) for key, value in raw.items()
            ):
                raise ValueError("result must be an object of role to dotted-path strings.")
            new = dict(raw)
        old = values[field]
        if old != new:
            edits.append(ProposalEdit(field=field, old=old, new=new, who=who, at=at))
            values[field] = new
    values["edits"] = tuple(edits)
    changed = ProposedTool.model_validate(values)
    tools = dict(proposal.tools)
    tools[tool_name] = changed

    from .profile_model import check_proposal

    review = ProposalReply(
        model=(
            proposal.categorizer_version.removeprefix("model-v1:")
            if proposal.categorizer_version.startswith("model-v1:")
            else None
        ),
        tools=tuple(
            ProposalReplyTool(
                name=name,
                category=row.category,
                arguments=row.arguments,
                result=row.result,
                reason=row.reason,
            )
            for name, row in tools.items()
        ),
    )
    warned_rows = check_proposal(review, [row.contract for row in tools.values()]).tools
    warned = {row.name: row.warnings for row in warned_rows}
    tools = {
        name: row.model_copy(update={"warnings": warned.get(name, ())})
        for name, row in tools.items()
    }
    return proposal.model_copy(update={"tools": tools})


def load_profile(home: str | os.PathLike[str]) -> Profile | None:
    path = profile_path(home)
    if not path.exists():
        return migrate_toolmap(home)
    try:
        return Profile.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValidationError, ValueError) as exc:
        raise ToolResultUnreadable(
            f"{path} is not a broker profile Tick can trust: {exc}. Run `tick broker "
            "propose`, then confirm each callable tool again."
        ) from exc


def confirm_profile(
    home: str | os.PathLike[str],
    profile: Profile,
    *,
    actor: str,
    at: datetime,
    via: str | None = None,
    transcript_hash: str | None = None,
) -> Path:
    """Persist approval and append evidence; a hash alone is not a signature."""
    if not actor.strip():
        raise ValueError("profile confirmation must name the local actor")
    path = save_profile(home, profile)
    source = via or ("api" if actor == "box-api" else "cli")
    if source not in {"api", "chat", "cli"}:
        raise ValueError("profile confirmation source must be api, chat, or cli")
    if source == "chat" and not transcript_hash:
        raise ValueError("chat profile confirmation requires its transcript hash")
    payload = {
        "event": "profile_confirmed",
        "profile_hash": profile.profile_hash,
        "actor": actor,
        "via": source,
        "at": at,
        "tools": sorted(profile.tools),
    }
    if transcript_hash is not None:
        payload["transcript_hash"] = transcript_hash
    Ledger(profile_ledger_path(home), clock=lambda: at).append(
        RecordKind.NOTE,
        payload,
        source=DataSource.RUNTIME,
    )
    return path


def has_confirmation_note(home: str | os.PathLike[str], profile_hash: str) -> bool:
    return any(
        record.kind is RecordKind.NOTE
        and record.payload.get("event") == "profile_confirmed"
        and record.payload.get("profile_hash") == profile_hash
        for record in read(profile_ledger_path(home))
    )


def diff_profile(profile: Profile, live: Sequence[ToolContract]) -> tuple[DriftDifference, ...]:
    """Readable per-tool changes; additions are unmapped rather than profile drift."""
    current = {tool.name: tool for tool in live}
    differences: list[DriftDifference] = []
    for name, stored in profile.tools.items():
        observed = current.get(name)
        if observed is None:
            if stored.category.callable:
                differences.append(DriftDifference(tool=name, changes=("removed",)))
            continue
        if stored.contract.contract_hash == observed.contract_hash:
            continue
        if not stored.category.callable:
            continue
        changes: list[str] = []
        old = stored.contract
        if old.input_schema != observed.input_schema:
            changes.append("changed input schema")
        if old.output_schema != observed.output_schema:
            changes.append("changed output schema")
        if old.execution != observed.execution:
            changes.append("changed execution semantics")
        if old.title != observed.title:
            changes.append("changed title")
        if old.description != observed.description:
            changes.append("changed description")
        if old.annotations != observed.annotations:
            changes.append("changed annotations")
        differences.append(
            DriftDifference(tool=name, changes=tuple(changes or ("changed contract",)))
        )
    for name in sorted(set(current) - set(profile.tools)):
        differences.append(DriftDifference(tool=name, changes=("added; unmapped",)))
    return tuple(differences)


_VERIFY_TOKEN = object()


class VerifiedSessionProfile:
    """Fresh per-tool authorization bound to one complete open-session inventory."""

    def __init__(
        self,
        *,
        profile: Profile,
        session: Any,
        contracts: Mapping[str, ToolContract],
        states: Mapping[str, ToolState],
        inventory_hash_value: str,
        confirmation_recorded: bool,
        token: object,
    ) -> None:
        if token is not _VERIFY_TOKEN:
            raise TypeError(
                "VerifiedSessionProfile can only be created from a verified open session"
            )
        self.profile = profile
        self.session = session
        self.contracts = dict(contracts)
        self.states = dict(states)
        self.inventory_hash = inventory_hash_value
        self.confirmation_recorded = confirmation_recorded
        self._revoked_reason: str | None = None

    def revoke(self, reason: str) -> None:
        self._revoked_reason = reason

    def mapping_for(self, category: Category, *, require_proof: bool) -> ProfileTool:
        if self._revoked_reason is not None:
            raise CapabilityUnmapped(
                f"the session authorization was revoked ({self._revoked_reason}). Refresh the "
                "inventory and confirm any changed tool before trying again."
            )
        if not self.confirmation_recorded:
            raise CapabilityUnmapped(
                f"profile {self.profile.profile_hash} has no verifying profile_confirmed "
                "ledger note. Confirm the profile again before calling any broker tool."
            )
        mapping = self.profile.mapping_for(category)
        state = self.states.get(mapping.contract.name, ToolState.UNMAPPED)
        if state is not ToolState.CONFIRMED:
            raise CapabilityUnmapped(
                f"{mapping.contract.name} is {state.value}; {category.value} is unavailable "
                "and no broker call was made. Run `tick broker status`, then reconfirm "
                "that exact tool if its contract changed."
            )
        if require_proof and not mapping.proved:
            raise CapabilityUnmapped(
                f"{mapping.contract.name} is confirmed but not proven for its exact contract "
                "and mapping. Run `tick broker prove` with user-supplied probe inputs."
            )
        return mapping

    def refresh_tool(self, name: str) -> None:
        tools = self.session.list_tools()
        live = {contract.name: contract for contract in _contracts(tools)}
        refreshed_states: dict[str, ToolState] = {}
        for live_name, observed_contract in live.items():
            stored = self.profile.tools.get(live_name)
            if stored is None:
                refreshed_states[live_name] = ToolState.UNMAPPED
            elif stored.category.denied:
                refreshed_states[live_name] = ToolState.DENIED
            elif stored.contract.contract_hash == observed_contract.contract_hash:
                refreshed_states[live_name] = ToolState.CONFIRMED
            else:
                refreshed_states[live_name] = ToolState.DRIFTED
        for stored_name, stored in self.profile.tools.items():
            if stored_name not in live:
                refreshed_states[stored_name] = (
                    ToolState.DRIFTED if stored.category.callable else ToolState.DENIED
                )
        self.contracts = live
        self.states = refreshed_states
        self.inventory_hash = inventory_hash(tuple(live.values()))
        expected = self.contracts.get(name)
        observed = live.get(name)
        stored = self.profile.tools.get(name)
        if (
            expected is None
            or observed is None
            or stored is None
            or stored.contract.contract_hash != observed.contract_hash
        ):
            self.states[name] = ToolState.DRIFTED
            raise CapabilityUnmapped(
                f"{name} changed after session verification; the mutating call was refused "
                "before tools/call. Run `tick broker status`, then confirm that tool again."
            )


def verify_session_profile(
    profile: Profile,
    session: Any,
    *,
    server: str,
    account_id: str | None,
    confirmation_recorded: bool,
) -> VerifiedSessionProfile:
    """Bind an open session only after its complete inventory matches per tool."""
    if server != profile.server:
        raise ToolResultUnreadable(
            f"the requested server {server} does not match profile.server {profile.server}; "
            "the whole session is invalid and no tool can be called. Use the profile server."
        )
    if account_id != profile.account_id:
        raise ToolResultUnreadable(
            f"account {account_id} does not match profile account {profile.account_id}; "
            "the whole session is invalid and no tool can be called. Confirm a profile for it."
        )
    tools = session.list_tools()
    contracts = _contracts(tools)
    current = {tool.name: tool for tool in contracts}
    states: dict[str, ToolState] = {}
    for name, observed in current.items():
        stored = profile.tools.get(name)
        if stored is None:
            states[name] = ToolState.UNMAPPED
        elif stored.category.denied:
            states[name] = ToolState.DENIED
        elif stored.contract.contract_hash == observed.contract_hash:
            states[name] = ToolState.CONFIRMED
        else:
            states[name] = ToolState.DRIFTED
    for name, stored in profile.tools.items():
        if name not in current:
            states[name] = ToolState.DRIFTED if stored.category.callable else ToolState.DENIED
    return VerifiedSessionProfile(
        profile=profile,
        session=session,
        contracts=current,
        states=states,
        inventory_hash_value=inventory_hash(contracts),
        confirmation_recorded=confirmation_recorded,
        token=_VERIFY_TOKEN,
    )


def _proof_of(
    mapping: ProfileTool,
    payload: Any,
    account_id: str,
    arguments: Mapping[str, Any],
) -> ProofResult:
    """Resolve and type-check mapped result roles; missing remains explicit."""
    from .toolmap import decimal_at, dig, timestamp_at, whole_at

    resolved: list[str] = []
    unresolved: dict[str, str] = {}
    rows: list[Any] | None = None
    if "items" in mapping.result:
        candidate = dig(payload, mapping.result["items"])
        if isinstance(candidate, Sequence) and not isinstance(candidate, str | bytes):
            rows = list(candidate)
            resolved.append("items")
        elif isinstance(candidate, Mapping):
            rows = [candidate]
            resolved.append("items")
        else:
            unresolved["items"] = (
                f"the broker answer carries no list at {mapping.result['items']!r}"
            )
    scoped_rows = rows
    if rows is not None and "account" in mapping.result:
        account_path = mapping.result["account"]
        scoped_rows = [
            row
            for row in rows
            if isinstance(row, Mapping)
            and (account_path == "{account_id}" or str(dig(row, account_path)) == account_id)
        ]
        if not scoped_rows:
            unresolved["account"] = f"no result row is scoped to configured account {account_id}"
        elif (
            mapping.category in {Category.READ_ACCOUNTS, Category.READ_BALANCES}
            and len(scoped_rows) != 1
        ):
            unresolved["items"] = (
                f"the broker returned {len(scoped_rows)} configured-account rows; "
                "exactly one is required"
            )
            if "items" in resolved:
                resolved.remove("items")
    if rows is not None and mapping.category is Category.READ_HISTORY:
        count_argument = next(
            (
                name
                for name in mapping.arguments
                if _argument_for(name) == "count" and name in arguments
            ),
            None,
        )
        if count_argument is not None:
            try:
                expected_count = int(arguments[count_argument])
            except (TypeError, ValueError):
                unresolved["items"] = "the confirmed history count is not a whole number"
            else:
                if len(rows) != expected_count:
                    unresolved["items"] = (
                        f"the broker returned {len(rows)} history rows; "
                        f"the proof requested exactly {expected_count}"
                    )
            if "items" in unresolved and "items" in resolved:
                resolved.remove("items")
    for role, path in mapping.result.items():
        if role == "items":
            continue
        targets = scoped_rows if scoped_rows is not None else [payload]
        if not targets:
            unresolved[role] = (
                f"no configured-account result row exists to validate {path!r} and its type"
            )
            continue
        failures: list[str] = []
        matched_account = False
        for target in targets:
            if not isinstance(target, Mapping):
                failures.append("result row is not an object")
                continue
            if path == "{account_id}":
                value = arguments.get("account_id") or account_id
            elif role in {
                "price",
                "cash",
                "average_cost",
                "open",
                "high",
                "low",
                "close",
            }:
                value = decimal_at(target, path, role)
            elif role in {"quantity", "volume"}:
                value = whole_at(target, path, role)
            elif role in {"asof", "timestamp", "filled_at", "cancelled_at"}:
                value = timestamp_at(target, path, role)
            else:
                value = dig(target, path)
                if value is None or (isinstance(value, str) and not value.strip()):
                    value = None
            if hasattr(value, "reason"):
                failures.append(value.reason)
            elif value is None:
                failures.append(f"the broker answer carries no usable {path!r}")
            else:
                if role == "account" and str(value) == account_id:
                    matched_account = True
        if role == "account" and rows and not matched_account:
            failures.append(f"no result row is scoped to configured account {account_id}")
        if failures:
            unresolved[role] = "; ".join(dict.fromkeys(failures))
        else:
            resolved.append(role)
    success = not unresolved
    return ProofResult(
        success=success,
        resolved=tuple(sorted(resolved)),
        unresolved=unresolved,
        detail=(
            "all confirmed result paths resolved with expected types, timezone-aware "
            "timestamps, cardinality, and configured-account scope"
            if success
            else "one or more confirmed result paths did not resolve; this tool remains unproven"
        ),
    )


def prove_profile(
    profile: Profile,
    verified: VerifiedSessionProfile,
    *,
    probe_values: Mapping[str, Any],
    at: datetime,
) -> tuple[Profile, Mapping[str, ProofResult]]:
    """Exercise confirmed reads and preflight only, using caller-supplied inputs.

    ``order.place``, ``order.replace``, and ``order.cancel`` are never called.
    A preflight proves only itself.  Values not inherent in the configured
    profile (for example a symbol or bar count) must be in ``probe_values``;
    this function never invents one.
    """
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import SchemaError
    from jsonschema.exceptions import ValidationError as SchemaValidationError

    outcomes: dict[str, ProofResult] = {}
    updated: dict[str, ProfileTool] = dict(profile.tools)
    for name, stored in profile.tools.items():
        if not (
            stored.category.value.startswith("read.") or stored.category is Category.ORDER_PREFLIGHT
        ):
            continue
        try:
            mapping = verified.mapping_for(stored.category, require_proof=False)
            values = {"account_id": profile.account_id, **ORDER_VALUES, **probe_values}
            if stored.category is Category.READ_HISTORY and "count" in values:
                try:
                    count = int(values["count"])
                except (TypeError, ValueError):
                    count = 0
                if count >= 1:
                    values = {
                        **values,
                        **history_values(str(values.get("symbol", "")), count, now=at),
                    }
            missing = mapping.missing_placeholders(values)
            if missing:
                needs = sorted({key for keys in missing.values() for key in keys})
                proof = ProofResult(
                    success=False,
                    resolved=(),
                    unresolved={"needs": ", ".join(needs)},
                    detail=(
                        f"this tool needs probe values for: {', '.join(needs)}. Supply them "
                        "and prove again."
                    ),
                )
                outcomes[name] = proof
                values_dump = mapping.model_dump(mode="python")
                values_dump.update(
                    {
                        "proved_contract_hash": stored.contract.contract_hash,
                        "proved_mapping_hash": stored.mapping_hash,
                        "proved_at": at,
                        "proof": proof,
                    }
                )
                updated[name] = ProfileTool.model_validate(values_dump)
                del mapping
                continue
            arguments = mapping.render(values)
            Draft202012Validator.check_schema(mapping.contract.input_schema)
            Draft202012Validator(mapping.contract.input_schema).validate(arguments)
            payload = verified.session.call_tool(name, arguments)
            proof = _proof_of(mapping, payload, profile.account_id, arguments)
        except (
            CapabilityUnmapped,
            ToolResultUnreadable,
            KeyError,
            SchemaError,
            SchemaValidationError,
        ) as exc:
            proof = ProofResult(
                success=False,
                resolved=(),
                unresolved={"call": str(exc)},
                detail=(
                    f"the proof call was refused or unreadable: {str(exc)[:240]}. Inspect "
                    "this tool's mapping and the probe values, then prove again."
                ),
            )
        outcomes[name] = proof
        values = (
            mapping.model_dump(mode="python")
            if "mapping" in locals()
            else stored.model_dump(mode="python")
        )
        values.update(
            {
                "proved_contract_hash": stored.contract.contract_hash,
                "proved_mapping_hash": stored.mapping_hash,
                "proved_at": at,
                "proof": proof,
            }
        )
        updated[name] = ProfileTool.model_validate(values)
        if "mapping" in locals():
            del mapping
    rebuilt = build_profile(
        server=profile.server,
        account_id=profile.account_id,
        tools=updated,
        inventory_hash=profile.inventory_hash,
        data_class=profile.data_class,
        sanction=profile.sanction,
        profile_format_version=profile.profile_format_version,
        canonicalizer_version=profile.canonicalizer_version,
        category_registry_version=profile.category_registry_version,
        state=profile.state,
        observed_inventory_hash=profile.observed_inventory_hash,
        drift=profile.drift,
    )
    return rebuilt, outcomes


def prove_proposal(
    proposal: ProfileProposal,
    session: Any,
    *,
    probe_values: Mapping[str, Any],
    at: datetime,
) -> Mapping[str, ProofResult]:
    """Exercise proposed reads and preflight without creating authorization.

    Setup proof is deliberately narrower than a verified profile: the live
    contract must still match exactly, only read tools and preflight may be
    called, and no result is installed into the callable profile. Missing
    person values remain named in the outcome instead of being invented.
    """
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import SchemaError
    from jsonschema.exceptions import ValidationError as SchemaValidationError

    live = {contract.name: contract for contract in _contracts(session.list_tools())}
    outcomes: dict[str, ProofResult] = {}
    for name, mapping in proposal.tools.items():
        if mapping.category is None or not (
            mapping.category.value.startswith("read.")
            or mapping.category is Category.ORDER_PREFLIGHT
        ):
            continue
        try:
            observed = live.get(name)
            if observed is None or observed.contract_hash != mapping.contract.contract_hash:
                raise CapabilityUnmapped(
                    f"{name} no longer matches the proposed contract. Refresh the inventory "
                    "and propose the complete document again."
                )
            values: dict[str, Any] = {**ORDER_VALUES, **probe_values}
            if proposal.account_id is not None:
                values["account_id"] = proposal.account_id
            if mapping.category is Category.READ_HISTORY and "count" in values:
                count = int(values["count"])
                if count >= 1:
                    values.update(history_values(str(values.get("symbol", "")), count, now=at))
            required = set(mapping.contract.input_schema.get("required") or ())
            missing: dict[str, tuple[str, ...]] = {}
            for argument, template in mapping.arguments.items():
                absent = tuple(key for key in _placeholders(template) if key not in values)
                if absent and argument in required:
                    missing[argument] = absent
            if missing:
                needs = sorted({key for keys in missing.values() for key in keys})
                outcomes[name] = ProofResult(
                    success=False,
                    resolved=(),
                    unresolved={"needs": ", ".join(needs)},
                    detail=(
                        f"this tool needs probe values for: {', '.join(needs)}. Supply them "
                        "and prove again."
                    ),
                )
                continue
            arguments: dict[str, Any] = {}
            for argument, template in mapping.arguments.items():
                wanted = _placeholders(template)
                if wanted and not wanted.intersection(values) and argument not in required:
                    continue
                expected_type = (
                    mapping.contract.input_schema.get("properties", {})
                    .get(argument, {})
                    .get("type")
                )
                arguments[argument] = _render_template(
                    template,
                    values,
                    exact_strings=expected_type == "string",
                )
            Draft202012Validator.check_schema(mapping.contract.input_schema)
            Draft202012Validator(mapping.contract.input_schema).validate(arguments)
            payload = session.call_tool(name, arguments)
            outcomes[name] = _proof_of(
                mapping, payload, proposal.account_id or "unbound account", arguments
            )
        except (
            CapabilityUnmapped,
            ToolResultUnreadable,
            SchemaError,
            SchemaValidationError,
            TypeError,
            ValueError,
        ) as exc:
            outcomes[name] = ProofResult(
                success=False,
                resolved=(),
                unresolved={"call": str(exc)},
                detail=(
                    f"the proof call was refused or unreadable: {str(exc)[:240]}. Inspect "
                    "this tool's mapping and the probe values, then prove again."
                ),
            )
    return outcomes


def migrate_toolmap(home: str | os.PathLike[str]) -> Profile | None:
    """Convert the prototype map once into drifted evidence, never confirmation."""
    root = Path(home)
    candidates = (root / "broker" / "toolmap.json", root / "robinhood" / "toolmap.json")
    old_path = next((path for path in candidates if path.exists()), None)
    if old_path is None:
        return None
    try:
        raw = json.loads(old_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ToolResultUnreadable(
            f"prototype tool map {old_path} is unreadable ({exc}); no broker tool is callable."
        ) from exc
    from tick.auth.provider import ROBINHOOD_MCP_URL

    from .toolmap import Capability

    categories = {
        Capability.QUOTE.value: Category.READ_QUOTE,
        Capability.POSITIONS.value: Category.READ_POSITIONS,
        Capability.ACCOUNT.value: Category.READ_BALANCES,
        Capability.PLACE_ORDER.value: Category.ORDER_PLACE,
        Capability.CANCEL_ORDER.value: Category.ORDER_CANCEL,
        Capability.LIST_ORDERS.value: Category.READ_ORDERS,
    }
    tools: dict[str, ProfileTool] = {}
    for capability_name, mapped in (raw.get("capabilities") or {}).items():
        category = categories.get(capability_name)
        if category is None or not isinstance(mapped, Mapping):
            continue
        name = str(mapped.get("tool") or "")
        if not name:
            continue
        shape = {"name": name, "input_schema": {}, "output_schema": None, "execution": None}
        shape_digest = _hash(shape)
        contract_digest = _hash(
            {"shape_hash": shape_digest, "title": None, "description": None, "annotations": None}
        )
        contract = ToolContract(
            name=name,
            title=None,
            description=None,
            input_schema={},
            output_schema=None,
            annotations=None,
            execution=None,
            shape_hash=shape_digest,
            contract_hash=contract_digest,
        )
        arguments = dict(mapped.get("arguments") or {})
        result = dict(mapped.get("result") or {})
        when = raw.get("discovered_at") or "1970-01-01T00:00:00+00:00"
        tools[name] = ProfileTool(
            category=category,
            contract=contract,
            arguments=arguments,
            result=result,
            confirmed_contract_hash=contract.contract_hash,
            mapping_hash=mapping_hash(category, arguments, result),
            confirmed_at=when,
            confirmed_by="terminal",
            categorizer_version="prototype-toolmap-migration",
            proved_contract_hash=None,
            proved_mapping_hash=None,
            proved_at=None,
            proof=None,
        )
    approved_inventory = inventory_hash(tuple(tool.contract for tool in tools.values()))
    profile = build_profile(
        server=ROBINHOOD_MCP_URL,
        account_id=str(raw.get("account_id") or ""),
        tools=tools,
        inventory_hash=approved_inventory,
        data_class="display_only",
        sanction="official",
        profile_format_version=PROFILE_FORMAT_VERSION,
        canonicalizer_version=CANONICALIZER_VERSION,
        category_registry_version=CATEGORY_REGISTRY_VERSION,
        state=ProfileState.DRIFTED,
        observed_inventory_hash=None,
        drift=(DriftDifference(tool="prototype tool map", changes=("requires confirmation",)),),
    )
    save_profile(root, profile)
    return profile
