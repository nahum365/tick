"""The interview's structure: required slots, schemas, and validators.

Questions are data so the no-advice constraint can be tested over the whole
script. A slot has no default: an unanswered field stays absent, and the next
operation is another question rather than a value Tick selected.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import TypeAdapter

from tick.agents import ModelAgentSpec, Provider
from tick.engine import check_cadence
from tick.runtime import ApprovalMode
from tick.spec import SYMBOL_PATTERN, Cadence, ExactDecimal, Rule, Session

__all__ = [
    "AgentKind",
    "SLOTS",
    "Slot",
    "meaningful_slots",
    "slot_by_name",
]


class AgentKind(StrEnum):
    """The two document shapes the interview may produce."""

    RULE = "rule"
    MODEL = "model"


Validator = Callable[[Any], Any]


@dataclass(frozen=True, slots=True)
class Slot:
    """One required answer and the exact schema used to extract it."""

    name: str
    question: str
    explains: str
    type: str
    schema: Mapping[str, Any]
    validator: Validator
    kinds: frozenset[AgentKind]


def _schema(adapter: TypeAdapter[Any]) -> dict[str, Any]:
    schema = adapter.json_schema()

    def remove_defaults(node: Any) -> None:
        if isinstance(node, dict):
            node.pop("default", None)
            for value in node.values():
                remove_defaults(value)
        elif isinstance(node, list):
            for value in node:
                remove_defaults(value)

    remove_defaults(schema)
    return schema


def _validate_universe(value: Any) -> list[str]:
    items = TypeAdapter(list[str]).validate_python(value)
    if not items:
        raise ValueError("universe must name at least one symbol")
    seen: set[str] = set()
    for symbol in items:
        if not SYMBOL_PATTERN.match(symbol):
            raise ValueError(
                f"universe symbol {symbol!r} must use the validated placeholder-shaped format"
            )
        if symbol in seen:
            raise ValueError(f"universe lists {symbol!r} more than once")
        seen.add(symbol)
    return sorted(items)


def _validate_cadence(value: Any) -> dict[str, Any]:
    cadence = TypeAdapter(Cadence).validate_python(value)
    check_cadence(cadence)
    return cadence.model_dump(mode="json")


def _validate_kind(value: Any) -> str:
    return TypeAdapter(AgentKind).validate_python(value).value


def _validate_rules(value: Any) -> list[dict[str, Any]]:
    rules = TypeAdapter(list[Rule]).validate_python(value)
    if not rules:
        raise ValueError("a rule agent must have at least one rule")
    ids = [rule.id for rule in rules]
    if len(ids) != len(set(ids)):
        raise ValueError("rule ids must be unique")
    return [rule.model_dump(mode="json") for rule in rules]


def _validate_provider(value: Any) -> str:
    return TypeAdapter(Provider).validate_python(value).value


def _validate_model(value: Any) -> str:
    text = TypeAdapter(str).validate_python(value)
    return ModelAgentSpec._check_model(text)


def _validate_instructions(value: Any) -> str:
    text = TypeAdapter(str).validate_python(value)
    if not text.strip():
        raise ValueError("instructions must contain the user's words")
    return text


def _validate_percentage(value: Any) -> str:
    exact = TypeAdapter(ExactDecimal).validate_python(value)
    if exact <= 0 or exact > 100:
        raise ValueError("a cage percentage must be greater than zero and at most one hundred")
    return format(exact, "f")


def _validate_position_count(value: Any) -> int:
    count = TypeAdapter(int).validate_python(value)
    if count < 1:
        raise ValueError("the maximum position count must be at least one")
    return count


def _validate_order_amount(value: Any) -> str:
    exact = TypeAdapter(ExactDecimal).validate_python(value)
    if exact <= 0:
        raise ValueError("the maximum order amount must be greater than zero")
    return format(exact, "f")


def _validate_session(value: Any) -> str:
    return TypeAdapter(Session).validate_python(value).value


def _validate_approval(value: Any) -> str:
    return TypeAdapter(ApprovalMode).validate_python(value).value


ALL_KINDS = frozenset({AgentKind.RULE, AgentKind.MODEL})
RULE_ONLY = frozenset({AgentKind.RULE})
MODEL_ONLY = frozenset({AgentKind.MODEL})

_DECIMAL_SCHEMA = _schema(TypeAdapter(ExactDecimal))
_INTEGER_SCHEMA = _schema(TypeAdapter(int))

SLOTS: tuple[Slot, ...] = (
    Slot(
        "universe",
        "Which symbols are in the universe?",
        "The list of symbols this agent may look at and trade. Orders for anything outside it "
        "refuse before they reach the broker.",
        "symbol list",
        _schema(TypeAdapter(list[str])),
        _validate_universe,
        ALL_KINDS,
    ),
    Slot(
        "cadence",
        "What cadence will the agent use?",
        "How often the runtime evaluates the rules on market days. Nothing happens between "
        "evaluations.",
        "cadence",
        _schema(TypeAdapter(Cadence)),
        _validate_cadence,
        ALL_KINDS,
    ),
    Slot(
        "kind",
        "Is this a rule agent or a model-driven agent?",
        "Rule agents run only the structured conditions written here. Model-driven agents run "
        "your instructions through your connected model inside the same limits.",
        "agent kind",
        _schema(TypeAdapter(AgentKind)),
        _validate_kind,
        ALL_KINDS,
    ),
    Slot(
        "rules",
        "What are the structured conditions and actions for each rule?",
        "Each rule is a condition the runtime checks and the action it takes when the condition "
        "is true. The runtime evaluates them exactly as written.",
        "rule list",
        _schema(TypeAdapter(list[Rule])),
        _validate_rules,
        RULE_ONLY,
    ),
    Slot(
        "provider",
        "Which connected provider will this agent use?",
        "The connected model provider that runs this agent's instructions on your server.",
        "provider",
        _schema(TypeAdapter(Provider)),
        _validate_provider,
        MODEL_ONLY,
    ),
    Slot(
        "model",
        "Which model identifier will this agent use?",
        "The exact model identifier from that provider. It is recorded with every decision the "
        "agent makes.",
        "model identifier",
        _schema(TypeAdapter(str)),
        _validate_model,
        MODEL_ONLY,
    ),
    Slot(
        "instructions",
        "What exact instructions will this model-driven agent run?",
        "The text your model receives on every evaluation. It can only act within the limits "
        "recorded here.",
        "instructions",
        _schema(TypeAdapter(str)),
        _validate_instructions,
        MODEL_ONLY,
    ),
    Slot(
        "cage.max_position_pct",
        "What is the maximum position percentage?",
        "The largest share of the account any single position may reach. The runtime refuses "
        "orders that would exceed it, whatever the rules say.",
        "exact decimal",
        _DECIMAL_SCHEMA,
        _validate_percentage,
        ALL_KINDS,
    ),
    Slot(
        "cage.max_positions",
        "What is the maximum position count?",
        "How many positions the agent may keep open at once. Further opening orders refuse until "
        "one closes.",
        "integer",
        _INTEGER_SCHEMA,
        _validate_position_count,
        ALL_KINDS,
    ),
    Slot(
        "cage.max_order_notional",
        "What is the maximum order amount?",
        "The largest amount a single order may be worth. Larger orders refuse.",
        "exact decimal",
        _DECIMAL_SCHEMA,
        _validate_order_amount,
        ALL_KINDS,
    ),
    Slot(
        "cage.max_daily_drawdown_pct",
        "What is the maximum daily drawdown percentage?",
        "The loss within one day, measured against that day's opening equity, after "
        "which the runtime places no new opening orders for the rest of the session. "
        "The agent keeps evaluating on every tick, closing orders still go through, and "
        "the limit resets with the next session.",
        "exact decimal",
        _DECIMAL_SCHEMA,
        _validate_percentage,
        ALL_KINDS,
    ),
    Slot(
        "cage.allowed_session",
        "Which session is allowed?",
        "The part of the trading day during which the agent may place orders.",
        "session",
        _schema(TypeAdapter(Session)),
        _validate_session,
        ALL_KINDS,
    ),
    Slot(
        "approval",
        "Which approval mode will this agent use?",
        "Whether you confirm each order before it is sent, or grant standing approval within "
        "these limits. You choose this again when you adopt the agent.",
        "approval mode",
        _schema(TypeAdapter(ApprovalMode)),
        _validate_approval,
        ALL_KINDS,
    ),
)


def meaningful_slots(kind: AgentKind) -> tuple[Slot, ...]:
    """The required slots for one shape, in interview order."""
    return tuple(slot for slot in SLOTS if kind in slot.kinds)


def slot_by_name(name: str) -> Slot:
    """Resolve a persisted slot name without accepting an unknown field."""
    for slot in SLOTS:
        if slot.name == name:
            return slot
    raise KeyError(name)
