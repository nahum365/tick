"""The explanation a person reads before they let a spec run.

It is rendered from the VALIDATED spec, deterministically, not written by the
model. That is the load-bearing decision in this module and it is worth stating
plainly: a model that writes its own explanation can describe a rule it did not
emit, and the failure is invisible — the prose reads well, the document does
something else, and the user approved the prose. Rendering from the document
means the explanation cannot drift from what will run, because there is only
one source for both.

The second half, `what_it_cannot_know`, is the honesty half. It is derived from
the indicators the rule actually uses: a rule reading `price` cannot know why
the price moved; one reading `position_qty` in paper mode is reading a
simulation, not a brokerage; a `sell` cannot become a short. Every sentence
here is a mechanical fact about the runtime, never a judgement about the
strategy — Tick states what the machine can see and lets the person conclude.
"""

from __future__ import annotations

from dataclasses import dataclass

from tick.engine import describe_cadence
from tick.spec import (
    Action,
    AllOf,
    AnyOf,
    ChangePct,
    Compare,
    ComparisonOp,
    Ema,
    Not,
    NumberLiteral,
    Rule,
    Side,
    Sma,
    StrategySpec,
    indicators_in,
)

__all__ = ["RuleExplanation", "describe_action", "describe_condition", "explain"]


@dataclass(frozen=True, slots=True)
class RuleExplanation:
    """One rule, in words, with what it is blind to."""

    rule_id: str
    what_it_does: str
    what_it_cannot_know: tuple[str, ...]


_OPERATORS: dict[ComparisonOp, str] = {
    ComparisonOp.GT: "is above",
    ComparisonOp.LT: "is below",
    ComparisonOp.GTE: "is at or above",
    ComparisonOp.LTE: "is at or below",
    ComparisonOp.CROSSES_ABOVE: "crosses above",
    ComparisonOp.CROSSES_BELOW: "crosses below",
}

_OPERANDS: dict[str, str] = {
    "price": "the price",
    "position_qty": "the number of shares held",
    "position_pct_of_equity": "the position's share of account equity",
    "cash": "settled cash",
    "day_of_week": "the weekday (Monday is 1)",
}


def _operand_words(node: object) -> str:
    if isinstance(node, Sma):
        return f"its {node.n}-bar simple moving average"
    if isinstance(node, Ema):
        return f"its {node.n}-bar exponential moving average"
    if isinstance(node, ChangePct):
        return f"the percent change over {node.n_bars} bars"
    if isinstance(node, NumberLiteral):
        return format(node.value, "f")
    return _OPERANDS[node.kind]  # type: ignore[attr-defined]


def describe_condition(condition: Compare | AllOf | AnyOf | Not) -> str:
    """The `when` half of a rule, in words that match the document exactly."""
    if isinstance(condition, Compare):
        return (
            f"{_operand_words(condition.left)} {_OPERATORS[condition.op]} "
            f"{_operand_words(condition.right)}"
        )
    if isinstance(condition, Not):
        return f"it is not the case that {describe_condition(condition.of)}"
    joiner = " and " if isinstance(condition, AllOf) else " or "
    parts = [describe_condition(child) for child in condition.of]
    return "(" + joiner.join(parts) + ")"


def describe_action(action: Action) -> str:
    """The `then` half of a rule, in words."""
    verb = "buy" if action.side is Side.BUY else "sell"
    if action.size.kind == "all":
        whole = "the whole position" if action.side is Side.SELL else "with all buying power"
        return f"{verb} {whole}, as a {action.order_type.value} order"
    return f"{verb} {action.size.label()}, as a {action.order_type.value} order"


def _blind_spots(rule: Rule) -> tuple[str, ...]:
    """What this rule's own inputs cannot tell the runtime."""
    nodes = indicators_in(rule.when)
    kinds = {node.kind for node in nodes}  # type: ignore[attr-defined]
    spots = [
        "why anything moved. The runtime reads prices and this account, and nothing "
        "else: no news, no earnings, no filings, no analyst opinion.",
        "whether this is a good idea for you. Tick checks the spec against the cage "
        "you set; it forms no view about your goals.",
    ]
    if kinds & {"price", "sma", "ema", "change_pct"}:
        spots.append(
            "anything between the bars it is given. It sees the series it was handed, "
            "not every trade inside each bar."
        )
    if _has_cross(rule):
        spots.append(
            "which way the first bar of a series crossed. A cross compares this bar "
            "with the one before it, so the first bar can never fire this rule."
        )
    if kinds & {"position_qty", "position_pct_of_equity", "cash"}:
        spots.append(
            "positions or cash it was not shown. In paper mode these are the "
            "simulated account's numbers, not your brokerage's."
        )
    if rule.then.side is Side.SELL:
        spots.append(
            "how to sell what the account does not hold. Tick is long-only: this can "
            "only close a position, never open a short."
        )
    return tuple(spots)


def _has_cross(rule: Rule) -> bool:
    def walk(condition: Compare | AllOf | AnyOf | Not) -> bool:
        if isinstance(condition, Compare):
            return condition.op.is_cross
        if isinstance(condition, Not):
            return walk(condition.of)
        return any(walk(child) for child in condition.of)

    return walk(rule.when)


def explain(spec: StrategySpec) -> tuple[RuleExplanation, ...]:
    """One explanation per rule, in the spec's own order. Every rule, always."""
    cadence = describe_cadence(spec.cadence)
    universe = ", ".join(spec.universe)
    return tuple(
        RuleExplanation(
            rule_id=rule.id,
            what_it_does=(
                f"Checked {cadence}, for each of {universe}: when "
                f"{describe_condition(rule.when)}, {describe_action(rule.then)}."
            ),
            what_it_cannot_know=_blind_spots(rule),
        )
        for rule in spec.rules
    )
