"""Conditions — the closed boolean grammar a rule fires on.

Four node kinds and nothing else: a comparison, and the three combinators
`all_of` / `any_of` / `not`. There is no expression string anywhere in the
spec, which is the point: a spec agent's decision is reproducible from the
document plus the market data, and a reviewer can read the whole of what an
agent is allowed to consider.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, model_validator

from .base import SpecModel
from .indicators import Indicator, IndicatorNode, Operand

#: How deeply conditions may nest inside one rule. A limit exists so a spec
#: stays reviewable by a human, which is the whole reason it is a document.
MAX_CONDITION_DEPTH = 8


class ComparisonOp(StrEnum):
    """The closed set of comparison operators."""

    GT = ">"
    LT = "<"
    GTE = ">="
    LTE = "<="
    CROSSES_ABOVE = "crosses_above"
    CROSSES_BELOW = "crosses_below"

    @property
    def is_cross(self) -> bool:
        return self in (ComparisonOp.CROSSES_ABOVE, ComparisonOp.CROSSES_BELOW)


class Compare(SpecModel):
    """`left op right`, where left is always something the runtime measured."""

    kind: Literal["compare"] = "compare"
    left: Indicator
    op: ComparisonOp
    right: Operand

    @model_validator(mode="after")
    def _check_cross_operands(self) -> Compare:
        """A cross needs a per-bar history on both sides (a constant counts).

        `price crosses_above sma(50)` and `price crosses_above number(200)` are
        both meaningful. `cash crosses_above number(500)` is not: the runtime
        keeps no per-bar history of cash, so there is no previous bar to have
        crossed from, and answering it would mean inventing one.
        """
        if not self.op.is_cross:
            return self
        if not self.left.is_series:
            raise ValueError(
                f"{self.op.value}: {self.left.label()} has no per-bar history to cross"
            )
        if not self.right.is_series and self.right.kind != "number":
            raise ValueError(
                f"{self.op.value}: {self.right.label()} has no per-bar history to cross"
            )
        return self


class AllOf(SpecModel):
    """True when every listed condition is true."""

    kind: Literal["all_of"] = "all_of"
    of: list[Condition] = Field(min_length=1)


class AnyOf(SpecModel):
    """True when at least one listed condition is true."""

    kind: Literal["any_of"] = "any_of"
    of: list[Condition] = Field(min_length=1)


class Not(SpecModel):
    """True when the wrapped condition is false."""

    kind: Literal["not"] = "not"
    of: Condition


Condition = Annotated[Compare | AllOf | AnyOf | Not, Field(discriminator="kind")]

CONDITION_TYPES: tuple[type[SpecModel], ...] = (Compare, AllOf, AnyOf, Not)

#: Every legal `kind` for a condition node, derived from the models.
CONDITION_KINDS: frozenset[str] = frozenset(
    model.model_fields["kind"].default for model in CONDITION_TYPES
)

AllOf.model_rebuild()
AnyOf.model_rebuild()
Not.model_rebuild()


def condition_depth(condition: Compare | AllOf | AnyOf | Not) -> int:
    """Nesting depth of a condition tree; a bare comparison is depth 1."""
    if isinstance(condition, Compare):
        return 1
    if isinstance(condition, Not):
        return 1 + condition_depth(condition.of)
    return 1 + max(condition_depth(child) for child in condition.of)


def indicators_in(condition: Compare | AllOf | AnyOf | Not) -> list[IndicatorNode]:
    """Every indicator and literal referenced by a condition tree, in order.

    The engine uses this to know exactly which data a rule needs before it may
    evaluate — a missing input refuses the rule rather than defaulting it.
    """
    if isinstance(condition, Compare):
        return [condition.left, condition.right]
    if isinstance(condition, Not):
        return indicators_in(condition.of)
    found: list[IndicatorNode] = []
    for child in condition.of:
        found.extend(indicators_in(child))
    return found
