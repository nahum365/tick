"""Shared base model and scalar types for the strategy spec.

Every model in the spec package is frozen and forbids unknown fields. Both
properties serve the immutability primitive in `canonical.py`: a spec that
silently dropped an unrecognised key, or that could be mutated after
validation, would have a `spec_id` that does not describe what the runtime
actually executes.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated

from pydantic import BaseModel, BeforeValidator, ConfigDict


class SpecModel(BaseModel):
    """Frozen, closed base for every spec node."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def _reject_binary_float(value: object) -> object:
    """Refuse a binary float where an exact decimal is required.

    Money and percentages are `Decimal`, never `float` (CLAUDE.md). The spec
    loader parses JSON numbers with `parse_float=Decimal`, so a float can only
    reach here from Python code — where accepting it would quietly substitute
    a binary approximation for the number the caller wrote.
    """
    if isinstance(value, float):
        raise ValueError(
            f"{value!r} is a binary float; write an exact decimal instead "
            f'(a JSON string like "12.50", or Decimal("12.50"))'
        )
    return value


#: An exact decimal. Money amounts and percentages both use it.
ExactDecimal = Annotated[Decimal, BeforeValidator(_reject_binary_float)]
