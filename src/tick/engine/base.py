"""Shared base model, scalars, and the arithmetic context of the engine.

Two things here are load-bearing.

**Every engine value is a frozen, closed model.** A decision, an intent, a
quote and a refusal all end up in the record (slice 03), and a value that could
be mutated after it was recorded would make the record describe something that
never happened.

**Arithmetic runs in an explicit `decimal` context.** `Decimal` division reads
the *thread's* context, which any host application can change. The engine's
claim is that the same spec against the same data produces the same decisions
on any machine, so it does not inherit that setting: every computed number is
produced inside `ENGINE_CONTEXT` and rounded to `QUANTUM`.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Context, Decimal, localcontext

from pydantic import BaseModel, ConfigDict

from tick.spec import ExactDecimal

__all__ = [
    "CENTS",
    "ENGINE_CONTEXT",
    "EngineModel",
    "ExactDecimal",
    "QUANTUM",
    "engine_arithmetic",
    "quantize_money",
    "quantize_value",
]

#: The arithmetic context every computed engine number is produced in. Wide
#: enough that no intermediate is lost, fixed so no host can change it.
ENGINE_CONTEXT = Context(prec=34, rounding=ROUND_HALF_EVEN)

#: Computed indicator values are rounded here, so the number in the record is
#: exactly the number the comparison was made on, byte for byte, everywhere.
QUANTUM = Decimal("0.00000001")

#: Money reported to a human is cents.
CENTS = Decimal("0.01")


class EngineModel(BaseModel):
    """Frozen, closed base for every engine value."""

    model_config = ConfigDict(extra="forbid", frozen=True)


@contextmanager
def engine_arithmetic() -> Iterator[None]:
    """Run a block in `ENGINE_CONTEXT` instead of the thread's context."""
    with localcontext(ENGINE_CONTEXT):
        yield


def quantize_value(value: Decimal) -> Decimal:
    """Round a computed indicator value to `QUANTUM`, half to even."""
    return value.quantize(QUANTUM, rounding=ROUND_HALF_EVEN)


def quantize_money(value: Decimal) -> Decimal:
    """Round a money amount to cents, half up — the way a statement reads."""
    return value.quantize(CENTS, rounding=ROUND_HALF_UP)
