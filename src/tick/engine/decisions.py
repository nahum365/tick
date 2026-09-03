"""What a tick produces: an evaluated decision per rule, per symbol.

A `Decision` is written whether or not the rule fired, and it carries the
numbers the condition was judged on. That is what makes the record (slice 03)
worth keeping: "did not fire" with `sma(20) = 184.21, sma(50) = 191.04` beside
it is a reviewable statement, while a bare "did not fire" is a claim the user
has to take on trust.

Exactly one of three things follows a decision:

- **an `OrderIntent`** — the rule fired and the order it asks for is
  expressible: whole shares, a real price, and the provenance of that price;
- **a `Refusal`** — something needed was unavailable, or the order would break
  a rule of the product (long-only, a size that rounds to nothing). A refusal
  is a *value*, carried forward, reported, recorded. It is never a zero, and
  never a smaller order that nobody asked for;
- **neither** — the condition was evaluated and was false.

An intent carries no authority. The cage (`engine/cage.py`) decides what may
be placed, and it takes intents from any source — a spec rule today, a caged
model agent later — precisely so the limits do not depend on who proposed the
order.
"""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum

from pydantic import AwareDatetime, model_validator

from tick.spec import Side

from .base import EngineModel, ExactDecimal
from .market import Unavailable

__all__ = [
    "Decision",
    "EvaluatedValue",
    "OrderIntent",
    "Refusal",
    "RefusalCode",
]


class RefusalCode(StrEnum):
    """Why an evaluated rule produced no order. Every one has served words."""

    QUOTE_UNAVAILABLE = "quote_unavailable"
    BARS_UNAVAILABLE = "bars_unavailable"
    INDICATOR_UNAVAILABLE = "indicator_unavailable"
    EQUITY_UNAVAILABLE = "equity_unavailable"
    SELL_EXCEEDS_POSITION = "sell_exceeds_position"
    NO_POSITION_TO_SELL = "no_position_to_sell"
    SIZE_ROUNDS_TO_ZERO = "size_rounds_to_zero"
    #: Added with model agents (slice 07). A rule can only propose an order over
    #: its own spec's universe, so neither of these can arise from one; a model
    #: proposes whatever it likes, and both are how the runtime says no.
    SYMBOL_OUTSIDE_UNIVERSE = "symbol_outside_universe"
    MODEL_OUTPUT_INVALID = "model_output_invalid"


class EvaluatedValue(EngineModel):
    """One number a condition was judged on, or the reason there was none.

    `label` is the grammar's own name for the operand (`sma(20)`, `cash`), with
    `@prev` appended for the previous bar's value of a crossing comparison.
    """

    label: str
    value: ExactDecimal | None
    unavailable: Unavailable | None

    @model_validator(mode="after")
    def _check(self) -> EvaluatedValue:
        if (self.value is None) == (self.unavailable is None):
            raise ValueError(
                f"{self.label}: an evaluated value carries a number or an "
                f"unavailability, never both and never neither"
            )
        return self

    @classmethod
    def of(cls, label: str, result: Decimal | Unavailable) -> EvaluatedValue:
        if isinstance(result, Unavailable):
            return cls(label=label, value=None, unavailable=result)
        return cls(label=label, value=result, unavailable=None)


class OrderIntent(EngineModel):
    """A proposed order: whole shares, at a priced estimate, with its provenance.

    `source` names who proposed it (`"rule:golden-cross"`, and later
    `"model:<id>"`), because the cage rejects intents by name and a rejection
    nobody can trace is not a report.

    `est_notional` is an *estimate* — `qty × est_price` at the quote the engine
    read. The fill price is whatever the broker actually gets, and the record
    keeps both.
    """

    source: str
    symbol: str
    side: Side
    qty: int
    est_price: ExactDecimal
    est_notional: ExactDecimal
    price_asof: AwareDatetime
    price_source: str
    reason: str

    @model_validator(mode="after")
    def _check(self) -> OrderIntent:
        if self.qty < 1:
            raise ValueError(
                f"{self.source}: an intent for {self.qty} shares is not an order; "
                f"a size that rounds to nothing refuses instead"
            )
        if self.est_price <= 0:
            raise ValueError(f"{self.source}: est_price ({self.est_price}) must be > 0")
        if self.est_notional <= 0:
            raise ValueError(f"{self.source}: est_notional ({self.est_notional}) must be > 0")
        for name in ("source", "symbol", "price_source", "reason"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"an intent must carry a {name}")
        return self

    def describe(self) -> str:
        """Mechanical past-tense-ready wording: `buy 12 XYZ at $184.20`."""
        return f"{self.side.value} {self.qty} {self.symbol} at ${self.est_price}"


class Refusal(EngineModel):
    """A rule that could have acted and did not, with a reason a human can read."""

    source: str
    symbol: str
    code: RefusalCode
    reason: str

    @model_validator(mode="after")
    def _check(self) -> Refusal:
        if not self.reason.strip():
            raise ValueError(f"{self.code.value}: a refusal must say why")
        return self

    def __str__(self) -> str:
        return f"{self.source} refused on {self.symbol}: {self.reason}"


class Decision(EngineModel):
    """One rule evaluated against one symbol at one moment."""

    rule_id: str
    symbol: str
    at: AwareDatetime
    fired: bool
    values: tuple[EvaluatedValue, ...]
    intent: OrderIntent | None
    refusal: Refusal | None

    @model_validator(mode="after")
    def _check(self) -> Decision:
        if self.intent is not None and self.refusal is not None:
            raise ValueError(
                f"rule {self.rule_id!r} on {self.symbol}: a decision carries an "
                f"intent or a refusal, never both"
            )
        if self.intent is not None and not self.fired:
            raise ValueError(
                f"rule {self.rule_id!r} on {self.symbol}: an intent exists only "
                f"where the condition fired"
            )
        return self

    @property
    def acted(self) -> bool:
        return self.intent is not None
