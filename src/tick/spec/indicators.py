"""Indicators — the leaves of the condition grammar.

The grammar is CLOSED: there is no free-form expression, no eval, no user
code. Every quantity a rule may look at is one of the tagged models below,
each with typed parameters. That closure is what makes a spec reproducible
and what lets the engine (slice 02) know exactly which data it must fetch
before it may decide anything.

Indicators are evaluated in a per-symbol context supplied by the engine: in a
spec whose universe is ["XYZ", "ABCD"], `price` means "the price of the
symbol currently under evaluation". The grammar deliberately has no symbol
override yet; adding one is a wire change, not a default.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from .base import ExactDecimal, SpecModel


class _Indicator(SpecModel):
    """Common behaviour for indicator nodes."""

    def label(self) -> str:
        """A short human name used verbatim in validation messages."""
        raise NotImplementedError  # pragma: no cover - abstract

    @property
    def is_series(self) -> bool:
        """True when the indicator has a per-bar history.

        `crosses_above` / `crosses_below` compare this bar against the one
        before it, so they are only meaningful over a series.
        """
        raise NotImplementedError  # pragma: no cover - abstract


def _positive_window(value: int, name: str, param: str) -> None:
    if value < 1:
        raise ValueError(f"{name}({value}): {param} must be >= 1")


class Price(_Indicator):
    """The latest trade price of the symbol under evaluation."""

    kind: Literal["price"] = "price"

    def label(self) -> str:
        return "price"

    @property
    def is_series(self) -> bool:
        return True


class Sma(_Indicator):
    """Simple moving average of the last `n` bars."""

    kind: Literal["sma"] = "sma"
    n: int

    @model_validator(mode="after")
    def _check_window(self) -> Sma:
        _positive_window(self.n, "sma", "n")
        return self

    def label(self) -> str:
        return f"sma({self.n})"

    @property
    def is_series(self) -> bool:
        return True


class Ema(_Indicator):
    """Exponential moving average over an `n`-bar window."""

    kind: Literal["ema"] = "ema"
    n: int

    @model_validator(mode="after")
    def _check_window(self) -> Ema:
        _positive_window(self.n, "ema", "n")
        return self

    def label(self) -> str:
        return f"ema({self.n})"

    @property
    def is_series(self) -> bool:
        return True


class ChangePct(_Indicator):
    """Percent change over the last `n_bars` bars, as a percentage (5 == +5%)."""

    kind: Literal["change_pct"] = "change_pct"
    n_bars: int

    @model_validator(mode="after")
    def _check_window(self) -> ChangePct:
        _positive_window(self.n_bars, "change_pct", "n_bars")
        return self

    def label(self) -> str:
        return f"change_pct({self.n_bars})"

    @property
    def is_series(self) -> bool:
        return True


class PositionQty(_Indicator):
    """Shares currently held of the symbol under evaluation."""

    kind: Literal["position_qty"] = "position_qty"

    def label(self) -> str:
        return "position_qty"

    @property
    def is_series(self) -> bool:
        return False


class PositionPctOfEquity(_Indicator):
    """That position's market value as a percentage of account equity."""

    kind: Literal["position_pct_of_equity"] = "position_pct_of_equity"

    def label(self) -> str:
        return "position_pct_of_equity"

    @property
    def is_series(self) -> bool:
        return False


class Cash(_Indicator):
    """Settled cash available in the Agentic account."""

    kind: Literal["cash"] = "cash"

    def label(self) -> str:
        return "cash"

    @property
    def is_series(self) -> bool:
        return False


class DayOfWeek(_Indicator):
    """ISO weekday of the current tick in ET: Monday == 1 ... Sunday == 7."""

    kind: Literal["day_of_week"] = "day_of_week"

    def label(self) -> str:
        return "day_of_week"

    @property
    def is_series(self) -> bool:
        return False


class NumberLiteral(_Indicator):
    """A constant written into the spec.

    Only legal on the right-hand side of a comparison, so a spec cannot say
    `5 > 3`; a comparison always starts from something the runtime measured.
    """

    kind: Literal["number"] = "number"
    value: ExactDecimal

    def label(self) -> str:
        return f"number({self.value})"

    @property
    def is_series(self) -> bool:
        return False


#: Public name for the indicator base class, for type annotations elsewhere.
IndicatorNode = _Indicator

INDICATOR_TYPES: tuple[type[_Indicator], ...] = (
    Price,
    Sma,
    Ema,
    ChangePct,
    PositionQty,
    PositionPctOfEquity,
    Cash,
    DayOfWeek,
)

#: Every legal `kind` for a measured indicator, derived from the models so the
#: two can never drift apart.
INDICATOR_KINDS: frozenset[str] = frozenset(
    model.model_fields["kind"].default for model in INDICATOR_TYPES
)

#: Indicator kinds plus the constant literal — everything an operand may be.
OPERAND_KINDS: frozenset[str] = INDICATOR_KINDS | {NumberLiteral.model_fields["kind"].default}

Indicator = Annotated[
    Price | Sma | Ema | ChangePct | PositionQty | PositionPctOfEquity | Cash | DayOfWeek,
    Field(discriminator="kind"),
]

Operand = Annotated[
    Price
    | Sma
    | Ema
    | ChangePct
    | PositionQty
    | PositionPctOfEquity
    | Cash
    | DayOfWeek
    | NumberLiteral,
    Field(discriminator="kind"),
]
