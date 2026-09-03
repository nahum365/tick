"""The strategy spec itself: cadence, actions, the cage, rules, and the document.

The cage is the part with authority (CLAUDE.md invariant 3). Every field of it
is REQUIRED — a cage with a default is a cage nobody chose, and the whole
claim of the product is that the limits an agent runs under were set on
purpose and are recorded.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from .base import ExactDecimal, SpecModel
from .conditions import MAX_CONDITION_DEPTH, Condition, condition_depth

#: Symbols as Robinhood writes them: up to five capitals, optional class
#: suffix (`XYZ.A`). Lowercase is refused rather than upcased, so the document
#: on disk is the document the runtime executes. The examples here and in the
#: refusal below are placeholders: Tick names no real security anywhere the
#: product surfaces, and a validation message is a surface (CLAUDE.md
#: invariant 14).
SYMBOL_PATTERN = re.compile(r"^[A-Z]{1,5}(\.[A-Z]{1,2})?$")

#: Rule ids are slugs: they appear in the record, in notifications, and in
#: validation messages, so they stay short and typeable.
RULE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")

#: A regular US equities session is 390 minutes; a cadence cannot be coarser
#: than the session it runs in.
MINUTES_IN_REGULAR_SESSION = 390

MAX_NAME_LENGTH = 120


# --------------------------------------------------------------------------
# Cadence
# --------------------------------------------------------------------------


class DailyOpen(SpecModel):
    """Evaluate once, just after the regular session opens."""

    kind: Literal["daily_open"] = "daily_open"


class DailyClose(SpecModel):
    """Evaluate once, shortly before the regular session closes."""

    kind: Literal["daily_close"] = "daily_close"


class EveryNMinutes(SpecModel):
    """Evaluate every `n` minutes while the session is open."""

    kind: Literal["every_n_minutes"] = "every_n_minutes"
    n: int

    @model_validator(mode="after")
    def _check_n(self) -> EveryNMinutes:
        if self.n < 1 or self.n > MINUTES_IN_REGULAR_SESSION:
            raise ValueError(
                f"every_n_minutes({self.n}): n must be between 1 and {MINUTES_IN_REGULAR_SESSION}"
            )
        return self


Cadence = Annotated[DailyOpen | DailyClose | EveryNMinutes, Field(discriminator="kind")]

CADENCE_TYPES: tuple[type[SpecModel], ...] = (DailyOpen, DailyClose, EveryNMinutes)
CADENCE_KINDS: frozenset[str] = frozenset(
    model.model_fields["kind"].default for model in CADENCE_TYPES
)


# --------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------


class Side(StrEnum):
    """The two sides an order can have, and there is no third.

    Tick is long-only (CLAUDE.md invariant 9). A Robinhood Agentic account
    places long equity orders: a buy opens or adds to a position, a sell closes
    one. There is no third member for opening a short or covering one, no
    negative quantity anywhere below this type, and no path that turns an
    oversized sell into a short — the engine refuses it whole when it sizes it
    (`engine/evaluate.py`), and the paper broker and the Robinhood adapter each
    refuse it again against their own view of what is held.

    The enum is where that is said because a third member would silently make
    every one of those checks incomplete: they all ask "is this a sell, and is
    it bigger than the position?", which is the right question for exactly two
    sides.
    """

    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    """Prototype orders are market orders only.

    The field is still required. An order type is meaning-bearing, and the day
    a limit order exists a spec written today must not silently acquire it.
    """

    MARKET = "market"


class SharesSize(SpecModel):
    """A whole number of shares."""

    kind: Literal["shares"] = "shares"
    shares: int

    @model_validator(mode="after")
    def _check(self) -> SharesSize:
        if self.shares < 1:
            raise ValueError(f"shares({self.shares}): shares must be >= 1")
        return self

    def label(self) -> str:
        return f"{self.shares} shares"


class NotionalSize(SpecModel):
    """A dollar amount."""

    kind: Literal["notional"] = "notional"
    notional: ExactDecimal

    @model_validator(mode="after")
    def _check(self) -> NotionalSize:
        if self.notional <= 0:
            raise ValueError(f"notional({self.notional}): notional must be > 0")
        return self

    def label(self) -> str:
        return f"${self.notional} notional"


class PctOfEquitySize(SpecModel):
    """A percentage of account equity, in (0, 100]."""

    kind: Literal["pct_of_equity"] = "pct_of_equity"
    pct_of_equity: ExactDecimal

    @model_validator(mode="after")
    def _check(self) -> PctOfEquitySize:
        if self.pct_of_equity <= 0 or self.pct_of_equity > 100:
            raise ValueError(f"pct_of_equity({self.pct_of_equity}): must be > 0 and <= 100")
        return self

    def label(self) -> str:
        return f"{self.pct_of_equity}% of equity"


class AllSize(SpecModel):
    """The whole position (sell) or the whole buying power (buy)."""

    kind: Literal["all"] = "all"

    def label(self) -> str:
        return "all"


Size = Annotated[SharesSize | NotionalSize | PctOfEquitySize | AllSize, Field(discriminator="kind")]

SIZE_TYPES: tuple[type[SpecModel], ...] = (SharesSize, NotionalSize, PctOfEquitySize, AllSize)
SIZE_KINDS: frozenset[str] = frozenset(model.model_fields["kind"].default for model in SIZE_TYPES)


class Action(SpecModel):
    """What a rule does when it fires.

    Exactly one sizing is expressible because `size` is a tagged union: there
    is no shape in which two sizings are set, and none in which none is.
    """

    side: Side
    size: Size
    order_type: OrderType

    def label(self) -> str:
        return f"{self.side.value} {self.size.label()}"


# --------------------------------------------------------------------------
# The cage
# --------------------------------------------------------------------------


class Session(StrEnum):
    """When the runtime may place orders. Regular hours only for now."""

    REGULAR_HOURS = "regular_hours"


class Cage(SpecModel):
    """The deterministic limits the runtime enforces, whatever the rules say.

    Every field is required. These are the numbers that make a model agent
    supervisable and a spec agent bounded, so none of them may be inherited
    from a library default that nobody read.
    """

    max_position_pct: ExactDecimal
    max_positions: int
    max_order_notional: ExactDecimal
    max_daily_drawdown_pct: ExactDecimal
    allowed_session: Session

    @model_validator(mode="after")
    def _check_bounds(self) -> Cage:
        if self.max_position_pct <= 0 or self.max_position_pct > 100:
            raise ValueError(
                f"cage.max_position_pct({self.max_position_pct}): must be > 0 and <= 100"
            )
        if self.max_positions < 1:
            raise ValueError(f"cage.max_positions({self.max_positions}): must be >= 1")
        if self.max_order_notional <= 0:
            raise ValueError(f"cage.max_order_notional({self.max_order_notional}): must be > 0")
        if self.max_daily_drawdown_pct <= 0 or self.max_daily_drawdown_pct > 100:
            raise ValueError(
                f"cage.max_daily_drawdown_pct({self.max_daily_drawdown_pct}): "
                f"must be > 0 and <= 100"
            )
        return self


# --------------------------------------------------------------------------
# Rules and the document
# --------------------------------------------------------------------------


class Rule(SpecModel):
    """One `when → then` pair. The id is how the record and notifications
    refer to it, so it is unique within a spec."""

    id: str
    when: Condition
    then: Action

    @model_validator(mode="after")
    def _check(self) -> Rule:
        if not RULE_ID_PATTERN.match(self.id):
            raise ValueError(
                f"rule id {self.id!r}: must be 1-40 characters of a-z, 0-9, "
                f"'_' or '-', starting with a letter or digit"
            )
        depth = condition_depth(self.when)
        if depth > MAX_CONDITION_DEPTH:
            raise ValueError(
                f"rule {self.id!r}: condition nests {depth} levels deep; "
                f"the limit is {MAX_CONDITION_DEPTH}"
            )
        return self


class StrategySpec(SpecModel):
    """A complete, deterministic strategy an agent executes.

    Two specs that mean the same thing hash to the same `spec_id`; any change
    at all produces a different one (see `canonical.py`). That is the
    immutability primitive the record in slice 03 chains against.
    """

    name: str
    version: int
    universe: list[str]
    cadence: Cadence
    rules: list[Rule]
    cage: Cage

    @field_validator("name")
    @classmethod
    def _check_name(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("name must not start or end with whitespace")
        if not value:
            raise ValueError("name must not be empty")
        if len(value) > MAX_NAME_LENGTH:
            raise ValueError(f"name must be at most {MAX_NAME_LENGTH} characters")
        if any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise ValueError("name must not contain control characters")
        return value

    @field_validator("version")
    @classmethod
    def _check_version(cls, value: int) -> int:
        if value < 1:
            raise ValueError(f"version({value}): must be >= 1")
        return value

    @field_validator("universe")
    @classmethod
    def _check_universe(cls, value: list[str]) -> list[str]:
        """Refuse anything malformed, refuse duplicates, then sort.

        Sorting is a meaning-preserving normalisation of a set, and it means
        two specs that hold the same symbols hash alike however they were
        typed. Duplicates are REFUSED rather than silently collapsed: quietly
        rewriting the author's document is how a spec stops describing what
        the runtime runs.
        """
        if not value:
            raise ValueError("universe must name at least one symbol")
        seen: set[str] = set()
        for symbol in value:
            if not SYMBOL_PATTERN.match(symbol):
                raise ValueError(
                    f"universe symbol {symbol!r}: must be 1-5 capital letters with an "
                    f"optional class suffix (e.g. 'XYZ', 'XYZ.A')"
                )
            if symbol in seen:
                raise ValueError(f"universe lists {symbol!r} more than once")
            seen.add(symbol)
        return sorted(value)

    @field_validator("rules")
    @classmethod
    def _check_rules(cls, value: list[Rule]) -> list[Rule]:
        if not value:
            raise ValueError("a spec must have at least one rule")
        seen: set[str] = set()
        for rule in value:
            if rule.id in seen:
                raise ValueError(f"rules use the id {rule.id!r} more than once")
            seen.add(rule.id)
        return value

    @model_validator(mode="after")
    def _check_rules_against_cage(self) -> StrategySpec:
        """Refuse a rule the cage could never let run as written.

        This catches only the provably contradictory cases — a single order
        that on its own exceeds a ceiling. Everything that depends on live
        state (an add that would push an existing position over the cap) is
        the runtime's job, enforced per order. Catching the static half here
        means the contradiction is reported to the author, not silently
        clamped at 09:31.
        """
        for rule in self.rules:
            size = rule.then.size
            if isinstance(size, NotionalSize) and size.notional > self.cage.max_order_notional:
                raise ValueError(
                    f"rule {rule.id!r}: orders ${size.notional} but "
                    f"cage.max_order_notional is ${self.cage.max_order_notional}"
                )
            if (
                isinstance(size, PctOfEquitySize)
                and rule.then.side is Side.BUY
                and size.pct_of_equity > self.cage.max_position_pct
            ):
                raise ValueError(
                    f"rule {rule.id!r}: buys {size.pct_of_equity}% of equity but "
                    f"cage.max_position_pct is {self.cage.max_position_pct}%"
                )
        return self

    def rule(self, rule_id: str) -> Rule:
        """The rule with this id. Raises KeyError if there is none."""
        for candidate in self.rules:
            if candidate.id == rule_id:
                return candidate
        raise KeyError(rule_id)
