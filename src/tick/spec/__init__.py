"""The strategy spec — Tick's deterministic contract.

A spec is a document: a universe, a cadence, a list of `when → then` rules
over a closed grammar, and a cage of hard limits. Nothing in it is code and
nothing in it is free-form, so the same spec against the same data produces
the same decisions, and `spec_id` names exactly what an agent will do.

    from tick.spec import load_spec, spec_id

    spec = load_spec("strategies/dip-buyer.json")
    print(spec_id(spec))

CLAUDE.md invariants this package carries:

- **3, the spec decides.** The grammar here is the only shape an LLM's output
  can take before it may reach a broker; there is no escape hatch in it.
- **5, no number is fabricated.** Money is `Decimal` and a binary float is
  refused, so nothing is silently rounded on the way in.
- **No silent meaning-bearing defaults.** Every field of `Cage`, both fields
  of `Action`, and the cadence are required.
"""

from __future__ import annotations

from .base import ExactDecimal, SpecModel
from .canonical import (
    canonical_bytes,
    canonical_dumps,
    canonical_encode,
    canonical_json,
    sha256_hex,
    short_spec_id,
    spec_id,
)
from .conditions import (
    CONDITION_KINDS,
    MAX_CONDITION_DEPTH,
    AllOf,
    AnyOf,
    Compare,
    ComparisonOp,
    Condition,
    Not,
    condition_depth,
    indicators_in,
)
from .errors import SpecError, SpecFormatError, SpecValidationError
from .indicators import (
    INDICATOR_KINDS,
    OPERAND_KINDS,
    Cash,
    ChangePct,
    DayOfWeek,
    Ema,
    Indicator,
    IndicatorNode,
    NumberLiteral,
    Operand,
    PositionPctOfEquity,
    PositionQty,
    Price,
    Sma,
)
from .io import (
    MAX_RAW_DEPTH,
    dump_spec,
    load_spec,
    load_spec_file,
    loads_spec,
    parse_spec,
)
from .strategy import (
    CADENCE_KINDS,
    SIZE_KINDS,
    SYMBOL_PATTERN,
    Action,
    AllSize,
    Cadence,
    Cage,
    DailyClose,
    DailyOpen,
    EveryNMinutes,
    NotionalSize,
    OrderType,
    PctOfEquitySize,
    Rule,
    Session,
    SharesSize,
    Side,
    Size,
    StrategySpec,
)

__all__ = [
    "CADENCE_KINDS",
    "CONDITION_KINDS",
    "INDICATOR_KINDS",
    "MAX_CONDITION_DEPTH",
    "MAX_RAW_DEPTH",
    "OPERAND_KINDS",
    "SIZE_KINDS",
    "SYMBOL_PATTERN",
    "Action",
    "AllOf",
    "AllSize",
    "AnyOf",
    "Cadence",
    "Cage",
    "Cash",
    "ChangePct",
    "Compare",
    "ComparisonOp",
    "Condition",
    "DailyClose",
    "DailyOpen",
    "DayOfWeek",
    "Ema",
    "EveryNMinutes",
    "ExactDecimal",
    "Indicator",
    "IndicatorNode",
    "Not",
    "NotionalSize",
    "NumberLiteral",
    "Operand",
    "OrderType",
    "PctOfEquitySize",
    "PositionPctOfEquity",
    "PositionQty",
    "Price",
    "Rule",
    "Session",
    "SharesSize",
    "Side",
    "Size",
    "Sma",
    "SpecError",
    "SpecFormatError",
    "SpecModel",
    "SpecValidationError",
    "StrategySpec",
    "canonical_bytes",
    "canonical_dumps",
    "canonical_encode",
    "canonical_json",
    "condition_depth",
    "dump_spec",
    "indicators_in",
    "load_spec",
    "load_spec_file",
    "loads_spec",
    "parse_spec",
    "sha256_hex",
    "short_spec_id",
    "spec_id",
]
