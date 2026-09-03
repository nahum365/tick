"""Traceability: every symbol and every number in a spec came from the user.

This is the compiler's enforcement half, and the reason the product can say it
translates rather than authors. The prompt tells the model not to invent; this
module checks, deterministically, on the model's output, and turns anything it
cannot trace into a question to put back to the user.

The rule, exactly:

- **Every symbol** in `universe` must appear in the user's text (case-insensitive,
  on a word boundary).
- **Every number** anywhere in the validated spec — a threshold, a lookback, a
  size, a cadence, each of the four cage limits — must equal a number that
  appears in the user's text.

`version` is the single exemption, and it is structural rather than a choice: a
new spec is version 1, which is a fact about the document's format and not a
parameter of anyone's strategy.

The walk is REFLECTIVE, over the validated document, rather than a hand-written
list of fields. A hand-written list fails open — a numeric field added to the
grammar later would silently stop being checked — and failing open here means
shipping an invented number into a live order. Reflection fails the other way:
a new field with no question mapped to it still refuses, with a generic
question naming the field. That is annoying and safe, in that order.

**What this check is not.** It matches values, not roles: a text saying "at most
5 positions" traces a spec that used 5 as a lookback. Catching that needs the
model's own reasoning, which is exactly the thing this check exists not to
trust. It is a floor — it makes fabrication from nothing impossible, and leaves
misuse of the user's own numbers to the explanation the user reads.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from decimal import Decimal, InvalidOperation

from tick.spec import StrategySpec

__all__ = ["numbers_in", "questions_for_untraceable", "symbol_is_in"]

#: A number as a person writes one: 10, 1,000, 12.50, .5, $5,000, 20%.
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?|\.\d+")

#: Paths whose value is a property of the document format, not of the strategy.
_EXEMPT_PATHS: frozenset[str] = frozenset({"version"})

_CAGE_QUESTIONS: Mapping[str, str] = {
    "cage.max_position_pct": (
        "What is the most of your account any one position may be, as a percentage?"
    ),
    "cage.max_positions": "How many positions at most may this agent hold at once?",
    "cage.max_order_notional": "What is the largest dollar amount a single order may be?",
    "cage.max_daily_drawdown_pct": (
        "How far may your account fall in a day before this agent stops, as a percentage?"
    ),
    "cadence.n": "How often should this run? Say it in minutes, or say at the open or the close.",
}

_RULE_QUESTIONS: Mapping[str, str] = {
    "then.size.shares": "How many shares should the rule {rule!r} trade?",
    "then.size.notional": "How many dollars should the rule {rule!r} trade at a time?",
    "then.size.pct_of_equity": (
        "What percentage of your account should the rule {rule!r} trade at a time?"
    ),
}

_RULE_SUFFIX_QUESTIONS: Mapping[str, str] = {
    "value": "What number should the rule {rule!r} compare against?",
    "n": "How many bars should the lookback in the rule {rule!r} cover?",
    "n_bars": "How many bars should the lookback in the rule {rule!r} cover?",
}


def numbers_in(text: str) -> set[Decimal]:
    """Every number the user wrote, as exact decimals.

    Thousands separators are dropped, so "$5,000" and "5000" are the same
    number. A currency symbol or a percent sign around a number is not part of
    it: the grammar already says which field means dollars and which means
    percent.
    """
    found: set[Decimal] = set()
    for match in _NUMBER.finditer(text):
        try:
            found.add(Decimal(match.group().replace(",", "")))
        except InvalidOperation:  # pragma: no cover - the pattern cannot produce one
            continue
    return found


def symbol_is_in(symbol: str, text: str) -> bool:
    """True when the user's words name this symbol, on a word boundary."""
    return re.search(rf"\b{re.escape(symbol)}\b", text, flags=re.IGNORECASE) is not None


def _numeric_leaves(node: object, path: str) -> Iterator[tuple[str, Decimal]]:
    """Every number in a dumped document, with the path that reached it."""
    if isinstance(node, Mapping):
        for key, value in node.items():
            child = f"{path}.{key}" if path else str(key)
            yield from _numeric_leaves(value, child)
        return
    if isinstance(node, Sequence) and not isinstance(node, str | bytes):
        for index, value in enumerate(node):
            yield from _numeric_leaves(value, f"{path}[{index}]")
        return
    if isinstance(node, bool):
        return
    if isinstance(node, int | Decimal):
        yield path, Decimal(node)


def _show(value: Decimal) -> str:
    return format(value, "f")


def _question_for(path: str, value: Decimal, *, rule_id: str | None) -> str:
    """The question to ask about one number that could not be traced."""
    shown = _show(value)
    tail = f" Nothing in what you wrote says {shown}."
    if rule_id is None:
        general = _CAGE_QUESTIONS.get(path)
        if general is not None:
            return general + tail
        return f"What value did you want for {path}?" + tail
    specific = _RULE_QUESTIONS.get(path)
    if specific is None:
        specific = _RULE_SUFFIX_QUESTIONS.get(path.rsplit(".", 1)[-1])
    if specific is None:
        specific = "What value did you want for " + path + " in the rule {rule!r}?"
    return specific.format(rule=rule_id) + tail


def questions_for_untraceable(spec: StrategySpec, text: str) -> tuple[str, ...]:
    """Questions for everything in `spec` that is not in `text`. Empty means traceable."""
    questions: list[str] = []

    missing_symbols = [symbol for symbol in spec.universe if not symbol_is_in(symbol, text)]
    if missing_symbols:
        questions.append(
            "Which symbols should this trade? The compiled spec names "
            f"{', '.join(missing_symbols)}, which you did not name."
        )

    supplied = numbers_in(text)
    document = spec.model_dump(mode="python")

    # Everything outside `rules` is walked whole rather than field by field, so
    # a numeric field added to the grammar later is checked from the day it
    # exists instead of the day somebody remembers to list it here.
    for key, section in document.items():
        if key == "rules":
            continue
        for path, value in _numeric_leaves(section, str(key)):
            if path not in _EXEMPT_PATHS and value not in supplied:
                questions.append(_question_for(path, value, rule_id=None))

    for index, rule in enumerate(document["rules"]):
        rule_id = spec.rules[index].id
        for path, value in _numeric_leaves(rule, ""):
            if path not in _EXEMPT_PATHS and value not in supplied:
                questions.append(_question_for(path, value, rule_id=rule_id))

    deduplicated: list[str] = []
    for question in questions:
        if question not in deduplicated:
            deduplicated.append(question)
    return tuple(deduplicated)
