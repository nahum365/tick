"""The traceability check, on its own — the enforcement half of "never author".

Everything here is about one question: could this number or this symbol have
come from the person who typed the words? If not, the compiler asks rather than
compiles. The check is deliberately blunt in the safe direction — it fails
CLOSED, refusing what it cannot trace, including a field nobody has mapped a
question to yet.
"""

from __future__ import annotations

import copy
from decimal import Decimal

import pytest

from tick.compile import numbers_in, questions_for_untraceable, symbol_is_in
from tick.spec import parse_spec

WORDS = (
    "Buy 10 shares of XYZ when the price crosses above its 50 bar simple moving "
    "average. Check once at the close. Keep at most 3 positions, no more than 20% "
    "of the account in any one of them, no single order over $5,000, and stop for "
    "the day if the account falls 4%."
)

DOCUMENT = {
    "name": "XYZ 50 bar cross",
    "version": 1,
    "universe": ["XYZ"],
    "cadence": {"kind": "daily_close"},
    "rules": [
        {
            "id": "cross-up",
            "when": {
                "kind": "compare",
                "left": {"kind": "price"},
                "op": "crosses_above",
                "right": {"kind": "sma", "n": 50},
            },
            "then": {
                "side": "buy",
                "size": {"kind": "shares", "shares": 10},
                "order_type": "market",
            },
        }
    ],
    "cage": {
        "max_position_pct": "20",
        "max_positions": 3,
        "max_order_notional": "5000",
        "max_daily_drawdown_pct": "4",
        "allowed_session": "regular_hours",
    },
}


def spec_with(**changes):
    """The traceable document, with one part replaced."""
    document = copy.deepcopy(DOCUMENT)
    for path, value in changes.items():
        node = document
        parts = path.split(".")
        for part in parts[:-1]:
            node = node[int(part)] if part.isdigit() else node[part]
        node[parts[-1]] = value
    return parse_spec(document)


def test_a_spec_whose_every_number_is_in_the_words_asks_nothing():
    assert questions_for_untraceable(spec_with(), WORDS) == ()


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("max_position_pct", "25", "What is the most of your account any one position may be"),
        ("max_positions", 7, "How many positions at most may this agent hold at once?"),
        ("max_order_notional", "9000", "What is the largest dollar amount a single order may be?"),
        ("max_daily_drawdown_pct", "8", "How far may your account fall in a day"),
    ],
)
def test_every_cage_limit_is_traced_and_asked_about_by_name(field, value, expected):
    """A cage nobody chose is not a cage — including one the model chose."""
    spec = spec_with(**{f"cage.{field}": value})
    questions = questions_for_untraceable(spec, WORDS)

    assert len(questions) == 1
    assert expected in questions[0]
    assert str(value).strip('"') in questions[0]


def test_an_invented_lookback_asks_about_the_rule_that_uses_it():
    spec = spec_with(**{"rules.0.when.right": {"kind": "sma", "n": 200}})

    assert questions_for_untraceable(spec, WORDS) == (
        "How many bars should the lookback in the rule 'cross-up' cover? "
        "Nothing in what you wrote says 200.",
    )


def test_an_invented_comparison_number_asks_about_the_rule_that_uses_it():
    spec = spec_with(**{"rules.0.when.right": {"kind": "number", "value": "412.5"}})

    assert questions_for_untraceable(spec, WORDS) == (
        "What number should the rule 'cross-up' compare against? "
        "Nothing in what you wrote says 412.5.",
    )


def test_an_invented_order_size_asks_how_big_the_order_should_be():
    spec = spec_with(**{"rules.0.then.size": {"kind": "notional", "notional": "250"}})

    questions = questions_for_untraceable(spec, WORDS)
    assert questions == (
        "How many dollars should the rule 'cross-up' trade at a time? "
        "Nothing in what you wrote says 250.",
    )


def test_an_invented_cadence_asks_how_often_it_should_run():
    spec = spec_with(cadence={"kind": "every_n_minutes", "n": 15})

    questions = questions_for_untraceable(spec, WORDS)
    assert len(questions) == 1
    assert questions[0].startswith("How often should this run?")


def test_a_symbol_the_user_never_named_is_one_question_naming_it():
    spec = spec_with(universe=["XYZ", "ABCD"])

    assert questions_for_untraceable(spec, WORDS) == (
        "Which symbols should this trade? The compiled spec names ABCD, which you did not name.",
    )


def test_a_document_with_nothing_traceable_asks_about_all_of_it():
    """Words with no numbers in them can support no spec at all."""
    questions = questions_for_untraceable(spec_with(), "do something clever with XYZ")

    asked = " ".join(questions).lower()
    for expected in (
        "position may be",
        "how many positions",
        "largest dollar",
        "how far may your account fall",
        "how many bars",
        "how many shares",
    ):
        assert expected in asked
    assert len(questions) == 6


def test_the_version_is_the_one_number_that_needs_no_source():
    """`version: 1` is a fact about the document format, not about a strategy."""
    words = WORDS.replace("Buy 10 shares", "Buy 10 shares")
    assert "1" not in words.replace("10", "").replace("20", "")
    assert questions_for_untraceable(spec_with(), WORDS) == ()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("buy 10", {Decimal(10)}),
        ("$5,000 at most", {Decimal(5000)}),
        ("12.50 a share", {Decimal("12.50")}),
        ("down .5% today", {Decimal("0.5")}),
        ("between 1,250.75 and 3", {Decimal("1250.75"), Decimal(3)}),
        ("no numbers here", set()),
    ],
)
def test_numbers_are_read_the_way_a_person_writes_them(text, expected):
    assert numbers_in(text) == expected


def test_a_number_matches_however_it_is_spelled_as_a_decimal():
    """20 in the words traces 20.00 in the spec; they are the same number."""
    assert Decimal("20.00") in numbers_in("no more than 20% in one position")


@pytest.mark.parametrize(
    ("symbol", "text", "found"),
    [
        ("XYZ", "buy xyz today", True),
        ("XYZ", "buy XYZ today", True),
        ("XYZ", "buy XYZABC today", False),
        ("ABCD", "buy XYZ today", False),
        ("BRK.B", "hold brk.b forever", True),
    ],
)
def test_a_symbol_is_found_only_as_a_whole_word(symbol, text, found):
    assert symbol_is_in(symbol, text) is found
