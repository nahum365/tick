"""The explanation: rendered from the document, covering every rule.

The property under test is that the explanation cannot describe a rule the spec
does not contain, because it is generated from the spec. So the tests change
the document and assert the words follow — a window, an operator, a side — and
assert the honesty half names what the rule is blind to.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tick.compile import describe_action, describe_condition, explain
from tick.runtime import FORBIDDEN_PHRASES
from tick.spec import load_spec_file, parse_spec

VALID_SPECS = Path(__file__).resolve().parents[1] / "fixtures" / "specs" / "valid"

BASE = {
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
    document = json.loads(json.dumps(BASE))
    for key, value in changes.items():
        node = document
        parts = key.split(".")
        for part in parts[:-1]:
            node = node[int(part)] if part.isdigit() else node[part]
        node[parts[-1]] = value
    return parse_spec(document)


@pytest.mark.parametrize("path", sorted(VALID_SPECS.glob("*.json")), ids=lambda p: p.stem)
def test_the_explanation_covers_every_rule_id_in_the_spec(path: Path):
    """Every rule, in order, exactly once — including multi-rule specs."""
    spec = load_spec_file(path)
    explanation = explain(spec)

    assert [item.rule_id for item in explanation] == [rule.id for rule in spec.rules]
    assert all(item.what_it_does.endswith(".") for item in explanation)
    assert all(item.what_it_cannot_know for item in explanation)


def test_the_words_follow_the_document_rather_than_the_model():
    fifty = explain(spec_with())[0].what_it_does
    two_hundred = explain(spec_with(**{"rules.0.when.right": {"kind": "sma", "n": 200}}))[
        0
    ].what_it_does

    assert "its 50-bar simple moving average" in fifty
    assert "its 200-bar simple moving average" in two_hundred
    assert "crosses above the price" not in fifty


def test_the_explanation_names_the_cadence_and_the_universe():
    what = explain(spec_with(universe=["ABCD", "XYZ"]))[0].what_it_does

    assert "shortly before the close" in what
    assert "ABCD, XYZ" in what


def test_a_rule_reading_prices_says_it_cannot_know_why_anything_moved():
    blind = explain(spec_with())[0].what_it_cannot_know

    assert any("no news, no earnings" in item for item in blind)
    assert any("not every trade inside each bar" in item for item in blind)


def test_a_cross_says_the_first_bar_can_never_fire_it():
    blind = explain(spec_with())[0].what_it_cannot_know
    assert any("the first bar can never fire this rule" in item for item in blind)

    without_cross = explain(spec_with(**{"rules.0.when.op": ">"}))[0].what_it_cannot_know
    assert not any("first bar" in item for item in without_cross)


def test_a_sell_rule_says_the_runtime_is_long_only():
    """LONG ONLY, stated where the user reads it, not only where the engine enforces it."""
    sell = spec_with(**{"rules.0.then.side": "sell", "rules.0.then.size": {"kind": "all"}})
    blind = explain(sell)[0].what_it_cannot_know

    assert any("long-only" in item and "never open a short" in item for item in blind)
    assert "sell the whole position" in explain(sell)[0].what_it_does


def test_a_rule_reading_the_account_says_whose_account_it_is_in_paper():
    holding = spec_with(
        **{
            "rules.0.when": {
                "kind": "compare",
                "left": {"kind": "position_qty"},
                "op": ">",
                "right": {"kind": "number", "value": "0"},
            }
        }
    )
    blind = explain(holding)[0].what_it_cannot_know
    assert any("not your brokerage's" in item for item in blind)


def test_no_explanation_says_anything_the_product_may_not_say():
    """The notification grammar's forbidden words bind here too."""
    for path in sorted(VALID_SPECS.glob("*.json")):
        for item in explain(load_spec_file(path)):
            words = (item.what_it_does + " " + " ".join(item.what_it_cannot_know)).lower()
            for phrase in FORBIDDEN_PHRASES:
                assert phrase not in words, f"{path.stem}/{item.rule_id} says {phrase!r}"


def test_conditions_and_actions_read_as_the_document_reads():
    spec = spec_with(
        **{
            "rules.0.when": {
                "kind": "all_of",
                "of": [
                    {
                        "kind": "compare",
                        "left": {"kind": "price"},
                        "op": ">",
                        "right": {"kind": "number", "value": "10"},
                    },
                    {
                        "kind": "not",
                        "of": {
                            "kind": "compare",
                            "left": {"kind": "cash"},
                            "op": "<",
                            "right": {"kind": "number", "value": "500"},
                        },
                    },
                ],
            }
        }
    )
    rule = spec.rules[0]

    assert describe_condition(rule.when) == (
        "(the price is above 10 and it is not the case that settled cash is below 500)"
    )
    assert describe_action(rule.then) == "buy 10 shares, as a market order"
