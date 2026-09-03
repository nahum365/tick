"""The document: universe, cadence, actions, and the cage that has authority.

The cage tests are the load-bearing ones. CLAUDE.md invariant 3 says the
limits an agent runs under are set deliberately and enforced by code the model
cannot call; a cage field that could be omitted and quietly defaulted would
make that claim false at the first line of the first spec.
"""

from __future__ import annotations

import copy
from decimal import Decimal

import pytest
from pydantic import ValidationError

from tick.spec import (
    MAX_CONDITION_DEPTH,
    Action,
    AllSize,
    Cage,
    Compare,
    ComparisonOp,
    EveryNMinutes,
    NotionalSize,
    NumberLiteral,
    OrderType,
    PctOfEquitySize,
    Price,
    Rule,
    Session,
    SharesSize,
    Side,
    SpecValidationError,
    condition_depth,
    parse_spec,
)

CAGE_FIELDS = (
    "max_position_pct",
    "max_positions",
    "max_order_notional",
    "max_daily_drawdown_pct",
    "allowed_session",
)


def _valid_cage_kwargs() -> dict:
    return {
        "max_position_pct": Decimal("10"),
        "max_positions": 3,
        "max_order_notional": Decimal("500"),
        "max_daily_drawdown_pct": Decimal("2"),
        "allowed_session": Session.REGULAR_HOURS,
    }


# --------------------------------------------------------------------------
# The cage
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dropped", CAGE_FIELDS)
def test_every_cage_field_is_required(dropped, simple_document):
    """No cage field has a default. Dropping any one of them refuses the spec."""
    document = copy.deepcopy(simple_document)
    del document["cage"][dropped]
    with pytest.raises(SpecValidationError) as caught:
        parse_spec(document)
    assert f"cage.{dropped}: Field required" in caught.value.problems


@pytest.mark.parametrize("dropped", CAGE_FIELDS)
def test_the_cage_model_itself_has_no_defaults(dropped):
    kwargs = _valid_cage_kwargs()
    del kwargs[dropped]
    with pytest.raises(ValidationError):
        Cage(**kwargs)


@pytest.mark.parametrize(
    ("field", "bad", "expected"),
    [
        ("max_position_pct", Decimal("0"), "cage.max_position_pct(0): must be > 0 and <= 100"),
        ("max_position_pct", Decimal("101"), "cage.max_position_pct(101): must be > 0 and <= 100"),
        ("max_positions", 0, "cage.max_positions(0): must be >= 1"),
        ("max_order_notional", Decimal("0"), "cage.max_order_notional(0): must be > 0"),
        (
            "max_daily_drawdown_pct",
            Decimal("-1"),
            "cage.max_daily_drawdown_pct(-1): must be > 0 and <= 100",
        ),
    ],
)
def test_cage_bounds_are_enforced_with_a_readable_message(field, bad, expected):
    kwargs = _valid_cage_kwargs()
    kwargs[field] = bad
    with pytest.raises(ValidationError) as caught:
        Cage(**kwargs)
    assert expected in str(caught.value)


def test_the_cage_refuses_a_binary_float_for_money():
    kwargs = _valid_cage_kwargs()
    kwargs["max_order_notional"] = 500.0
    with pytest.raises(ValidationError) as caught:
        Cage(**kwargs)
    assert "binary float" in str(caught.value)


def test_only_regular_hours_is_a_session_today(simple_document):
    simple_document["cage"]["allowed_session"] = "extended_hours"
    with pytest.raises(SpecValidationError) as caught:
        parse_spec(simple_document)
    assert any("allowed_session" in problem for problem in caught.value.problems)


# --------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------


def test_an_action_names_its_order_type_explicitly():
    with pytest.raises(ValidationError):
        Action(side=Side.BUY, size=AllSize())


def test_the_action_side_admits_exactly_buy_and_sell(simple_document):
    """Long only (invariant 9): the spec cannot express a short, at all.

    The engine refuses a sell larger than the position and both brokers refuse
    it again, but every one of those checks asks "is this a sell, and is it
    bigger than what is held?" — a question that is only complete for two
    sides. So the two are pinned here, at the boundary the whole runtime reads
    from: a document naming a third fails to parse rather than reaching an
    engine that would treat whatever it is as one of these.
    """
    assert sorted(side.value for side in Side) == ["buy", "sell"]
    simple_document["rules"][0]["then"]["side"] = "short"
    with pytest.raises(SpecValidationError) as caught:
        parse_spec(simple_document)
    assert any("side" in problem for problem in caught.value.problems)


def test_market_is_the_only_order_type_today(simple_document):
    simple_document["rules"][0]["then"]["order_type"] = "limit"
    with pytest.raises(SpecValidationError) as caught:
        parse_spec(simple_document)
    assert any("order_type" in problem for problem in caught.value.problems)
    assert OrderType.MARKET.value == "market"


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        (SharesSize(shares=4), "4 shares"),
        (NotionalSize(notional=Decimal("250.00")), "$250.00 notional"),
        (PctOfEquitySize(pct_of_equity=Decimal("2.5")), "2.5% of equity"),
        (AllSize(), "all"),
    ],
)
def test_every_sizing_labels_itself(size, expected):
    assert size.label() == expected


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        ({"kind": "shares", "shares": 0}, "shares(0): shares must be >= 1"),
        ({"kind": "notional", "notional": "0"}, "notional(0): notional must be > 0"),
        (
            {"kind": "pct_of_equity", "pct_of_equity": "0"},
            "pct_of_equity(0): must be > 0 and <= 100",
        ),
        (
            {"kind": "pct_of_equity", "pct_of_equity": "100.01"},
            "pct_of_equity(100.01): must be > 0 and <= 100",
        ),
    ],
)
def test_sizings_refuse_a_quantity_that_orders_nothing(size, expected, simple_document):
    simple_document["rules"][0]["then"]["size"] = size
    with pytest.raises(SpecValidationError) as caught:
        parse_spec(simple_document)
    assert any(expected in problem for problem in caught.value.problems)


def test_a_size_cannot_set_two_sizings_at_once(simple_document):
    """The tagged union is what makes 'exactly one' structural: there is no
    shape in which both `shares` and `notional` are present."""
    simple_document["rules"][0]["then"]["size"] = {
        "kind": "shares",
        "shares": 1,
        "notional": "100",
    }
    with pytest.raises(SpecValidationError) as caught:
        parse_spec(simple_document)
    assert any("Extra inputs are not permitted" in problem for problem in caught.value.problems)


def test_a_size_must_say_which_sizing_it_is(simple_document):
    simple_document["rules"][0]["then"]["size"] = {"shares": 1}
    with pytest.raises(SpecValidationError) as caught:
        parse_spec(simple_document)
    assert any("kind" in problem for problem in caught.value.problems)


# --------------------------------------------------------------------------
# Rules against the cage
# --------------------------------------------------------------------------


def test_a_notional_order_larger_than_the_cage_is_refused(simple_document):
    simple_document["rules"][0]["then"]["size"] = {"kind": "notional", "notional": "5000.00"}
    with pytest.raises(SpecValidationError) as caught:
        parse_spec(simple_document)
    assert (
        "rule 'dip': orders $5000.00 but cage.max_order_notional is $500.00"
        in caught.value.problems
    )


def test_a_buy_larger_than_the_position_cap_is_refused(simple_document):
    simple_document["rules"][0]["then"]["size"] = {"kind": "pct_of_equity", "pct_of_equity": "50"}
    with pytest.raises(SpecValidationError) as caught:
        parse_spec(simple_document)
    assert (
        "rule 'dip': buys 50% of equity but cage.max_position_pct is 10.00%"
        in caught.value.problems
    )


def test_a_sell_is_not_bounded_by_the_position_cap(simple_document):
    """Selling more than the cap is how you get back under it."""
    simple_document["rules"][0]["then"] = {
        "side": "sell",
        "size": {"kind": "pct_of_equity", "pct_of_equity": "50"},
        "order_type": "market",
    }
    spec = parse_spec(simple_document)
    assert spec.rules[0].then.side is Side.SELL


def test_an_order_at_exactly_the_cage_ceiling_is_allowed(simple_document):
    simple_document["rules"][0]["then"]["size"] = {"kind": "notional", "notional": "500.00"}
    spec = parse_spec(simple_document)
    assert spec.rules[0].then.size.notional == Decimal("500.00")


# --------------------------------------------------------------------------
# Universe, cadence, rules, name, version
# --------------------------------------------------------------------------


def test_the_universe_is_sorted_so_typing_order_does_not_change_the_spec(simple_document):
    simple_document["universe"] = ["NVDA", "AAPL", "BRK.B"]
    assert parse_spec(simple_document).universe == ["AAPL", "BRK.B", "NVDA"]


def test_a_repeated_symbol_is_refused_not_silently_collapsed(simple_document):
    simple_document["universe"] = ["AAPL", "MSFT", "AAPL"]
    with pytest.raises(SpecValidationError) as caught:
        parse_spec(simple_document)
    assert "universe lists 'AAPL' more than once" in caught.value.problems


@pytest.mark.parametrize("symbol", ["aapl", "TOOLONGX", "AAPL ", "AA-PL", "", "BRK.BBB"])
def test_a_malformed_symbol_is_refused(symbol, simple_document):
    simple_document["universe"] = [symbol]
    with pytest.raises(SpecValidationError) as caught:
        parse_spec(simple_document)
    assert any("must be 1-5 capital letters" in problem for problem in caught.value.problems)


@pytest.mark.parametrize("symbol", ["A", "AAPL", "BRK.B", "GOOGL"])
def test_well_formed_symbols_are_accepted(symbol, simple_document):
    simple_document["universe"] = [symbol]
    assert parse_spec(simple_document).universe == [symbol]


def test_an_empty_universe_is_refused(simple_document):
    simple_document["universe"] = []
    with pytest.raises(SpecValidationError) as caught:
        parse_spec(simple_document)
    assert "universe must name at least one symbol" in caught.value.problems


def test_a_spec_needs_at_least_one_rule(simple_document):
    simple_document["rules"] = []
    with pytest.raises(SpecValidationError) as caught:
        parse_spec(simple_document)
    assert "a spec must have at least one rule" in caught.value.problems


def test_rule_ids_are_unique_within_a_spec(simple_document):
    simple_document["rules"].append(copy.deepcopy(simple_document["rules"][0]))
    with pytest.raises(SpecValidationError) as caught:
        parse_spec(simple_document)
    assert "rules use the id 'dip' more than once" in caught.value.problems


@pytest.mark.parametrize("rule_id", ["", "Dip", "_dip", "with space", "x" * 41])
def test_a_malformed_rule_id_is_refused(rule_id, simple_document):
    simple_document["rules"][0]["id"] = rule_id
    with pytest.raises(SpecValidationError) as caught:
        parse_spec(simple_document)
    assert any("must be 1-40 characters" in problem for problem in caught.value.problems)


def test_a_rule_can_be_looked_up_by_id(simple_document):
    spec = parse_spec(simple_document)
    assert spec.rule("dip").id == "dip"
    with pytest.raises(KeyError):
        spec.rule("nope")


@pytest.mark.parametrize("n", [1, 5, 390])
def test_every_n_minutes_accepts_a_cadence_inside_the_session(n):
    assert EveryNMinutes(n=n).n == n


@pytest.mark.parametrize("n", [0, -5, 391])
def test_every_n_minutes_refuses_a_cadence_outside_the_session(n):
    with pytest.raises(ValidationError) as caught:
        EveryNMinutes(n=n)
    assert f"every_n_minutes({n}): n must be between 1 and 390" in str(caught.value)


def test_an_unknown_cadence_is_refused(simple_document):
    simple_document["cadence"] = {"kind": "hourly"}
    with pytest.raises(SpecValidationError) as caught:
        parse_spec(simple_document)
    assert any("Input tag 'hourly'" in problem for problem in caught.value.problems)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("", "name must not be empty"),
        (" leading", "name must not start or end with whitespace"),
        ("trailing ", "name must not start or end with whitespace"),
        ("x" * 121, "name must be at most 120 characters"),
        ("bell\x07", "name must not contain control characters"),
    ],
)
def test_a_name_that_would_not_read_back_is_refused(name, expected, simple_document):
    simple_document["name"] = name
    with pytest.raises(SpecValidationError) as caught:
        parse_spec(simple_document)
    assert expected in caught.value.problems


@pytest.mark.parametrize("version", [0, -1])
def test_a_version_below_one_is_refused(version, simple_document):
    simple_document["version"] = version
    with pytest.raises(SpecValidationError) as caught:
        parse_spec(simple_document)
    assert f"version({version}): must be >= 1" in caught.value.problems


def test_a_validated_spec_cannot_be_mutated(simple_document):
    """The spec_id names a document. A mutable document would outlive its id."""
    spec = parse_spec(simple_document)
    with pytest.raises(ValidationError):
        spec.name = "something else"
    with pytest.raises(ValidationError):
        spec.cage.max_positions = 99


def test_a_rule_holds_the_condition_and_action_it_was_built_with():
    rule = Rule(
        id="x",
        when=Compare(left=Price(), op=ComparisonOp.LT, right=NumberLiteral(value=Decimal(1))),
        then=Action(side=Side.BUY, size=AllSize(), order_type=OrderType.MARKET),
    )
    assert rule.then.label() == "buy all"


# --------------------------------------------------------------------------
# Condition depth, enforced per rule
# --------------------------------------------------------------------------


def _nest(document: dict, depth: int) -> dict:
    """Wrap the fixture's condition in `depth` layers of `not`."""
    condition = document["rules"][0]["when"]
    for _ in range(depth):
        condition = {"kind": "not", "of": condition}
    document["rules"][0]["when"] = condition
    return document


def test_a_condition_may_nest_up_to_the_documented_limit(simple_document):
    spec = parse_spec(_nest(simple_document, MAX_CONDITION_DEPTH - 1))
    assert condition_depth(spec.rules[0].when) == MAX_CONDITION_DEPTH


def test_a_condition_nested_past_the_limit_is_refused(simple_document):
    with pytest.raises(SpecValidationError) as caught:
        parse_spec(_nest(simple_document, MAX_CONDITION_DEPTH))
    assert (
        f"rule 'dip': condition nests {MAX_CONDITION_DEPTH + 1} levels deep; "
        f"the limit is {MAX_CONDITION_DEPTH}" in caught.value.problems
    )


def test_a_comparison_cannot_start_from_a_constant(simple_document):
    """`left` is an Indicator, never a literal: a rule always starts from a
    measurement, so `5 > 3` is not a shape the grammar has."""
    simple_document["rules"][0]["when"]["left"] = {"kind": "number", "value": "5"}
    with pytest.raises(SpecValidationError) as caught:
        parse_spec(simple_document)
    assert any("Input tag 'number'" in problem for problem in caught.value.problems)
