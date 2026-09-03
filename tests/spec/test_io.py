"""Loading, dumping, and the sentences a rejection produces.

Two invariants meet here. **No number is fabricated** (CLAUDE.md 5): JSON
numbers are parsed as `Decimal`, so `750.25` in a file is `Decimal("750.25")`
in memory and back out again — a binary float never gets to round it. And
**the spec decides** (3): a document that does not validate produces a report
the author can act on, never a partially-accepted spec.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from tick.spec import (
    MAX_RAW_DEPTH,
    SpecError,
    SpecFormatError,
    SpecValidationError,
    dump_spec,
    load_spec,
    load_spec_file,
    loads_spec,
    parse_spec,
)

from .conftest import INVALID_DIR, invalid_paths, read_document, valid_paths

#: The message each shipped invalid fixture must produce. A fixture without an
#: entry here fails the coverage test below, so adding one forces a decision
#: about what the author is told.
EXPECTED_PROBLEM = {
    "cross-on-cash": "crosses_above: cash has no per-bar history to cross",
    "duplicate-rule-id": "rules use the id 'dip' more than once",
    "empty-universe": "universe must name at least one symbol",
    "lowercase-symbol": "universe symbol 'aapl'",
    "missing-cage-field": "cage.max_order_notional: Field required",
    "missing-order-type": "rule 'dip' then.order_type: Field required",
    "order-exceeds-cage": "rule 'too_big': orders $5000.00 but cage.max_order_notional is $1000.00",
    "sma-window-zero": "rule 'dip' references sma(0): n must be >= 1",
    "unknown-field": "cage.max_leverage: Extra inputs are not permitted",
    "unknown-indicator-kind": "rule 'dip' when.left: Input tag 'rsi'",
}


# --------------------------------------------------------------------------
# The shipped fixtures
# --------------------------------------------------------------------------


@pytest.mark.parametrize("path", valid_paths(), ids=lambda p: p.stem)
def test_every_valid_fixture_loads(path):
    spec = load_spec_file(path)
    assert spec.rules and spec.universe


@pytest.mark.parametrize("path", invalid_paths(), ids=lambda p: p.stem)
def test_every_invalid_fixture_is_refused_with_the_documented_message(path):
    with pytest.raises(SpecValidationError) as caught:
        load_spec_file(path)
    expected = EXPECTED_PROBLEM[path.stem]
    assert any(expected in problem for problem in caught.value.problems), caught.value.problems


def test_every_invalid_fixture_has_a_documented_message():
    assert {path.stem for path in invalid_paths()} == set(EXPECTED_PROBLEM)


def test_at_least_three_valid_fixtures_ship():
    assert len(valid_paths()) >= 3


# --------------------------------------------------------------------------
# Round trip
# --------------------------------------------------------------------------


@pytest.mark.parametrize("path", valid_paths(), ids=lambda p: p.stem)
def test_a_dump_reloads_to_an_equal_spec(path):
    original = load_spec_file(path)
    assert loads_spec(dump_spec(original)) == original


@pytest.mark.parametrize("path", valid_paths(), ids=lambda p: p.stem)
def test_dumping_is_idempotent(path):
    once = dump_spec(load_spec_file(path))
    assert dump_spec(loads_spec(once)) == once


def test_decimals_survive_the_round_trip_exactly():
    spec = load_spec_file(valid_paths()[0])
    rewritten = loads_spec(dump_spec(spec))
    assert rewritten.cage.max_order_notional == spec.cage.max_order_notional
    assert str(rewritten.cage.max_order_notional) == str(spec.cage.max_order_notional)


def test_a_json_number_is_read_as_an_exact_decimal(simple_document):
    """Written as a bare JSON number rather than a string, `0.1` must still be
    exactly 0.1 — `json.loads` would have made it a binary float."""
    document = json.loads(json.dumps(simple_document))
    document["cage"]["max_daily_drawdown_pct"] = 0.1
    spec = loads_spec(json.dumps(document))
    assert spec.cage.max_daily_drawdown_pct == Decimal("0.1")
    assert str(spec.cage.max_daily_drawdown_pct) == "0.1"


def test_a_python_float_is_refused_outright(simple_document):
    simple_document["cage"]["max_daily_drawdown_pct"] = 0.1
    with pytest.raises(SpecValidationError) as caught:
        parse_spec(simple_document)
    assert any("binary float" in problem for problem in caught.value.problems)


def test_a_trailing_zero_written_in_a_file_is_kept(tmp_path):
    path = tmp_path / "spec.json"
    document = read_document(valid_paths()[0])
    document["cage"]["max_order_notional"] = "1234.50"
    path.write_text(json.dumps(document, default=str), encoding="utf-8")
    assert str(load_spec_file(path).cage.max_order_notional) == "1234.50"


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------


def test_load_spec_reads_a_path_object():
    assert load_spec(valid_paths()[0]).name


def test_load_spec_reads_a_path_string():
    assert load_spec(str(valid_paths()[0])).name


def test_load_spec_reads_json_text(simple_document):
    text = json.dumps(simple_document)
    assert load_spec(text).name == "Minimal"


def test_load_spec_reads_indented_json_text(simple_document):
    text = "\n   " + json.dumps(simple_document, indent=4)
    assert load_spec(text).name == "Minimal"


def test_load_spec_refuses_something_that_is_neither():
    with pytest.raises(TypeError):
        load_spec(17)


def test_a_missing_file_names_the_path(tmp_path):
    with pytest.raises(SpecFormatError) as caught:
        load_spec_file(tmp_path / "nope.json")
    assert "nope.json" in str(caught.value)


def test_broken_json_is_a_format_error_not_a_validation_error():
    with pytest.raises(SpecFormatError):
        loads_spec("{ this is not json")


@pytest.mark.parametrize("document", ["[]", '"a spec"', "12"])
def test_a_spec_must_be_a_json_object(document):
    with pytest.raises(SpecFormatError) as caught:
        loads_spec(document)
    assert "JSON object" in str(caught.value)


def test_parse_spec_refuses_a_non_mapping():
    with pytest.raises(SpecFormatError):
        parse_spec(["not", "a", "mapping"])


def test_absurdly_nested_input_is_refused_before_parsing():
    """Guards the recursive descent itself, so a hostile document fails as a
    refusal rather than as a RecursionError from inside pydantic."""
    document = {"rules": []}
    for _ in range(MAX_RAW_DEPTH + 5):
        document = {"of": document}
    with pytest.raises(SpecFormatError) as caught:
        parse_spec(document)
    assert "nests deeper" in str(caught.value)


# --------------------------------------------------------------------------
# The shape of a rejection
# --------------------------------------------------------------------------


def test_a_rejection_names_the_file_it_came_from():
    path = INVALID_DIR / "sma-window-zero.json"
    with pytest.raises(SpecValidationError) as caught:
        load_spec_file(path)
    assert str(path) in str(caught.value)


def test_a_rejection_reports_every_problem_at_once(simple_document):
    simple_document["version"] = 0
    simple_document["universe"] = ["nope!"]
    with pytest.raises(SpecValidationError) as caught:
        parse_spec(simple_document)
    assert len(caught.value.problems) == 2
    assert "2 problems" in str(caught.value)


def test_a_single_problem_reads_in_the_singular(simple_document):
    simple_document["version"] = 0
    with pytest.raises(SpecValidationError) as caught:
        parse_spec(simple_document)
    assert "1 problem)" in str(caught.value)


def test_a_rejection_names_the_rule_by_its_id_not_its_index(simple_document):
    simple_document["rules"][0]["when"]["left"] = {"kind": "ema", "n": 0}
    with pytest.raises(SpecValidationError) as caught:
        parse_spec(simple_document)
    assert caught.value.problems == ("rule 'dip' references ema(0): n must be >= 1",)


def test_a_rejection_falls_back_to_an_index_when_the_rule_has_no_usable_id(simple_document):
    del simple_document["rules"][0]["id"]
    simple_document["rules"][0]["when"]["left"] = {"kind": "ema", "n": 0}
    with pytest.raises(SpecValidationError) as caught:
        parse_spec(simple_document)
    assert "rules[0] references ema(0): n must be >= 1" in caught.value.problems


def test_a_rejection_locates_a_problem_inside_a_nested_combinator(simple_document):
    leaf = simple_document["rules"][0]["when"]
    simple_document["rules"][0]["when"] = {
        "kind": "all_of",
        "of": [
            leaf,
            {"kind": "compare", "left": {"kind": "sma", "n": 0}, "op": ">", "right": leaf},
        ],
    }
    with pytest.raises(SpecValidationError) as caught:
        parse_spec(simple_document)
    assert any("rule 'dip' references sma(0)" in problem for problem in caught.value.problems)


def test_every_spec_failure_is_a_spec_error():
    assert issubclass(SpecFormatError, SpecError)
    assert issubclass(SpecValidationError, SpecError)


def test_the_source_is_carried_on_the_exception():
    path = Path(INVALID_DIR / "empty-universe.json")
    with pytest.raises(SpecValidationError) as caught:
        load_spec_file(path)
    assert caught.value.source == str(path)
