"""The two agent documents, and the discriminator that keeps them apart."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tick.agents import (
    MODEL_AGENT_KIND,
    ModelAgentSpec,
    agent_spec_id,
    dump_agent_spec,
    is_model_agent,
    load_agent_spec_file,
    parse_agent_spec,
)
from tick.spec import SpecError, SpecFormatError, StrategySpec, spec_id

from .conftest import build_model_spec, model_spec_document

RULE_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "specs" / "valid"


def test_a_document_with_no_kind_is_still_a_rule_agents_spec():
    """Adopting model agents changed no document written before them."""
    raw = json.loads((RULE_FIXTURES / "sma-cross-buyer.json").read_text(encoding="utf-8"))
    spec = parse_agent_spec(raw)
    assert isinstance(spec, StrategySpec)
    assert not is_model_agent(spec)


def test_the_id_of_a_rule_spec_is_unchanged_by_the_new_loader():
    """The discriminator is absent on the older kind, so no id moved."""
    path = RULE_FIXTURES / "sma-cross-buyer.json"
    through_the_agent_loader = load_agent_spec_file(path)
    assert agent_spec_id(through_the_agent_loader) == spec_id(through_the_agent_loader)


def test_a_model_agent_document_parses_into_its_own_type():
    spec = parse_agent_spec(model_spec_document())
    assert isinstance(spec, ModelAgentSpec)
    assert is_model_agent(spec)
    assert spec.kind == MODEL_AGENT_KIND
    assert spec.model == "claude-opus-5"


def test_an_unknown_kind_is_refused_rather_than_guessed_at():
    with pytest.raises(SpecFormatError, match="is not an agent kind"):
        parse_agent_spec(model_spec_document() | {"kind": "wishful_agent"})


def test_a_model_agent_must_name_the_model_that_decides():
    with pytest.raises(SpecError, match="model"):
        parse_agent_spec(model_spec_document(model="   "))


def test_a_model_agent_needs_a_cage_and_every_field_of_it():
    document = model_spec_document()
    del document["cage"]["max_daily_drawdown_pct"]
    with pytest.raises(SpecError, match="max_daily_drawdown_pct"):
        parse_agent_spec(document)


def test_a_model_agents_universe_is_validated_like_any_other():
    with pytest.raises(SpecError, match="universe"):
        parse_agent_spec(model_spec_document(universe=["xyz"]))
    with pytest.raises(SpecError, match="more than once"):
        parse_agent_spec(model_spec_document(universe=["XYZ", "XYZ"]))
    with pytest.raises(SpecError, match="at least one symbol"):
        parse_agent_spec(model_spec_document(universe=[]))


def test_a_model_agents_universe_is_sorted_so_two_orders_hash_alike():
    one = build_model_spec(universe=["XYZ", "ABCD"])
    other = build_model_spec(universe=["ABCD", "XYZ"])
    assert agent_spec_id(one) == agent_spec_id(other)


def test_changing_the_model_changes_the_agents_identity():
    """The model is part of the document a person approved, so it is in the id."""
    assert agent_spec_id(build_model_spec(model="claude-opus-5")) != agent_spec_id(
        build_model_spec(model="claude-sonnet-5")
    )


def test_a_model_agent_document_round_trips_through_its_own_dump(tmp_path: Path):
    spec = build_model_spec()
    path = tmp_path / "agent.json"
    path.write_text(dump_agent_spec(spec), encoding="utf-8")
    assert agent_spec_id(load_agent_spec_file(path)) == agent_spec_id(spec)


def test_a_model_agent_document_forbids_fields_nobody_declared():
    with pytest.raises(SpecError):
        parse_agent_spec(model_spec_document() | {"rules": []})


def test_a_file_that_is_not_json_says_so(tmp_path: Path):
    path = tmp_path / "agent.json"
    path.write_text("not json at all", encoding="utf-8")
    with pytest.raises(SpecFormatError, match="could not read the spec as JSON"):
        load_agent_spec_file(path)


def test_a_missing_file_says_which_one(tmp_path: Path):
    with pytest.raises(SpecFormatError, match="could not read agent spec file"):
        load_agent_spec_file(tmp_path / "nowhere.json")
