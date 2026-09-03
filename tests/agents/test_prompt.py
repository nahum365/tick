"""What goes up: the user's own words, this tick's snapshot, and the schema.

These are the audit's tests. "Tick contributes only the JSON schema, the
snapshot and the cage" is a claim about bytes on a wire, so it is checked by
subtracting the two things that may be in the prompt and asserting nothing is
left — not by reading the composer and agreeing with it.
"""

from __future__ import annotations

import json
from dataclasses import fields

import pytest

from tick.agents import (
    EMIT_TOOL_NAME,
    PROMPT_JOIN,
    ModelRequest,
    build_snapshot,
    intents_schema,
    snapshot_json,
)
from tick.engine import Quote, Unavailable

from .conftest import (
    INSTRUCTIONS,
    NOW,
    FakeModelClient,
    agent_for,
    build_model_spec,
    market,
    portfolio,
)


def composed(client: FakeModelClient) -> str:
    return client.request.composed_text()


def tick_once(client: FakeModelClient, **kwargs):
    agent = agent_for(client, **kwargs)
    spec = agent.spec
    return agent, agent.evaluate_tick(spec, market(), portfolio(XYZ=4), NOW)


# ----------------------------------------------------------------------
# Only the user's text, the snapshot, and the schema
# ----------------------------------------------------------------------


def test_the_prompt_is_the_users_instructions_and_the_snapshot_and_nothing_else():
    """Subtract both halves from the composed text; what remains is whitespace."""
    client = FakeModelClient.answering([])
    tick_once(client)

    text = composed(client)
    assert text.startswith(INSTRUCTIONS)
    remainder = text[len(INSTRUCTIONS) :]
    snapshot = json.loads(remainder.strip())
    assert snapshot_json(snapshot) == remainder.strip()
    assert remainder[: len(PROMPT_JOIN)] == PROMPT_JOIN
    assert remainder.replace(snapshot_json(snapshot), "").strip() == ""


def test_the_request_has_nowhere_to_put_text_of_ticks_own():
    """Structural, not editorial: `ModelRequest` has no system field at all."""
    names = {field.name for field in fields(ModelRequest)}
    assert names == {"model", "messages", "tools", "max_tokens"}
    assert "system" not in names


def test_the_only_message_is_the_users_own_role():
    client = FakeModelClient.answering([])
    tick_once(client)
    assert [message["role"] for message in client.request.messages] == ["user"]


def test_the_only_tool_offered_is_the_generated_intent_schema():
    client = FakeModelClient.answering([])
    tick_once(client)
    tools = client.request.tools
    assert [tool["name"] for tool in tools] == [EMIT_TOOL_NAME]
    assert tools[0]["input_schema"] == intents_schema()


def test_the_users_instructions_go_up_verbatim():
    """Not paraphrased, not trimmed, not wrapped in anything."""
    words = "  leading space, trailing newline, and a  double  space\n"
    client = FakeModelClient.answering([])
    tick_once(client, instructions=words)
    assert composed(client).startswith(words)


def test_the_model_asked_is_the_one_the_document_pins():
    client = FakeModelClient.answering([])
    tick_once(client, spec=build_model_spec(model="claude-sonnet-5"))
    assert client.request.model == "claude-sonnet-5"


# ----------------------------------------------------------------------
# The snapshot: facts, and stated absences
# ----------------------------------------------------------------------


def test_the_snapshot_carries_the_universe_the_positions_and_the_cage():
    client = FakeModelClient.answering([])
    agent, _ = tick_once(client)
    snapshot = json.loads(composed(client)[len(INSTRUCTIONS) :].strip())

    assert snapshot["universe"] == ["ABCD", "XYZ"]
    assert [row["symbol"] for row in snapshot["positions"]] == ["XYZ"]
    assert snapshot["positions"][0]["quantity"] == 4
    assert snapshot["cage"]["max_position_pct"] == "100.00"
    assert snapshot["cage"]["long_only"] is True
    assert snapshot["cash"] == "10000.00"


def test_a_missing_price_is_named_as_missing_and_never_zeroed():
    snapshot = build_snapshot(
        universe=["XYZ"],
        cadence_kind="daily_close",
        quotes={"XYZ": Unavailable(what="quote for XYZ", reason="the feed said nothing")},
        portfolio=portfolio(),
        equity=Unavailable(what="equity", reason="a held symbol has no price"),
        cage=build_model_spec().cage,
        now=NOW,
    )
    assert snapshot["quotes"]["XYZ"] == {"unavailable": "the feed said nothing"}
    assert snapshot["equity"] == {"unavailable": "a held symbol has no price"}
    assert "0" not in json.dumps(snapshot["quotes"])


def test_every_number_in_the_snapshot_is_a_string_not_a_json_float():
    """A binary float is not the number it was written as, in a prompt as anywhere."""
    client = FakeModelClient.answering([])
    tick_once(client)
    snapshot = json.loads(
        composed(client)[len(INSTRUCTIONS) :].strip(),
        parse_float=lambda raw: pytest.fail(f"the snapshot carried a JSON float: {raw}"),
    )
    assert isinstance(snapshot["cash"], str)


def test_a_quote_in_the_snapshot_carries_its_provenance():
    snapshot = build_snapshot(
        universe=["XYZ"],
        cadence_kind="daily_close",
        quotes={"XYZ": Quote(symbol="XYZ", price="118.00", asof=NOW, source="fixture:XYZ")},
        portfolio=portfolio(),
        equity="10000.00",
        cage=build_model_spec().cage,
        now=NOW,
    )
    assert snapshot["quotes"]["XYZ"] == {
        "price": "118.00",
        "as_of": NOW.isoformat(),
        "source": "fixture:XYZ",
    }


def test_the_snapshot_names_no_security_of_ticks_own():
    """The universe is the user's document. Tick adds nothing to it."""
    client = FakeModelClient.answering([])
    tick_once(client, spec=build_model_spec(universe=["WXY"]))
    snapshot = json.loads(composed(client)[len(INSTRUCTIONS) :].strip())
    assert snapshot["universe"] == ["WXY"]
