"""`tick agent add` and `tick run` for a model-driven agent, end to end.

One thing is replaced: `client_for`, the single place the CLI builds a client
for the provider the document pins, from the user's own environment.
Everything else runs for real: the document is validated, the instructions
file is copied, the model's intents go through the cage, the paper broker
fills what survives, and the record is written and verified.

No test here sets a real key or opens a socket.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from tests.test_product_constraints import REAL_TICKERS
from tick import cli
from tick.agents import EMIT_TOOL_NAME, MissingApiKey, read_model_reply
from tick.records import RecordKind, read
from tick.runtime import AgentRun

from .conftest import MARKET_FIXTURES

runner = CliRunner()

AT = "2026-09-01T11:00:00-04:00"

INSTRUCTIONS = "My own words, written by me.\nBuy XYZ when it looks right to me.\n"

MODEL_DOCUMENT: dict[str, Any] = {
    "kind": "model_agent",
    "name": "My model agent",
    "version": 1,
    "universe": ["ABCD", "XYZ"],
    "cadence": {"kind": "daily_close"},
    "provider": "anthropic",
    "model": "claude-opus-5",
    "cage": {
        "max_position_pct": "100.00",
        "max_positions": 5,
        "max_order_notional": "1000000.00",
        "max_daily_drawdown_pct": "50.00",
        "allowed_session": "regular_hours",
    },
}


def invoke(*args: str, **kwargs):
    return runner.invoke(cli.app, list(args), **kwargs)


@pytest.fixture
def model_file(tmp_path: Path) -> Path:
    path = tmp_path / "model-agent.json"
    path.write_text(json.dumps(MODEL_DOCUMENT), encoding="utf-8")
    return path


@pytest.fixture
def instructions_file(tmp_path: Path) -> Path:
    path = tmp_path / "my-words.md"
    path.write_text(INSTRUCTIONS, encoding="utf-8")
    return path


class Answers:
    """The prepared reply the patched client hands back, and what it was asked."""

    def __init__(self) -> None:
        self.intents: list[Any] = []
        self.requests: list[Any] = []
        self.providers: list[Any] = []
        self.built = 0

    def install(self, monkeypatch: pytest.MonkeyPatch, *, key_missing: bool = False) -> Answers:
        answers = self

        class FakeClient:
            def propose(self, request):
                answers.requests.append(request)
                return read_model_reply(
                    SimpleNamespace(
                        model="claude-opus-5-20260401",
                        stop_reason="tool_use",
                        stop_details=None,
                        content=[
                            SimpleNamespace(
                                type="tool_use",
                                name=EMIT_TOOL_NAME,
                                input={"intents": answers.intents},
                            )
                        ],
                    )
                )

        def build(provider):
            answers.built += 1
            answers.providers.append(provider)
            if key_missing:
                raise MissingApiKey("no model API key. Set ANTHROPIC_API_KEY in this shell.")
            return FakeClient()

        monkeypatch.setattr(cli, "client_for", build)
        return self


@pytest.fixture
def answers(monkeypatch: pytest.MonkeyPatch) -> Answers:
    return Answers().install(monkeypatch)


def add_model(model_file: Path, instructions_file: Path, *extra: str) -> str:
    # These tests predate the per-order default (2026-09-02) and exercise the
    # standing path deliberately; a test that wants the default passes nothing.
    if "--approve" not in extra:
        extra = ("--approve", "standing", *extra)
    result = invoke(
        "agent",
        "add",
        str(model_file),
        "--max-cancels",
        "2",
        "--instructions",
        str(instructions_file),
        *extra,
    )
    assert result.exit_code == 0, result.output
    return result.output.splitlines()[0].strip()


def run_once(agent_id: str, *extra: str, **kwargs):
    return invoke(
        "run",
        agent_id,
        "--once",
        "--market",
        f"fixture:{MARKET_FIXTURES}",
        "--paper-cash",
        "10000.00",
        "--at",
        AT,
        *extra,
        **kwargs,
    )


def buy(symbol: str = "XYZ", qty: int = 2) -> dict[str, Any]:
    return {"symbol": symbol, "side": "buy", "qty": qty, "reason": "it fits my instructions"}


# ----------------------------------------------------------------------
# agent add
# ----------------------------------------------------------------------


def test_adding_a_model_agent_copies_the_users_instructions_beside_the_document(
    model_file: Path, instructions_file: Path, tick_home: Path
):
    agent_id = add_model(model_file, instructions_file)
    agent = AgentRun(tick_home, agent_id)
    assert agent.instructions() == INSTRUCTIONS
    assert agent.spec_path.exists()


def test_adding_a_model_agent_says_it_is_model_driven_and_names_the_model(
    model_file: Path, instructions_file: Path
):
    result = invoke(
        "agent",
        "add",
        str(model_file),
        "--max-cancels",
        "2",
        "--instructions",
        str(instructions_file),
    )
    assert "model-driven agent (claude-opus-5 via anthropic)" in result.output
    assert "paper mode" in result.output


def test_a_model_agent_without_instructions_is_refused_and_nothing_is_written(
    model_file: Path, tick_home: Path
):
    """Tick ships no default instructions, so there is nothing to fall back to."""
    result = invoke("agent", "add", str(model_file), "--max-cancels", "2")

    assert result.exit_code == 1
    assert "--instructions" in result.output
    assert "will not write one" in result.output
    assert AgentRun.list_ids(tick_home) == []


def test_an_empty_instructions_file_is_refused(model_file: Path, tmp_path: Path, tick_home: Path):
    empty = tmp_path / "empty.md"
    empty.write_text("   \n", encoding="utf-8")
    result = invoke(
        "agent", "add", str(model_file), "--max-cancels", "2", "--instructions", str(empty)
    )
    assert result.exit_code == 1
    assert "is empty" in result.output
    assert AgentRun.list_ids(tick_home) == []


def test_a_missing_instructions_file_names_the_path(model_file: Path, tmp_path: Path):
    result = invoke(
        "agent",
        "add",
        str(model_file),
        "--max-cancels",
        "2",
        "--instructions",
        str(tmp_path / "nowhere.md"),
    )
    assert result.exit_code == 1
    assert "could not read the instructions file" in result.output


def test_a_rule_agent_given_instructions_is_refused(tmp_path: Path, instructions_file: Path):
    """An agent holding words nothing reads is worse than one that refused."""
    from .conftest import spec_document

    rule_file = tmp_path / "rule.json"
    rule_file.write_text(json.dumps(spec_document()), encoding="utf-8")
    result = invoke(
        "agent",
        "add",
        str(rule_file),
        "--max-cancels",
        "2",
        "--instructions",
        str(instructions_file),
    )
    assert result.exit_code == 1
    assert "reads no instructions file" in result.output


def test_the_listing_says_which_agents_are_model_driven(
    model_file: Path, instructions_file: Path, tmp_path: Path
):
    from .conftest import spec_document

    rule_file = tmp_path / "rule.json"
    rule_file.write_text(json.dumps(spec_document()), encoding="utf-8")
    invoke("agent", "add", str(rule_file), "--max-cancels", "2")
    add_model(model_file, instructions_file)

    output = invoke("agents").output
    assert "model:claude-opus-5" in output
    assert "  rule  " in output


def test_status_names_the_kind_and_the_model(model_file: Path, instructions_file: Path):
    agent_id = add_model(model_file, instructions_file)
    output = invoke("status", agent_id).output
    assert "kind: model_agent" in output
    assert "model: claude-opus-5" in output
    assert "rules: " in output


# ----------------------------------------------------------------------
# run
# ----------------------------------------------------------------------


def test_a_model_agent_ticks_places_and_records(
    model_file: Path, instructions_file: Path, tick_home: Path, answers: Answers
):
    answers.intents = [buy(qty=2)]
    agent_id = add_model(model_file, instructions_file)

    result = run_once(agent_id)

    assert result.exit_code == 0, result.output
    assert "Your model agent (claude-opus-5) bought 2 XYZ at $118.00 — simulated." in result.output
    agent = AgentRun(tick_home, agent_id)
    assert RecordKind.FILL in [record.kind for record in read(agent.ledger_path)]
    assert agent.verify_ledger().ok


def test_the_run_says_which_model_decides_and_whose_key_it_runs_on(
    model_file: Path, instructions_file: Path, answers: Answers
):
    answers.intents = []
    agent_id = add_model(model_file, instructions_file)
    result = run_once(agent_id)
    assert "decided by claude-opus-5 via anthropic, on your own account" in result.output


def test_the_prompt_that_went_up_is_the_users_own_words_and_the_snapshot(
    model_file: Path, instructions_file: Path, answers: Answers
):
    answers.intents = []
    run_once(add_model(model_file, instructions_file))

    assert len(answers.requests) == 1
    text = answers.requests[0].composed_text()
    assert text.startswith(INSTRUCTIONS)
    snapshot = json.loads(text[len(INSTRUCTIONS) :].strip())
    assert snapshot["universe"] == ["ABCD", "XYZ"]


def test_an_intent_outside_the_users_universe_never_reaches_the_broker(
    model_file: Path, instructions_file: Path, tick_home: Path, answers: Answers
):
    answers.intents = [buy(symbol="WXY")]
    agent_id = add_model(model_file, instructions_file)

    result = run_once(agent_id)

    assert result.exit_code == 0, result.output
    kinds = [record.kind for record in read(AgentRun(tick_home, agent_id).ledger_path)]
    assert RecordKind.FILL not in kinds
    assert RecordKind.REFUSAL in kinds
    assert "proposed an order that was rejected" in result.output


def test_a_missing_api_key_refuses_with_the_variables_name_and_places_nothing(
    model_file: Path, instructions_file: Path, monkeypatch: pytest.MonkeyPatch
):
    """Bring your own model: there is no endpoint of Tick's to fall back to."""
    Answers().install(monkeypatch, key_missing=True)
    agent_id = add_model(model_file, instructions_file)

    result = run_once(agent_id)

    assert result.exit_code == 2
    assert "ANTHROPIC_API_KEY" in result.output


def test_a_model_agent_whose_instructions_were_emptied_refuses_to_run(
    model_file: Path, instructions_file: Path, tick_home: Path, answers: Answers
):
    agent_id = add_model(model_file, instructions_file)
    AgentRun(tick_home, agent_id).instructions_path.write_text("\n", encoding="utf-8")

    result = run_once(agent_id)

    assert result.exit_code == 2
    assert "will not supply one" in result.output
    assert answers.built == 0


def test_a_stopped_model_agent_is_never_asked_anything(
    model_file: Path, instructions_file: Path, answers: Answers
):
    answers.intents = [buy()]
    agent_id = add_model(model_file, instructions_file)
    invoke("stop", agent_id, "--reason", "not today")

    result = run_once(agent_id)

    assert "stopped: not today" in result.output
    assert answers.requests == []


def test_no_new_help_text_names_a_security_or_calls_an_agent_an_ai_agent():
    output = invoke("agent", "add", "--help").output
    assert "ai agent" not in output.lower()
    assert REAL_TICKERS.findall(output) == []
