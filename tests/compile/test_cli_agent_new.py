"""`tick agent new` — the surface a person compiles on.

The provider is swapped for the recorded fake at the one place the CLI builds
one (`AnthropicSpecProposer.for_environment`), so these tests exercise the real
command: the same compiler, the same traceability check, the same files on
disk, and no network.

What is being pinned here is mostly what the command does NOT do: it writes
nothing when it refuses, it registers nothing without a cancel guard, and it
puts no security in any help text.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.test_product_constraints import REAL_TICKERS
from tick import cli
from tick.compile import API_KEY_ENV
from tick.runtime import AgentRun

from .conftest import FakeAnthropic

runner = CliRunner()


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """TICK_HOME under tmp, always: a test must never write to the developer's."""
    directory = tmp_path / "tick-home"
    monkeypatch.setenv("TICK_HOME", str(directory))
    monkeypatch.setenv(API_KEY_ENV, "sk-ant-not-a-real-key")
    return directory


def use(fixture: str, monkeypatch: pytest.MonkeyPatch) -> FakeAnthropic:
    """Swap the provider the CLI would build for the recorded one."""
    fake = FakeAnthropic.replaying(fixture)
    monkeypatch.setattr(
        cli.AnthropicSpecProposer,
        "for_environment",
        classmethod(lambda cls, **kwargs: fake),
    )
    return fake


def invoke(*args: str):
    return runner.invoke(cli.app, list(args))


def test_compiling_writes_the_spec_and_prints_what_it_will_do(monkeypatch, tmp_path: Path):
    fake = use("simple-cross-buy.json", monkeypatch)
    out = tmp_path / "strategy.json"

    result = invoke("agent", "new", fake.text, "--out", str(out))

    assert result.exit_code == 0, result.output
    assert out.exists()
    assert "cross-up:" in result.output
    assert "its 50-bar simple moving average" in result.output
    assert "it cannot know" in result.output
    assert str(out) in result.output
    assert "came from your own words" in result.output


def test_the_printed_model_id_is_the_one_that_compiled_it(monkeypatch, tmp_path: Path):
    fake = use("simple-cross-buy.json", monkeypatch)

    result = invoke("agent", "new", fake.text, "--out", str(tmp_path / "s.json"))

    assert f"compiled by {fake.model}" in result.output


def test_the_compiled_spec_loads_back_as_the_document_that_was_explained(
    monkeypatch, tmp_path: Path
):
    from tick.spec import load_spec_file

    fake = use("simple-cross-buy.json", monkeypatch)
    out = tmp_path / "strategy.json"
    invoke("agent", "new", fake.text, "--out", str(out))

    spec = load_spec_file(out)
    assert spec.universe == ["XYZ"]
    assert spec.rules[0].id == "cross-up"


def test_with_no_out_the_spec_lands_under_tick_home(monkeypatch, home: Path):
    fake = use("simple-cross-buy.json", monkeypatch)

    result = invoke("agent", "new", fake.text)

    assert result.exit_code == 0, result.output
    written = sorted((home / "compiled").glob("*.json"))
    assert len(written) == 1
    assert str(written[0]) in result.output


def test_a_refusal_prints_the_questions_and_writes_nothing(monkeypatch, home: Path):
    fake = use("missing-everything.json", monkeypatch)

    result = invoke("agent", "new", fake.text)

    assert result.exit_code == 2
    assert "Which symbols should this trade?" in result.output
    assert "Tick will not choose it for you" in result.output
    assert not home.exists() or list(home.rglob("*.json")) == []


def test_an_invented_number_refuses_at_the_command_line_too(monkeypatch, home: Path):
    fake = use("invented-threshold.json", monkeypatch)

    result = invoke("agent", "new", fake.text)

    assert result.exit_code == 2
    assert "How many bars should the lookback in the rule 'above-trend' cover?" in result.output
    assert list(home.rglob("*.json")) == []


def test_a_model_that_cannot_produce_a_valid_spec_writes_nothing(monkeypatch, home: Path):
    fake = use("invalid-twice.json", monkeypatch)

    result = invoke("agent", "new", fake.text)

    assert result.exit_code == 2
    assert "sma(0)" in result.output and "sma(-3)" in result.output
    assert list(home.rglob("*.json")) == []


def test_add_without_a_cancel_guard_is_refused_before_the_model_is_asked(monkeypatch, home: Path):
    fake = use("simple-cross-buy.json", monkeypatch)

    result = invoke("agent", "new", fake.text, "--add")

    assert result.exit_code == 1
    assert "--max-cancels" in result.output
    assert fake.calls == [], "nothing was compiled, so nothing was billed"
    assert not home.exists()


def test_add_registers_the_compiled_agent_in_paper_mode(monkeypatch, home: Path):
    fake = use("simple-cross-buy.json", monkeypatch)

    result = invoke("agent", "new", fake.text, "--add", "--max-cancels", "2")

    assert result.exit_code == 0, result.output
    ids = AgentRun.list_ids(home)
    assert len(ids) == 1
    agent = AgentRun(home, ids[0])
    assert agent.state.mode.value == "paper"
    assert agent.state.max_cancels_per_session == 2
    assert ids[0] in result.output


def test_without_a_key_the_command_says_whose_key_it_wants(monkeypatch, home: Path):
    monkeypatch.delenv(API_KEY_ENV, raising=False)

    result = invoke("agent", "new", "buy 1 share of XYZ when the price is below 2")

    assert result.exit_code == 2
    assert API_KEY_ENV in result.output
    assert not home.exists()


def test_the_help_names_no_security_and_calls_it_a_rule_agent():
    output = invoke("agent", "new", "--help").output
    flat = " ".join(output.split())

    assert REAL_TICKERS.findall(output) == []
    assert "ai agent" not in output.lower()
    assert "rule agent" in flat
    assert "ANTHROPIC_API_KEY" in flat
