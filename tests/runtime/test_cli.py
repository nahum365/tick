"""The `tick` command, driven through typer's runner with TICK_HOME in tmp.

Nothing here reaches a network or a broker: the market is a fixture directory
the test points `--market` at, and the account is the local paper simulation.
`--at` freezes the moment, so a test never depends on when it is run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tick import cli
from tick.cli import app
from tick.records import RecordKind, read
from tick.runtime import AgentRun

from .conftest import MARKET_FIXTURES, spec_document

runner = CliRunner()

#: A moment inside the 2026-09-01 session, in ET.
AT = "2026-09-01T11:00:00-04:00"


@pytest.fixture
def spec_file(tmp_path: Path) -> Path:
    path = tmp_path / "strategy.json"
    path.write_text(json.dumps(spec_document()), encoding="utf-8")
    return path


def invoke(*args: str, **kwargs):
    return runner.invoke(app, list(args), **kwargs)


def add(spec_file: Path, *extra: str) -> str:
    # These tests predate the per-order default (2026-09-02) and exercise the
    # standing path deliberately; a test that wants the default passes nothing.
    if "--approve" not in extra:
        extra = ("--approve", "standing", *extra)
    result = invoke("agent", "add", str(spec_file), "--max-cancels", "2", *extra)
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


# ----------------------------------------------------------------------
# Help text
# ----------------------------------------------------------------------


# The no-security and no-"AI agent" scans over `--help` live in
# `tests/test_product_constraints.py`, which walks the command tree instead of
# listing it: the list that used to be here was written in slice 04 and never
# learned about the connect and broker commands slice 06 added. What stays here
# is what this file is about — the sentences the help text has to SAY.


def test_the_top_level_help_says_agents_are_rule_agents():
    result = invoke("--help")
    assert "rule agent" in result.output.lower()


def test_cli_callback_keeps_the_two_installed_codex_executables_on_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    home = tmp_path / "tick-home"
    monkeypatch.setenv("TICK_HOME", str(home))
    monkeypatch.setenv("PATH", "/usr/bin")

    cli._prepend_home_bin()

    assert cli.os.environ["PATH"].split(cli.os.pathsep)[0] == str(home / "bin")


def test_the_market_option_says_tick_ships_no_market_data():
    assert "Tick ships no market data" in _unwrapped(invoke("run", "--help").output)


def _unwrapped(text: str) -> str:
    """Rich wraps help text into a box; join it back into readable prose."""
    return " ".join(line.strip("│ ").strip() for line in text.splitlines())


# ----------------------------------------------------------------------
# agent add / agents
# ----------------------------------------------------------------------


def test_adding_an_agent_validates_copies_and_prints_its_id(spec_file: Path, tick_home: Path):
    agent_id = add(spec_file)
    assert len(agent_id) == 12
    assert (tick_home / "agents" / agent_id / "spec.json").exists()
    assert AgentRun.list_ids(tick_home) == [agent_id]


def test_adding_an_invalid_spec_says_what_is_wrong(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(spec_document(universe=[])), encoding="utf-8")
    result = invoke("agent", "add", str(bad), "--max-cancels", "1")
    assert result.exit_code == 1
    assert "universe" in result.output


def test_the_cancel_limit_is_required(spec_file: Path):
    """A cancel guard nobody chose is not a guard, so there is no default."""
    result = invoke("agent", "add", str(spec_file))
    assert result.exit_code != 0
    assert "--max-cancels" in _unwrapped(result.output)


def test_agents_lists_what_is_there(spec_file: Path):
    agent_id = add(spec_file)
    result = invoke("agents")
    assert result.exit_code == 0
    assert agent_id in result.output
    assert "paper/standing" in result.output


def test_agents_on_an_empty_home_says_how_to_add_one():
    result = invoke("agents")
    assert result.exit_code == 0
    assert "tick agent add" in result.output


# ----------------------------------------------------------------------
# run
# ----------------------------------------------------------------------


def test_one_paper_tick_fills_notifies_and_records(spec_file: Path, tick_home: Path):
    agent_id = add(spec_file)
    result = run_once(agent_id)

    assert result.exit_code == 0, result.output
    assert "Your rule 'always' fired: bought 2 XYZ at $118.00 — simulated." in result.output
    agent = AgentRun(tick_home, agent_id)
    assert [record.kind for record in read(agent.ledger_path)] == [
        RecordKind.NOTE,  # the simulated account this run opened with
        RecordKind.DECISION,
        RecordKind.ORDER,
        RecordKind.FILL,
    ]
    assert agent.verify_ledger().ok


def test_the_opening_paper_balance_is_recorded_so_the_ledger_implies_no_continuity(
    spec_file: Path, tick_home: Path
):
    agent_id = add(spec_file)
    run_once(agent_id)
    note = next(iter(read(AgentRun(tick_home, agent_id).ledger_path)))
    assert note.payload["event"] == "paper_account_opened"
    assert note.payload["cash"] == "10000.00"


def test_a_missing_market_option_is_refused(spec_file: Path):
    agent_id = add(spec_file)
    result = invoke("run", agent_id, "--once", "--paper-cash", "10000.00")
    assert result.exit_code != 0
    assert "--market" in _unwrapped(result.output)


def test_an_unknown_market_scheme_says_there_is_no_feed(spec_file: Path):
    agent_id = add(spec_file)
    result = invoke("run", agent_id, "--once", "--market", "robinhood", "--paper-cash", "10000.00")
    assert result.exit_code == 1
    assert "no live market-data source yet" in result.output


def test_a_paper_balance_of_zero_is_refused(spec_file: Path):
    agent_id = add(spec_file)
    result = invoke(
        "run",
        agent_id,
        "--once",
        "--market",
        f"fixture:{MARKET_FIXTURES}",
        "--paper-cash",
        "0",
        "--at",
        AT,
    )
    assert result.exit_code == 1
    assert "greater than zero" in result.output


def test_at_requires_once(spec_file: Path):
    agent_id = add(spec_file)
    result = invoke(
        "run",
        agent_id,
        "--market",
        f"fixture:{MARKET_FIXTURES}",
        "--paper-cash",
        "10000.00",
        "--at",
        AT,
    )
    assert result.exit_code == 1
    assert "requires --once" in result.output


def test_a_naive_at_is_refused(spec_file: Path):
    agent_id = add(spec_file)
    result = invoke(
        "run",
        agent_id,
        "--once",
        "--market",
        f"fixture:{MARKET_FIXTURES}",
        "--paper-cash",
        "10000.00",
        "--at",
        "2026-09-01T11:00:00",
    )
    assert result.exit_code == 1
    assert "unstated zone" in result.output


def test_running_an_agent_that_does_not_exist_says_how_to_list_them():
    result = invoke(
        "run", "0123456789ab", "--once", "--market", "fixture:.", "--paper-cash", "1.00"
    )
    assert result.exit_code == 1
    assert "tick agent add" in result.output


def test_outside_the_session_nothing_is_read(spec_file: Path, tick_home: Path):
    agent_id = add(spec_file)
    result = run_once(agent_id, "--at", "2026-09-05T11:00:00-04:00")  # a Saturday
    assert result.exit_code == 0
    assert "the market was closed" in result.output


# ----------------------------------------------------------------------
# Approval
# ----------------------------------------------------------------------


def test_each_mode_asks_on_stdin_and_a_no_places_nothing(spec_file: Path, tick_home: Path):
    agent_id = add(spec_file, "--approve", "each")
    result = run_once(agent_id, input="n\n")

    assert result.exit_code == 0, result.output
    assert "Place buy 2 XYZ at $118.00?" in result.output
    assert "you declined" in result.output
    kinds = [record.kind for record in read(AgentRun(tick_home, agent_id).ledger_path)]
    assert RecordKind.FILL not in kinds
    assert RecordKind.REFUSAL in kinds


def test_each_mode_places_on_a_yes(spec_file: Path, tick_home: Path):
    agent_id = add(spec_file, "--approve", "each")
    result = run_once(agent_id, input="y\n")
    assert "fired: bought 2 XYZ" in result.output
    assert RecordKind.FILL in [
        record.kind for record in read(AgentRun(tick_home, agent_id).ledger_path)
    ]


def test_the_approval_mode_can_be_changed_on_a_run(spec_file: Path, tick_home: Path):
    agent_id = add(spec_file)
    run_once(agent_id, "--approve", "each", input="n\n")
    assert AgentRun(tick_home, agent_id).state.approval.value == "each"


# ----------------------------------------------------------------------
# --live
# ----------------------------------------------------------------------


def test_a_live_run_refuses_the_paper_only_options(spec_file: Path, tick_home: Path):
    """Prices from a file while orders go to a real account is the worst pairing."""
    agent_id = add(spec_file)
    result = invoke(
        "run",
        agent_id,
        "--once",
        "--live",
        "--market",
        f"fixture:{MARKET_FIXTURES}",
        "--paper-cash",
        "10000.00",
    )
    assert result.exit_code == 2
    assert "a live run takes no --market and no --paper-cash" in result.output
    assert AgentRun(tick_home, agent_id).state.mode.value == "paper"


def test_a_refused_live_run_leaves_the_agent_in_paper(spec_file: Path, tick_home: Path):
    """A failure that locks the safe mode away is still a failure."""
    agent_id = add(spec_file)
    invoke("run", agent_id, "--once", "--live", "--live-standing-ok")
    assert AgentRun(tick_home, agent_id).state.mode.value == "paper"
    assert run_once(agent_id).exit_code == 0  # and it still runs


# ----------------------------------------------------------------------
# stop / status
# ----------------------------------------------------------------------


def test_stop_sets_the_switch_and_says_how_to_clear_it(spec_file: Path, tick_home: Path):
    agent_id = add(spec_file)
    result = invoke("stop", agent_id, "--reason", "enough for today")

    assert result.exit_code == 0
    assert "enough for today" in result.output
    assert "remove" in result.output
    assert AgentRun(tick_home, agent_id).stop_requested()


def test_a_stopped_agent_places_nothing_on_the_next_run(spec_file: Path, tick_home: Path):
    agent_id = add(spec_file)
    invoke("stop", agent_id)
    result = run_once(agent_id)

    assert result.exit_code == 0
    assert "Tick stopped agent" in result.output
    kinds = [record.kind for record in read(AgentRun(tick_home, agent_id).ledger_path)]
    assert kinds == [RecordKind.STOP]


def test_status_shows_the_mode_the_record_and_the_stop(spec_file: Path):
    agent_id = add(spec_file)
    run_once(agent_id)
    result = invoke("status", agent_id)

    assert result.exit_code == 0
    assert "mode: paper" in result.output
    assert "approval: standing" in result.output
    assert "universe: XYZ" in result.output
    assert "ledger verified: 4 records" in result.output


# ----------------------------------------------------------------------
# ledger
# ----------------------------------------------------------------------


def test_the_ledger_can_be_read_without_naming_a_subcommand(spec_file: Path):
    agent_id = add(spec_file)
    run_once(agent_id)
    result = invoke("ledger", agent_id, "--tail", "1")

    assert result.exit_code == 0
    assert json.loads(result.output.strip())["kind"] == "fill"


def test_verifying_reports_and_exits_zero_when_intact(spec_file: Path):
    agent_id = add(spec_file)
    run_once(agent_id)
    result = invoke("ledger", agent_id, "--verify")
    assert result.exit_code == 0
    assert "ledger verified: 4 records" in result.output


def _tamper(agent: AgentRun) -> None:
    agent.ledger_path.write_text(agent.ledger_path.read_text().replace('"XYZ"', '"WXY"', 1))


def test_a_tampered_record_fails_verification_and_names_the_next_step(
    spec_file: Path, tick_home: Path
):
    agent_id = add(spec_file)
    run_once(agent_id)
    _tamper(AgentRun(tick_home, agent_id))

    result = invoke("ledger", agent_id, "--verify")
    assert result.exit_code == 1
    assert "ledger failed at line" in result.output

    listing = invoke("ledger", agent_id)
    assert listing.exit_code == 1
    assert f"tick ledger new {agent_id}" in listing.output


def test_a_tampered_record_stops_the_run_with_no_order(spec_file: Path, tick_home: Path):
    agent_id = add(spec_file)
    run_once(agent_id)
    agent = AgentRun(tick_home, agent_id)
    _tamper(agent)
    before = agent.ledger_path.read_bytes()

    result = run_once(agent_id)
    assert result.exit_code == 1
    assert f"tick ledger new {agent_id}" in result.output
    assert "Nothing was placed and nothing was recorded" in result.output
    assert agent.ledger_path.read_bytes() == before


def test_ledger_new_starts_the_successor_with_its_genesis_note(spec_file: Path, tick_home: Path):
    agent_id = add(spec_file)
    run_once(agent_id)
    agent = AgentRun(tick_home, agent_id)
    original = agent.ledger_path
    _tamper(agent)
    broken = original.read_bytes()

    result = invoke("ledger", "new", agent_id)
    assert result.exit_code == 0, result.output
    assert "started records.002.jsonl" in result.output
    assert "ledger_succeeded" in result.output
    assert original.read_bytes() == broken  # the evidence is untouched

    successor = agent.directory / "records.002.jsonl"
    note = next(iter(read(successor)))
    assert note.payload["predecessor"] == "records.jsonl"
    assert note.payload["predecessor_head_hash"]
    assert note.payload["reason"]


def test_the_agent_runs_again_once_the_successor_exists(spec_file: Path, tick_home: Path):
    agent_id = add(spec_file)
    run_once(agent_id)
    _tamper(AgentRun(tick_home, agent_id))
    invoke("ledger", "new", agent_id)

    result = run_once(agent_id)
    assert result.exit_code == 0, result.output
    assert "fired: bought 2 XYZ" in result.output


def test_ledger_new_is_refused_while_the_record_still_verifies(spec_file: Path):
    agent_id = add(spec_file)
    run_once(agent_id)
    result = invoke("ledger", "new", agent_id)
    assert result.exit_code == 1
    assert "still the ledger" in result.output


def test_ledger_new_on_an_agent_with_no_record_says_so(spec_file: Path):
    agent_id = add(spec_file)
    result = invoke("ledger", "new", agent_id)
    assert result.exit_code == 1
    assert "nothing to succeed" in result.output


def test_a_spec_below_the_cadence_floor_is_refused_when_it_is_added(tmp_path: Path):
    """Unrunnable is said at `add`, not at 09:31 on the day it first ticks."""
    fast = tmp_path / "fast.json"
    fast.write_text(
        json.dumps(spec_document(cadence={"kind": "every_n_minutes", "n": 1})), encoding="utf-8"
    )
    result = invoke("agent", "add", str(fast), "--max-cancels", "1")
    assert result.exit_code == 1
    assert "excessive market data usage" in result.output
    assert "every_n_minutes(5)" in result.output


def test_the_floor_itself_is_addable(tmp_path: Path):
    ok = tmp_path / "ok.json"
    ok.write_text(
        json.dumps(spec_document(cadence={"kind": "every_n_minutes", "n": 5})), encoding="utf-8"
    )
    assert invoke("agent", "add", str(ok), "--max-cancels", "1").exit_code == 0


def test_one_unreadable_agent_does_not_hide_the_others(
    spec_file: Path, tmp_path: Path, tick_home: Path
):
    """The listing is how a user finds the agent they want to stop."""
    good = add(spec_file)
    second = tmp_path / "second.json"
    second.write_text(json.dumps(spec_document(name="Second")), encoding="utf-8")
    broken = add(second)
    (tick_home / "agents" / broken / "state.json").write_text("{}", encoding="utf-8")

    result = invoke("agents")
    assert result.exit_code == 0
    assert good in result.output
    assert f"{broken}  UNREADABLE" in result.output
