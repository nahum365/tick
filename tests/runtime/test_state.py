"""An agent's directory: creation, the spec copy, the kill switch, the ledger."""

from __future__ import annotations

import json
import os
import stat
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from tick.records import DataSource, RecordKind, verify
from tick.runtime import (
    AGENT_ID_LENGTH,
    STOP_FILE,
    AgentRun,
    AgentState,
    ApprovalMode,
    LedgerQuarantined,
    Mode,
    RuntimeStateError,
    agent_id_for,
    state_summary,
)
from tick.spec import spec_id

from .conftest import StepClock, build_spec

CREATED = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)


def make(home: Path, spec=None, **kwargs) -> AgentRun:
    return AgentRun.create(
        home,
        spec if spec is not None else build_spec(),
        max_cancels_per_session=kwargs.pop("max_cancels_per_session", 2),
        approval=kwargs.pop("approval", ApprovalMode.STANDING),
        created_at=kwargs.pop("created_at", CREATED),
        instructions=kwargs.pop("instructions", None),
    )


# ----------------------------------------------------------------------
# Creation
# ----------------------------------------------------------------------


def test_an_agent_is_named_by_its_spec(tick_home: Path):
    spec = build_spec()
    run = make(tick_home, spec)
    assert run.agent_id == spec_id(spec)[:AGENT_ID_LENGTH]
    assert run.state.spec_id == spec_id(spec)


def test_the_spec_is_copied_not_referenced(tick_home: Path, tmp_path: Path):
    """The agent runs the copy under TICK_HOME; the file it was added from can go."""
    spec = build_spec()
    run = make(tick_home, spec)
    assert run.spec_path.exists()
    assert spec_id(run.spec) == spec_id(spec)


def test_the_spec_copy_is_written_read_only(tick_home: Path):
    mode = stat.S_IMODE(os.stat(make(tick_home).spec_path).st_mode)
    assert mode == 0o400


def test_state_and_the_directory_are_private(tick_home: Path):
    run = make(tick_home)
    assert stat.S_IMODE(os.stat(run.state_path).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(run.directory).st_mode) == 0o700
    assert stat.S_IMODE(os.stat(run.directory.parent).st_mode) == 0o700


def test_a_fresh_agent_is_paper_with_no_session_yet(tick_home: Path):
    """Paper is the default and live is an explicit act (invariant 2)."""
    state = make(tick_home).state
    assert state.mode is Mode.PAPER
    assert state.day_start_equity is None
    assert state.session_date is None
    assert state.last_tick is None
    assert state.cancels_this_session == 0


def test_adding_the_same_spec_twice_is_the_same_agent(tick_home: Path):
    spec = build_spec()
    first, second = make(tick_home, spec), make(tick_home, spec)
    assert first.agent_id == second.agent_id
    assert AgentRun.list_ids(tick_home) == [first.agent_id]


def test_two_different_specs_are_two_agents(tick_home: Path):
    one = make(tick_home, build_spec(name="One"))
    two = make(tick_home, build_spec(name="Two"))
    assert one.agent_id != two.agent_id
    assert AgentRun.list_ids(tick_home) == sorted([one.agent_id, two.agent_id])


def test_the_cancel_guard_has_no_default(tick_home: Path):
    """A limit nobody chose is not a limit; the parameter is required."""
    import inspect

    parameters = inspect.signature(AgentRun.create).parameters
    for name in ("max_cancels_per_session", "approval", "created_at"):
        assert parameters[name].default is inspect.Parameter.empty


# ----------------------------------------------------------------------
# Loading, and the spec that must not change
# ----------------------------------------------------------------------


def test_loading_an_agent_that_does_not_exist_says_what_to_do(tick_home: Path):
    with pytest.raises(RuntimeStateError, match="tick agent add"):
        AgentRun.load(tick_home, "0123456789ab")


def test_an_id_that_is_not_an_id_is_refused(tick_home: Path):
    for bad in ("../escape", "AGENT", "zzz", ""):
        with pytest.raises(RuntimeStateError):
            AgentRun(tick_home, bad)


def test_editing_the_spec_copy_stops_the_agent_being_loaded(tick_home: Path):
    """An agent executes the document it was created for, or it does not run."""
    run = make(tick_home)
    os.chmod(run.spec_path, 0o600)
    document = json.loads(run.spec_path.read_text())
    document["name"] = "Something else"
    run.spec_path.write_text(json.dumps(document))

    with pytest.raises(RuntimeStateError, match="fixed at creation"):
        AgentRun.load(tick_home, run.agent_id)


def test_state_that_is_not_agent_state_is_refused(tick_home: Path):
    run = make(tick_home)
    run.state_path.write_text('{"agent_id": "nope"}')
    with pytest.raises(RuntimeStateError, match="not agent state"):
        _ = run.state


def test_state_cannot_be_written_into_another_agents_directory(tick_home: Path):
    one, two = make(tick_home, build_spec(name="One")), make(tick_home, build_spec(name="Two"))
    with pytest.raises(RuntimeStateError, match="refusing to write state"):
        one.save_state(two.state)


# ----------------------------------------------------------------------
# The state document
# ----------------------------------------------------------------------


def test_the_opening_equity_and_its_session_are_set_together(tick_home: Path):
    state = make(tick_home).state
    with pytest.raises(ValueError, match="set together"):
        AgentState.model_validate(state.model_dump(mode="json") | {"day_start_equity": "100.00"})


def test_a_zero_opening_equity_is_refused(tick_home: Path):
    state = make(tick_home).state
    with pytest.raises(ValueError, match="day_start_equity"):
        AgentState.model_validate(
            state.model_dump(mode="json") | {"day_start_equity": "0", "session_date": "2026-09-01"}
        )


def test_opening_a_session_resets_the_cancel_counter(tick_home: Path):
    run = make(tick_home)
    state = run.state.opened(session_date=date(2026, 9, 1), equity=Decimal("10000.00")).cancelled()
    assert state.cancels_this_session == 1
    rolled = state.opened(session_date=date(2026, 9, 2), equity=Decimal("9000.00"))
    assert rolled.cancels_this_session == 0
    assert rolled.day_start_equity == Decimal("9000.00")


def test_state_round_trips_through_disk_with_exact_numbers(tick_home: Path):
    run = make(tick_home)
    run.save_state(run.state.opened(session_date=date(2026, 9, 1), equity=Decimal("10000.55")))
    assert run.state.day_start_equity == Decimal("10000.55")
    assert '"10000.55"' in run.state_path.read_text()


def test_a_binary_float_never_reaches_the_opening_equity(tick_home: Path):
    run = make(tick_home)
    with pytest.raises(ValueError, match="binary float"):
        run.state.opened(session_date=date(2026, 9, 1), equity=10000.55)


def test_the_cancel_allowance_counts_down(tick_home: Path):
    run = make(tick_home, max_cancels_per_session=2)
    state = run.state
    assert run.cancels_remaining(state) == 2
    assert run.cancels_remaining(state.cancelled().cancelled()) == 0
    assert run.cancels_remaining(state.cancelled().cancelled().cancelled()) == 0


# ----------------------------------------------------------------------
# The kill switch
# ----------------------------------------------------------------------


def test_the_kill_switch_is_the_presence_of_a_file(tick_home: Path):
    run = make(tick_home)
    assert not run.stop_requested()
    run.request_stop(reason="the user asked", at=CREATED)
    assert run.stop_requested()
    assert (run.directory / STOP_FILE).exists()
    assert "the user asked" in run.stop_reason()


def test_anyone_who_can_touch_the_directory_can_set_it(tick_home: Path):
    """No parsing, no flag, no running process: the file is the switch."""
    run = make(tick_home)
    (run.directory / STOP_FILE).write_text("")
    assert run.stop_requested()
    assert run.stop_reason() == "the kill switch is set"


def test_a_second_stop_leaves_the_first_reason_alone(tick_home: Path):
    run = make(tick_home)
    run.request_stop(reason="first", at=CREATED)
    run.request_stop(reason="second", at=CREATED)
    assert "first" in run.stop_reason()
    assert "second" not in run.stop_reason()


def test_concurrent_stops_leave_the_first_complete_file(tick_home: Path):
    """O_EXCL makes one complete STOP the linearization point under a race."""
    import threading

    run = make(tick_home)
    barrier = threading.Barrier(3)

    def request(reason: str) -> None:
        barrier.wait()
        run.request_stop(reason=reason, at=CREATED)

    threads = [threading.Thread(target=request, args=(reason,)) for reason in ("one", "two")]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    text = run.stop_path.read_text(encoding="utf-8")
    assert text in {
        f"one\nrequested at {CREATED.isoformat()}\n",
        f"two\nrequested at {CREATED.isoformat()}\n",
    }


def test_a_stop_must_say_why(tick_home: Path):
    with pytest.raises(ValueError, match="must say why"):
        make(tick_home).request_stop(reason="  ", at=CREATED)


# ----------------------------------------------------------------------
# Ledger generations and the successor ceremony
# ----------------------------------------------------------------------


def test_the_ledger_lives_in_the_agents_directory(tick_home: Path, clock: StepClock):
    run = make(tick_home)
    run.ledger(clock=clock).append(RecordKind.NOTE, {"text": "x"}, source=DataSource.RUNTIME)
    assert run.ledger_path == run.directory / "records.jsonl"
    assert run.verify_ledger().ok


def test_an_intact_ledger_lets_a_tick_proceed(tick_home: Path, clock: StepClock):
    run = make(tick_home)
    run.ledger(clock=clock).append(RecordKind.NOTE, {"text": "x"}, source=DataSource.RUNTIME)
    assert run.require_verified_ledger().ok


def test_a_tampered_ledger_quarantines_the_agent_and_names_the_next_step(
    tick_home: Path, clock: StepClock
):
    run = make(tick_home)
    ledger = run.ledger(clock=clock)
    ledger.append(RecordKind.NOTE, {"text": "one"}, source=DataSource.RUNTIME)
    ledger.append(RecordKind.NOTE, {"text": "two"}, source=DataSource.RUNTIME)
    run.ledger_path.write_text(run.ledger_path.read_text().replace("one", "won"))

    with pytest.raises(LedgerQuarantined) as raised:
        run.require_verified_ledger()
    assert raised.value.seq == 1
    assert raised.value.next_step == f"tick ledger new {run.agent_id}"
    assert "Nothing was placed and nothing was recorded" in str(raised.value)


def test_a_healthy_ledger_cannot_be_succeeded(tick_home: Path, clock: StepClock):
    """A successor leaves a broken chain behind; it is not a way to start a fresh page."""
    run = make(tick_home)
    run.ledger(clock=clock).append(RecordKind.NOTE, {"text": "x"}, source=DataSource.RUNTIME)
    with pytest.raises(RuntimeStateError, match="still the ledger"):
        run.start_successor_ledger(clock=clock)


def test_an_agent_with_no_ledger_has_nothing_to_succeed(tick_home: Path, clock: StepClock):
    with pytest.raises(RuntimeStateError, match="nothing to succeed"):
        make(tick_home).start_successor_ledger(clock=clock)


def test_a_successor_records_what_it_abandoned_and_leaves_it_untouched(
    tick_home: Path, clock: StepClock
):
    run = make(tick_home)
    ledger = run.ledger(clock=clock)
    ledger.append(RecordKind.NOTE, {"text": "one"}, source=DataSource.RUNTIME)
    second = ledger.append(RecordKind.NOTE, {"text": "two"}, source=DataSource.RUNTIME)
    third = ledger.append(RecordKind.NOTE, {"text": "three"}, source=DataSource.RUNTIME)
    original = run.ledger_path
    before = original.read_bytes()

    # Break the third record only: one and two still verify.
    original.write_text(original.read_text().replace('"three"', '"tree"'))
    broken = original.read_bytes()

    successor, note = run.start_successor_ledger(clock=clock)

    assert successor.name == "records.002.jsonl"
    assert original.read_bytes() == broken != before  # the evidence is left as it is
    assert note.kind is RecordKind.NOTE
    assert note.payload["predecessor"] == "records.jsonl"
    assert note.payload["predecessor_head_seq"] == second.seq
    assert note.payload["predecessor_head_hash"] == second.hash
    assert note.payload["predecessor_first_bad_seq"] == third.seq
    assert note.payload["reason"]
    assert note.payload["source"] == DataSource.RUNTIME.value
    assert verify(successor).ok


def test_the_abandoned_ledger_is_made_read_only(tick_home: Path, clock: StepClock):
    run = make(tick_home)
    run.ledger(clock=clock).append(RecordKind.NOTE, {"text": "one"}, source=DataSource.RUNTIME)
    run.ledger_path.write_text(run.ledger_path.read_text().replace("one", "won"))
    original = run.ledger_path

    run.start_successor_ledger(clock=clock)
    assert stat.S_IMODE(os.stat(original).st_mode) == 0o400


def test_the_successor_becomes_the_ledger_that_is_written_to(tick_home: Path, clock: StepClock):
    run = make(tick_home)
    run.ledger(clock=clock).append(RecordKind.NOTE, {"text": "one"}, source=DataSource.RUNTIME)
    run.ledger_path.write_text(run.ledger_path.read_text().replace("one", "won"))
    successor, _ = run.start_successor_ledger(clock=clock)

    assert run.ledger_path == successor
    assert [path.name for path in run.ledger_paths()] == ["records.jsonl", "records.002.jsonl"]
    run.require_verified_ledger()  # the agent can record again
    run.ledger(clock=clock).append(RecordKind.NOTE, {"text": "after"}, source=DataSource.RUNTIME)
    assert verify(successor).count == 2


def test_a_third_generation_follows_the_second(tick_home: Path, clock: StepClock):
    run = make(tick_home)
    run.ledger(clock=clock).append(RecordKind.NOTE, {"text": "one"}, source=DataSource.RUNTIME)
    run.ledger_path.write_text(run.ledger_path.read_text().replace("one", "won"))
    second, _ = run.start_successor_ledger(clock=clock)
    second.write_text(second.read_text().replace("succeeded", "SUCCEEDED"))

    third, _ = run.start_successor_ledger(clock=clock)
    assert third.name == "records.003.jsonl"
    assert [path.name for path in run.ledger_paths()] == [
        "records.jsonl",
        "records.002.jsonl",
        "records.003.jsonl",
    ]


# ----------------------------------------------------------------------
# The summary the CLI prints
# ----------------------------------------------------------------------


def test_a_summary_names_the_spec_without_reading_the_ledger(tick_home: Path):
    run = make(tick_home)
    summary = state_summary(run)
    assert summary["agent_id"] == run.agent_id
    assert summary["universe"] == ["XYZ"]
    assert summary["mode"] == "paper"
    assert summary["stopped"] is False
    assert summary["ledger"] == "records.jsonl"


def test_the_agent_id_is_derived_from_the_spec_alone():
    spec = build_spec()
    assert agent_id_for(spec) == spec_id(spec)[:AGENT_ID_LENGTH]
