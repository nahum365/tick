"""The append-only ledger: what it guarantees, and what it refuses to do.

The tests are written around four claims, each of which is a claim about
evidence rather than about a function:

1. what was appended is what verifies, in order;
2. changing any byte of any line is visible, at that line;
3. a ledger that does not verify is never extended;
4. nothing here can remove or edit anything — retirement is another append.

Every test drives an injected clock and writes under `tmp_path`. None of them
opens a socket, and there is no broker anywhere in this file.
"""

from __future__ import annotations

import json
import os
import stat
import threading
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from tick.records import (
    GENESIS_PREV_HASH,
    DataSource,
    Ledger,
    LedgerCorrupt,
    PayloadError,
    Record,
    RecordKind,
    ledger_lock_path,
    read,
    verify,
)
from tick.records import last as last_record

from .conftest import START, StepClock


def fill_payload(order: int) -> dict[str, object]:
    """A payload shaped like the ones the runtime will write, with exact money."""
    return {
        "order_id": f"o-{order}",
        "symbol": "XYZ",
        "side": "buy",
        "qty": order,
        "price": Decimal("184.20"),
    }


def append_run(ledger: Ledger, count: int) -> list[Record]:
    return [
        ledger.append(RecordKind.FILL, fill_payload(index), source=DataSource.PAPER)
        for index in range(1, count + 1)
    ]


def test_an_absent_ledger_verifies_as_empty(ledger_path: Path):
    result = verify(ledger_path)
    assert (result.ok, result.count, result.first_bad_seq, result.reason) == (True, 0, None, None)
    assert last_record(ledger_path) is None
    assert list(read(ledger_path)) == []


def test_the_first_record_chains_to_genesis_and_the_rest_to_their_predecessor(ledger: Ledger):
    records = append_run(ledger, 4)
    assert [record.seq for record in records] == [1, 2, 3, 4]
    assert records[0].prev_hash == GENESIS_PREV_HASH
    for previous, current in zip(records, records[1:], strict=False):
        assert current.prev_hash == previous.hash
    assert verify(ledger.path).ok
    assert verify(ledger.path).count == 4


def test_what_was_appended_is_what_reads_back(ledger: Ledger):
    written = append_run(ledger, 3)
    assert list(read(ledger.path)) == written
    assert ledger.last() == written[-1]


def test_the_record_is_stamped_by_the_injected_clock(ledger: Ledger, clock: StepClock):
    records = append_run(ledger, 2)
    assert [record.ts for record in records] == clock.readings
    assert records[0].ts == START


def test_a_clock_that_does_not_return_an_aware_datetime_is_refused(ledger_path: Path):
    naive = Ledger(ledger_path, clock=lambda: datetime(2026, 9, 1, 13, 30))
    with pytest.raises(PayloadError, match="no timezone"):
        naive.append(RecordKind.NOTE, {"text": "x"}, source=DataSource.FIXTURE)
    assert not ledger_path.exists()

    wrong = Ledger(ledger_path, clock=lambda: "2026-09-01")
    with pytest.raises(PayloadError, match="a record is stamped"):
        wrong.append(RecordKind.NOTE, {"text": "x"}, source=DataSource.FIXTURE)


def test_a_decimal_survives_the_round_trip_through_the_file(ledger: Ledger):
    """The number in the file is the number that was recorded, digit for digit."""
    ledger.append(RecordKind.FILL, {"price": Decimal("184.20")}, source=DataSource.PAPER)
    ledger.append(RecordKind.FILL, {"price": Decimal("0.00000001")}, source=DataSource.PAPER)
    prices = [record.payload["price"] for record in read(ledger.path)]
    assert prices == ["184.20", "1E-8"]
    assert [Decimal(price).as_tuple() for price in prices] == [
        Decimal("184.20").as_tuple(),
        Decimal("0.00000001").as_tuple(),
    ]
    assert '"price":"184.20"' in ledger.path.read_text(encoding="utf-8")


def test_a_binary_float_never_reaches_the_file(ledger: Ledger):
    with pytest.raises(PayloadError, match="binary float"):
        ledger.append(RecordKind.FILL, {"price": 184.2}, source=DataSource.PAPER)
    assert verify(ledger.path).count == 0


def test_every_record_carries_the_source_it_was_appended_with(ledger: Ledger):
    for source in DataSource:
        ledger.append(RecordKind.NOTE, {"text": "x"}, source=source)
    assert [record.source for record in read(ledger.path)] == list(DataSource)
    robinhood = [record for record in read(ledger.path) if record.source.derived_from_robinhood]
    assert len(robinhood) == 1


def test_a_payload_cannot_claim_a_different_origin_than_the_record(ledger: Ledger):
    """`source` is reserved: one row, one origin, and it is under the hash."""
    with pytest.raises(PayloadError, match="cannot claim two origins"):
        ledger.append(
            RecordKind.NOTE, {"text": "x", "source": "robinhood"}, source=DataSource.PAPER
        )
    assert verify(ledger.path).count == 0
    echoed = ledger.append(
        RecordKind.NOTE, {"text": "x", "source": "paper"}, source=DataSource.PAPER
    )
    assert echoed.source is DataSource.PAPER


# --------------------------------------------------------------------------
# Tamper evidence
# --------------------------------------------------------------------------


def edit_line(path: Path, index: int, before: str, after: str) -> None:
    """Rewrite one line of the ledger in place — what an attacker or a text
    editor would do, and what the chain exists to make visible."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    assert before in lines[index], f"{before!r} is not in line {index}"
    lines[index] = lines[index].replace(before, after)
    path.write_text("".join(lines), encoding="utf-8")


@pytest.mark.parametrize("index", [0, 1, 2, 3, 4])
def test_editing_any_line_fails_verification_at_that_record(ledger: Ledger, index: int):
    """One changed character anywhere, and the file names where."""
    append_run(ledger, 5)
    edit_line(ledger.path, index, '"qty":', '"qty_":')
    result = verify(ledger.path)
    assert result.ok is False
    assert result.first_bad_seq == index + 1
    assert result.count == index
    assert result.reason


@pytest.mark.parametrize(
    ("before", "after", "why"),
    [
        pytest.param('"symbol":"XYZ"', '"symbol":"ABCD"', "a different instrument", id="symbol"),
        pytest.param('"price":"184.20"', '"price":"18.42"', "a different price", id="price"),
        pytest.param('"side":"buy"', '"side":"sell"', "a different side", id="side"),
        pytest.param('"kind":"fill"', '"kind":"note"', "a different kind", id="kind"),
        pytest.param(
            '"ts":"2026-09-01T13:31:00+00:00"',
            '"ts":"2026-09-01T13:32:00+00:00"',
            "a different time",
            id="ts",
        ),
    ],
)
def test_changing_what_a_record_says_is_visible(ledger: Ledger, before: str, after: str, why: str):
    append_run(ledger, 3)
    edit_line(ledger.path, 1, before, after)
    result = verify(ledger.path)
    assert result.ok is False, why
    assert result.first_bad_seq == 2


def test_re_hashing_the_edited_record_still_breaks_the_chain(ledger: Ledger):
    """The interesting case: an editor who knows how the hash is computed.

    Rewriting record 2 *and* its hash makes record 2 self-consistent — and
    record 3 still quotes the old hash, so the break simply moves one line
    down. Repairing the whole tail is possible for anyone who can write the
    file (see `record.py`: tamper-evident, not tamper-proof); repairing one row
    is not.
    """
    records = append_run(ledger, 3)
    forged = Record.chained(
        seq=2,
        ts=records[1].ts,
        kind=records[1].kind,
        payload=records[1].payload | {"qty": 999},
        prev_hash=records[1].prev_hash,
    )
    lines = ledger.path.read_text(encoding="utf-8").splitlines(keepends=True)
    lines[1] = forged.as_line() + "\n"
    ledger.path.write_text("".join(lines), encoding="utf-8")

    result = verify(ledger.path)
    assert result.ok is False
    assert result.first_bad_seq == 3
    assert "does not match its predecessor" in result.reason


def test_a_chain_break_and_a_rewritten_line_are_different_refusals(ledger: Ledger):
    """Two failures, two sentences: the reader has to know which one this is.

    They call for different things. A line that is not in canonical form was
    rewritten in place or torn on the way to disk, and says nothing about the
    records around it. A hash that does not match its predecessor is a
    well-formed line whose chain no longer holds — something at or before it
    was edited, removed or reordered. Reporting either in the other's words
    sends the person holding the evidence looking in the wrong place.
    """
    records = append_run(ledger, 3)

    lines = ledger.path.read_text(encoding="utf-8").splitlines(keepends=True)
    rewritten = list(lines)
    rewritten[1] = json.dumps(json.loads(rewritten[1]), indent=None, separators=(", ", ": ")) + "\n"
    ledger.path.write_text("".join(rewritten), encoding="utf-8")
    canonical_failure = verify(ledger.path)
    assert canonical_failure.ok is False
    assert canonical_failure.first_bad_seq == 2
    assert "not in canonical form" in canonical_failure.reason
    assert "predecessor" not in canonical_failure.reason

    del lines[1]
    ledger.path.write_text("".join(lines), encoding="utf-8")
    chain_failure = verify(ledger.path)
    assert chain_failure.ok is False
    assert "not in canonical form" not in (chain_failure.reason or "")

    forged = Record.chained(
        seq=2,
        ts=records[1].ts,
        kind=records[1].kind,
        payload=records[1].payload | {"qty": 999},
        prev_hash=records[1].prev_hash,
    )
    lines = ledger.path.read_text(encoding="utf-8").splitlines(keepends=True)
    lines.insert(1, forged.as_line() + "\n")
    ledger.path.write_text("".join(lines), encoding="utf-8")
    predecessor_failure = verify(ledger.path)
    assert predecessor_failure.ok is False
    assert "does not match its predecessor" in predecessor_failure.reason
    assert "not in canonical form" not in predecessor_failure.reason


def test_removing_a_record_from_the_middle_is_visible(ledger: Ledger):
    append_run(ledger, 4)
    lines = ledger.path.read_text(encoding="utf-8").splitlines(keepends=True)
    del lines[1]
    ledger.path.write_text("".join(lines), encoding="utf-8")
    result = verify(ledger.path)
    assert result.ok is False
    assert result.first_bad_seq == 2
    assert "seq" in result.reason


def test_a_reordered_pair_is_visible(ledger: Ledger):
    append_run(ledger, 3)
    lines = ledger.path.read_text(encoding="utf-8").splitlines(keepends=True)
    lines[0], lines[1] = lines[1], lines[0]
    ledger.path.write_text("".join(lines), encoding="utf-8")
    assert verify(ledger.path).first_bad_seq == 1


def test_a_torn_final_line_is_visible(ledger: Ledger):
    """A crash mid-write leaves a line with no newline. It is not repaired."""
    append_run(ledger, 2)
    with ledger.path.open("a", encoding="utf-8") as handle:
        handle.write('{"seq":3,"kind":"fi')
    result = verify(ledger.path)
    assert result.ok is False
    assert result.first_bad_seq == 3
    assert "never finished being written" in result.reason


def test_a_file_that_is_not_a_ledger_at_all_is_reported_not_crashed(ledger_path: Path):
    ledger_path.parent.mkdir(parents=True)
    ledger_path.write_bytes(b"\xff\xfe not text at all\n")
    result = verify(ledger_path)
    assert result.ok is False
    assert result.first_bad_seq == 1
    assert "not UTF-8" in result.reason


def test_reading_a_tampered_ledger_raises_rather_than_yielding_half_a_chain(ledger: Ledger):
    append_run(ledger, 3)
    edit_line(ledger.path, 1, '"qty":2', '"qty":9')
    with pytest.raises(LedgerCorrupt) as exc:
        list(read(ledger.path))
    assert exc.value.at == 2
    with pytest.raises(LedgerCorrupt):
        last_record(ledger.path)


def test_verify_answers_where_read_raises(ledger: Ledger):
    """Asking is not the same as using: `verify` never raises for a broken chain."""
    append_run(ledger, 2)
    edit_line(ledger.path, 0, '"qty":1', '"qty":7')
    result = verify(ledger.path)
    assert result.ok is False
    assert "line 1" in str(result)


# --------------------------------------------------------------------------
# A broken ledger is never extended
# --------------------------------------------------------------------------


def test_appending_to_a_tampered_ledger_refuses_and_writes_nothing(ledger: Ledger):
    append_run(ledger, 3)
    edit_line(ledger.path, 0, '"qty":1', '"qty":8')
    before = ledger.path.read_bytes()

    with pytest.raises(LedgerCorrupt, match="will not be extended"):
        ledger.append(RecordKind.NOTE, {"text": "after"}, source=DataSource.PAPER)

    assert ledger.path.read_bytes() == before, "a refused append must write nothing at all"


def test_appending_after_a_torn_write_refuses(ledger: Ledger):
    append_run(ledger, 1)
    with ledger.path.open("a", encoding="utf-8") as handle:
        handle.write('{"seq":2,')
    with pytest.raises(LedgerCorrupt):
        ledger.append(RecordKind.NOTE, {"text": "x"}, source=DataSource.PAPER)


def test_truncating_the_tail_leaves_a_ledger_that_verifies_and_can_be_extended(ledger: Ledger):
    """The chain has no idea records 4 and 5 ever existed — and says so honestly.

    A hash chain detects edits *inside* it. Truncation of the tail leaves a
    shorter, perfectly valid chain, and appending continues from the new tail.
    That is the limit of what hashing alone can prove, and it is why nothing
    here claims the record is tamper-PROOF.
    """
    records = append_run(ledger, 5)
    lines = ledger.path.read_text(encoding="utf-8").splitlines(keepends=True)
    ledger.path.write_text("".join(lines[:3]), encoding="utf-8")

    assert verify(ledger.path).ok
    assert verify(ledger.path).count == 3

    continued = ledger.append(RecordKind.NOTE, {"text": "after"}, source=DataSource.PAPER)
    assert continued.seq == 4
    assert continued.prev_hash == records[2].hash
    assert verify(ledger.path).ok
    assert [record.seq for record in read(ledger.path)] == [1, 2, 3, 4]


# --------------------------------------------------------------------------
# Retirement, not deletion
# --------------------------------------------------------------------------


def test_retiring_a_record_appends_and_leaves_the_original_untouched(ledger: Ledger):
    records = append_run(ledger, 2)
    retirement = ledger.retire(1, reason="the fill was recorded twice", source=DataSource.PAPER)

    assert retirement.kind is RecordKind.RETIRED
    assert retirement.seq == 3
    assert retirement.payload["retires_seq"] == 1
    assert retirement.payload["reason"] == "the fill was recorded twice"

    still_there = list(read(ledger.path))
    assert still_there[0] == records[0]
    assert len(still_there) == 3
    assert verify(ledger.path).ok


def test_retiring_a_record_that_is_not_there_refuses(ledger: Ledger):
    append_run(ledger, 1)
    with pytest.raises(LookupError, match="nothing there to retire"):
        ledger.retire(7, reason="nope", source=DataSource.PAPER)
    assert verify(ledger.path).count == 1


def test_a_retirement_must_say_why(ledger: Ledger):
    append_run(ledger, 1)
    with pytest.raises(ValueError, match="must say why"):
        ledger.retire(1, reason="   ", source=DataSource.PAPER)


# --------------------------------------------------------------------------
# Concurrency, and the private file it writes
# --------------------------------------------------------------------------


def test_two_ledger_handles_on_one_file_interleave_into_one_chain(ledger_path: Path):
    """Two handles, as two processes would be: the lock keeps the chain single."""
    one = Ledger(ledger_path, clock=StepClock())
    two = Ledger(ledger_path, clock=StepClock())
    for _ in range(3):
        one.append(RecordKind.NOTE, {"who": "one"}, source=DataSource.PAPER)
        two.append(RecordKind.NOTE, {"who": "two"}, source=DataSource.PAPER)
    result = verify(ledger_path)
    assert result.ok and result.count == 6
    assert [record.seq for record in read(ledger_path)] == [1, 2, 3, 4, 5, 6]


def test_concurrent_appends_produce_one_unbroken_chain(ledger_path: Path):
    """Four threads, each with its own handle and its own advisory lock.

    The guarantee is what the module docstring claims and no more: writers that
    all take the lock serialise into one chain with no duplicated `seq`. A
    writer that ignores the lock is not stopped by it — advisory means advisory
    — it is only caught afterwards by `verify`.
    """
    writers = 4
    per_writer = 10
    errors: list[BaseException] = []

    def run(name: int) -> None:
        handle = Ledger(ledger_path, clock=StepClock())
        try:
            for index in range(per_writer):
                handle.append(
                    RecordKind.NOTE, {"writer": name, "n": index}, source=DataSource.PAPER
                )
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(name,)) for name in range(writers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors, errors
    assert not any(thread.is_alive() for thread in threads), "an append deadlocked"
    result = verify(ledger_path)
    assert result.ok, result.reason
    assert result.count == writers * per_writer
    assert [record.seq for record in read(ledger_path)] == list(range(1, writers * per_writer + 1))


def test_the_ledger_and_its_lock_are_private_to_the_user(ledger: Ledger):
    ledger.append(RecordKind.NOTE, {"text": "x"}, source=DataSource.PAPER)
    assert stat.S_IMODE(ledger.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(ledger_lock_path(ledger.path).stat().st_mode) == 0o600
    assert stat.S_IMODE(ledger.path.parent.stat().st_mode) == 0o700


def test_the_lock_is_a_sidecar_and_never_the_ledger_itself(ledger_path: Path):
    assert ledger_lock_path(ledger_path) != ledger_path
    assert ledger_lock_path(ledger_path).name == ledger_path.name + ".lock"


def test_the_ledger_is_written_as_one_line_per_record(ledger: Ledger):
    append_run(ledger, 3)
    text = ledger.path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert len(text.splitlines()) == 3


def test_the_file_is_only_ever_opened_for_appending(ledger: Ledger, monkeypatch):
    """No code path here truncates or rewrites the ledger (invariant 4).

    Pinned at the syscall: every `os.open` of the ledger carries `O_APPEND` and
    never `O_TRUNC`, so even a bug elsewhere in this module cannot shorten the
    file.
    """
    flags: list[int] = []
    real_open = os.open

    def spy(path, flag, *args, **kwargs):
        if str(path) == str(ledger.path):
            flags.append(flag)
        return real_open(path, flag, *args, **kwargs)

    monkeypatch.setattr(os, "open", spy)
    append_run(ledger, 2)

    assert flags, "the ledger was never opened"
    for flag in flags:
        assert flag & os.O_APPEND
        assert not flag & os.O_TRUNC
        assert not flag & os.O_RDWR


def test_a_record_out_of_time_order_is_recorded_rather_than_refused(ledger_path: Path):
    """`seq` is the order of record; `ts` is what the clock said.

    A clock that steps backwards (NTP does) must not brick the ledger: refusing
    would mean a corrected clock stops the runtime recording anything at all,
    which is exactly the failure that leaves a user with no evidence.
    """
    moments = iter(
        [datetime(2026, 9, 1, 13, 30, tzinfo=UTC), datetime(2026, 9, 1, 13, 29, tzinfo=UTC)]
    )
    ledger = Ledger(ledger_path, clock=lambda: next(moments))
    first = ledger.append(RecordKind.NOTE, {"n": 1}, source=DataSource.PAPER)
    second = ledger.append(RecordKind.NOTE, {"n": 2}, source=DataSource.PAPER)
    assert second.ts < first.ts
    assert second.seq > first.seq
    assert verify(ledger_path).ok


def test_appending_to_a_file_that_is_not_a_ledger_refuses_the_same_way(ledger_path: Path):
    """One judgement, one report: `append` refuses by the route `verify` answers by."""
    ledger_path.parent.mkdir(parents=True)
    ledger_path.write_bytes(b"\xff\xfe not text at all\n")
    ledger = Ledger(ledger_path, clock=StepClock())
    with pytest.raises(LedgerCorrupt, match="will not be extended"):
        ledger.append(RecordKind.NOTE, {"text": "x"}, source=DataSource.PAPER)
