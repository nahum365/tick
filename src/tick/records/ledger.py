"""The ledger: an append-only JSONL file that will not be extended once broken.

    ledger = Ledger(path, clock=utc_clock)
    ledger.append(RecordKind.FILL, {"order_id": "o-1"}, source=DataSource.PAPER)
    ledger.verify().ok

Four properties, and each one is a decision rather than an implementation
detail.

**Append is the only write.** There is no delete, no edit, no rewrite and no
compaction anywhere in this package (CLAUDE.md invariant 4). A row that no
longer applies is superseded by a `retired` record naming it, which is itself
appended; the superseded row stays exactly where it was, because a record that
can be withdrawn is not a record.

**A broken chain is never extended.** `append` verifies the file it is about to
add to and raises `LedgerCorrupt` if anything in it fails — so a tampered or
truncated ledger stops the runtime instead of quietly growing a second,
plausible-looking tail on top of evidence that has already been rewritten. What
the user can still do after that failure is start a new ledger beside the old
one; what nobody can do is make the old one look intact.

**Verification is a question, not an accident.** `verify(path)` walks the whole
chain and *answers* — `VerifyResult(ok, first_bad_seq, reason)` — because
"check this file" is something a person asks on purpose. Only `read` and
`append`, which are about to act on the contents, raise.

**The file is local, and stays local.** It lives under `TICK_HOME` on the
user's own machine. Nothing in this package opens a socket, and every payload
names its `source`, so a later export tool can refuse to ship the rows derived
from Robinhood's Trading MCP (2026-08-31 terms audit). The ledger is not a
telemetry stream and there is no upload path to disable.

Concurrency, and its limits. Every append takes an **advisory** exclusive
`flock` on a sidecar `<ledger>.lock`, and every read takes a shared one, so two
Tick processes on one machine interleave cleanly rather than producing two
records with the same `seq`. Advisory means exactly that: a process that writes
to the file without taking the lock is not stopped by it, only noticed
afterwards by `verify`. `flock` is POSIX (macOS and Linux; not Windows) and is
unreliable over NFS, so a ledger on a network share has no locking guarantee at
all — keep it on local disk. Nothing here coordinates across machines.
"""

from __future__ import annotations

import fcntl
import os
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from .errors import LedgerCorrupt, LedgerFormatError, PayloadError, RecordError
from .home import ensure_private_dir
from .record import (
    GENESIS_PREV_HASH,
    SOURCE_KEY,
    DataSource,
    Record,
    RecordKind,
    decode_line,
    encode_line,
)

__all__ = [
    "Ledger",
    "VerifyResult",
    "last",
    "ledger_lock_path",
    "read",
    "utc_clock",
    "verify",
]

#: The ledger and its lock are private to the user. The record describes what
#: the user owns and what their agents did with it; it is not world-readable,
#: for the same reason the token store is not (invariant 1). Directories are
#: created by `ensure_private_dir`, which is private at every level.
FILE_MODE = 0o600


def utc_clock() -> datetime:
    """The wall clock, in UTC, timezone-aware. The production `clock`."""
    return datetime.now(UTC)


def ledger_lock_path(path: str | os.PathLike[str]) -> Path:
    """The sidecar advisory lock for a ledger file.

    A separate file, so that locking never opens the ledger for writing and a
    stale lock can be removed without touching the record.
    """
    resolved = Path(path)
    return resolved.with_name(resolved.name + ".lock")


class VerifyResult(BaseModel):
    """The answer to "does this ledger verify?" — never a bare boolean.

    `first_bad_seq` is the 1-based position of the first LINE that failed,
    which in a chain that was intact up to that point is also its `seq` — and
    is rendered as "line N" for the reason `LedgerCorrupt` is: the position is
    countable, the seq is only what the line claims. `reason` says what was
    wrong in words the person holding the file can act on, and it names WHICH
    failure it was: not in canonical form, a hash that does not match its
    predecessor, or a hash that does not match its own contents. A passing
    result carries neither, and a failing result carries both: "it failed"
    with no position and no reason is not a report.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    ok: bool
    first_bad_seq: int | None
    reason: str | None
    #: How many records were read. On a failure, how many verified before it.
    count: int

    @model_validator(mode="after")
    def _check(self) -> VerifyResult:
        if self.ok and (self.first_bad_seq is not None or self.reason is not None):
            raise ValueError("a passing verification names no bad record and gives no reason")
        if not self.ok and (self.first_bad_seq is None or not (self.reason or "").strip()):
            raise ValueError("a failed verification must say which record failed and why")
        if self.count < 0:
            raise ValueError(f"count ({self.count}) cannot be negative")
        return self

    def __str__(self) -> str:
        if self.ok:
            return f"ledger verified: {self.count} records"
        return f"ledger failed at line {self.first_bad_seq}: {self.reason}"


def _read_text(path: Path, *, lock: bool) -> str | None:
    """The ledger's whole contents. `None` if the file is absent.

    A ledger is read whole because verification is a statement about the whole
    chain. `lock=True` takes the shared advisory lock, which is what stops a
    reader from seeing a line another process is in the middle of appending —
    that would look exactly like corruption and would be reported as such.
    `lock=False` is for a caller that already holds the exclusive lock;
    re-taking it on a second descriptor would block the process against itself.
    """
    if not path.exists():
        return None
    if lock:
        with _shared_lock(path):
            data = path.read_bytes()
    else:
        data = path.read_bytes()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LedgerFormatError(
            f"{path} is not UTF-8 text, so it is not a Tick ledger: {exc}"
        ) from exc


@contextmanager
def _shared_lock(path: Path) -> Iterator[None]:
    fd = os.open(ledger_lock_path(path), os.O_CREAT | os.O_RDWR, FILE_MODE)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _scan(text: str) -> Iterator[Record]:
    """Walk the chain, raising `LedgerCorrupt` at the first thing that is wrong."""
    if not text:
        return
    if not text.endswith("\n"):
        raise LedgerCorrupt(
            "the last line has no newline, so it was never finished being written; "
            "a torn ledger is not repaired in place — start a new one beside it",
            at=text.count("\n") + 1,
        )
    previous: Record | None = None
    for index, line in enumerate(text.split("\n")[:-1], start=1):
        if not line.strip():
            raise LedgerCorrupt("the line is blank; every line is one record", at=index)
        record = decode_line(line, at=index)
        expected_seq = 1 if previous is None else previous.seq + 1
        if record.seq != expected_seq:
            raise LedgerCorrupt(
                f"the record says seq {record.seq} but it is record {expected_seq} of the "
                f"file; a sequence that skips or repeats means records were removed or "
                f"reordered",
                at=index,
            )
        expected_prev = GENESIS_PREV_HASH if previous is None else previous.hash
        if record.prev_hash != expected_prev:
            raise LedgerCorrupt(
                f"the hash does not match its predecessor: this line follows "
                f"{record.prev_hash[:12]}… and the line before it hashes to "
                f"{expected_prev[:12]}…, so a line at or before this one was edited, "
                f"removed or reordered. The line itself is well-formed and in canonical "
                f"form; what broke is the chain",
                at=index,
            )
        yield record
        previous = record


def read(path: str | os.PathLike[str]) -> Iterator[Record]:
    """Every record in the ledger, oldest first, raising at the first break.

    Reading raises rather than answering because a caller iterating records is
    about to *use* them, and half a chain handed over silently is worse than no
    chain at all. Ask `verify` when the question is whether the file is intact.
    An absent ledger reads as no records.
    """
    text = _read_text(Path(path), lock=True)
    if text is None:
        return iter(())
    return _scan(text)


def last(path: str | os.PathLike[str]) -> Record | None:
    """The most recent record, or `None` when nothing has been recorded yet.

    Reads and verifies the whole chain on the way, so a `last` taken from a
    broken ledger raises instead of answering with a row that may be the tail
    of a rewritten file.
    """
    tail: Record | None = None
    for record in read(path):
        tail = record
    return tail


def verify(path: str | os.PathLike[str]) -> VerifyResult:
    """Walk the whole chain and report what is true of it.

    An absent or empty ledger verifies with `count == 0` — there is nothing in
    it to be wrong. That is not the same as proving nothing was deleted: a hash
    chain cannot detect the removal of the *entire* file, only of records
    inside one. Detecting that needs the head of the chain anchored somewhere
    else, which the prototype does not do.
    """
    return _verify_path(Path(path), lock=True)


def _verify_path(path: Path, *, lock: bool) -> VerifyResult:
    """`verify`, with the choice of taking the shared lock or not.

    `lock=False` is for `append`, which already holds the exclusive lock: it
    must reach the same judgement by the same route, so that "the ledger does
    not verify" means one thing and is reported one way.
    """
    try:
        text = _read_text(path, lock=lock)
    except LedgerFormatError as exc:
        return VerifyResult(ok=False, first_bad_seq=1, reason=str(exc), count=0)
    return _verify_text(text or "")


def _verify_text(text: str) -> VerifyResult:
    """`verify` over contents already in hand — the whole of its judgement."""
    count = 0
    try:
        for _ in _scan(text):
            count += 1
    except RecordError as exc:
        at = getattr(exc, "at", None)
        return VerifyResult(
            ok=False,
            first_bad_seq=at if at is not None else count + 1,
            reason=getattr(exc, "reason", None) or str(exc),
            count=count,
        )
    return VerifyResult(ok=True, first_bad_seq=None, reason=None, count=count)


class Ledger:
    """One append-only ledger file, with the clock that stamps it.

    `clock` is required and has no default. It is meaning-bearing — it decides
    what every record claims about when it happened — and a default would let a
    test, or a future caller with its own notion of time, acquire the system
    clock by omission. Production callers pass `utc_clock`.

    A `Ledger` is safe to share between threads of one process and between
    processes on one machine; see the module docstring for what "advisory"
    costs you.
    """

    def __init__(self, path: str | os.PathLike[str], *, clock: Callable[[], datetime]) -> None:
        self.path = Path(path)
        self._clock = clock
        self._guard = threading.RLock()
        self._depth = 0
        self._lock_fd: int | None = None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Ledger({str(self.path)!r})"

    @contextmanager
    def _exclusive(self) -> Iterator[None]:
        """Hold the advisory write lock, re-entrantly within this instance.

        The re-entrancy matters: `retire` reads under the lock and then appends
        under it, and `flock` on a second file descriptor of the same file
        would block this process against itself forever.
        """
        with self._guard:
            if self._depth == 0:
                ensure_private_dir(self.path.parent)
                fd = os.open(ledger_lock_path(self.path), os.O_CREAT | os.O_RDWR, FILE_MODE)
                fcntl.flock(fd, fcntl.LOCK_EX)
                self._lock_fd = fd
            self._depth += 1
            try:
                yield
            finally:
                self._depth -= 1
                if self._depth == 0 and self._lock_fd is not None:
                    fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                    os.close(self._lock_fd)
                    self._lock_fd = None

    def records(self) -> list[Record]:
        """Every record, verified. Raises `LedgerCorrupt` at the first break."""
        return list(read(self.path))

    def last(self) -> Record | None:
        """The most recent record, or `None`."""
        return last(self.path)

    def verify(self) -> VerifyResult:
        """Walk the chain and report. Never raises for a broken chain."""
        return verify(self.path)

    def append(
        self,
        kind: RecordKind,
        payload: Mapping[str, Any],
        *,
        source: DataSource,
    ) -> Record:
        """Append one record and return it, or refuse to write at all.

        The whole chain is verified first, under the write lock. The brief this
        was built to asks only that the *tail* verify; verifying everything is
        the same code, catches an edit anywhere in the file rather than only at
        the end, and costs one pass over a file with one record per tick in it.
        The cost is real and is on the ledger as debt: this is O(n) per append,
        and a ledger large enough for that to matter needs a persisted tail
        index, not a weaker check.

        `source` is required and names where the DATA came from; it is written
        into the payload, under the hash, so provenance cannot be edited off a
        row that keeps verifying.
        """
        kind = RecordKind(kind)
        source = DataSource(source)
        body = dict(payload)
        declared = body.get(SOURCE_KEY)
        if declared is not None and declared != source.value:
            raise PayloadError(
                f"the payload already says source={declared!r} but this record is being "
                f"written as {source.value!r}; {SOURCE_KEY!r} is the reserved name for "
                f"where the data came from, and one record cannot claim two origins"
            )
        body[SOURCE_KEY] = source.value

        with self._exclusive():
            result = _verify_path(self.path, lock=False)
            if not result.ok:
                raise LedgerCorrupt(
                    f"{self.path} does not verify ({result.reason}), so it will not be "
                    f"extended. Nothing was written. A ledger that has been rewritten is "
                    f"evidence of that; appending to it would bury the fact",
                    at=result.first_bad_seq,
                )
            tail = self._tail_unlocked()
            record = Record.chained(
                seq=1 if tail is None else tail.seq + 1,
                ts=self._timestamp(),
                kind=kind,
                payload=body,
                prev_hash=GENESIS_PREV_HASH if tail is None else tail.hash,
            )
            self._write(record)
        return record

    def retire(self, seq: int, *, reason: str, source: DataSource) -> Record:
        """Supersede record `seq` by appending a `retired` record naming it.

        This is what "delete" is in an append-only ledger: the original row is
        untouched and still verifies, and a reader sees both what was claimed
        and that it was later withdrawn, with the reason and the time.
        """
        if not reason.strip():
            raise ValueError("a retirement must say why; a withdrawal with no reason is a gap")
        with self._exclusive():
            existing = {record.seq for record in self._records_unlocked()}
            if seq not in existing:
                raise LookupError(
                    f"record {seq} is not in {self.path}; there is nothing there to retire"
                )
            return self.append(
                RecordKind.RETIRED,
                {"retires_seq": seq, "reason": reason},
                source=source,
            )

    def _records_unlocked(self) -> list[Record]:
        """Every record, verified, without re-taking a lock we already hold."""
        return list(_scan(_read_text(self.path, lock=False) or ""))

    def _tail_unlocked(self) -> Record | None:
        tail: Record | None = None
        for record in _scan(_read_text(self.path, lock=False) or ""):
            tail = record
        return tail

    def _timestamp(self) -> datetime:
        moment = self._clock()
        if not isinstance(moment, datetime):
            raise PayloadError(
                f"the clock returned a {type(moment).__name__}; a record is stamped "
                f"with a timezone-aware datetime"
            )
        return moment

    def _write(self, record: Record) -> None:
        """Append one line and get it onto the disk before we claim it happened.

        `fsync` is not ceremony here. A record the runtime has reported and the
        operating system has not yet written is a fill the user was told about
        and no longer has evidence of.
        """
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, FILE_MODE)
        try:
            os.write(fd, encode_line(record).encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
