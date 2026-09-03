"""Errors the record raises — the conditions under which it will not be written.

The record is the one thing in Tick that has to be trustworthy after the fact.
So every failure here is loud and none of them is recoverable in code: there is
no repair path, no "skip the bad line and carry on", and no API that edits or
deletes anything (CLAUDE.md invariant 4). A ledger whose chain does not verify
stops being appended to, and what happens next is a human decision made with
the broken file in front of them.

That direction is deliberate and it has a cost worth naming: a torn write from
a crash quarantines the file. The fail-safe question — after the failure, what
can the user still DO? — is answered by the runtime, which stops and notifies
rather than retrying (invariant 8), and by the fact that a new ledger can
always be started beside the old one. It is never answered by silently
repairing the evidence.
"""

from __future__ import annotations


class RecordError(Exception):
    """Base class for every record failure."""


class PayloadError(RecordError):
    """A payload carries something that cannot be recorded exactly.

    A binary float is the case that matters: `0.1` is not the number anybody
    wrote, and a record of a decision is worthless if the numbers in it are
    approximations of the ones the decision was made on (invariant 5).
    """


class LedgerCorrupt(RecordError):
    """A ledger line does not verify, so the ledger will not be extended.

    `at` is the 1-based position of the offending LINE in the file — which in
    an intact ledger is also its `seq`, but a line nobody has verified yet
    cannot have its own `seq` quoted back as fact. So the message says "line
    3": the position is something the reader can count, while the seq is
    something the file claims.

    The refusal always says WHICH failure it is, because they are different
    events and they ask different things of the person holding the file:

    - **not in canonical form** — the bytes are not the one encoding Tick
      writes for this record. Something rewrote the line in place, or a
      partial write left text that still parses. This says nothing about the
      chain: the record may be perfectly self-consistent.
    - **hash does not match its predecessor** — the line is well-formed and
      self-consistent, and the record before it is not the one it was written
      after. A line was edited, removed or reordered at or before this point.
    - **hash does not match its contents** — this line's own body and its own
      hash disagree, which is a field edited inside this line.

    Raised by `read` and by `append`; never by `verify`, which answers with a
    `VerifyResult`, because "is this file intact?" is a question somebody
    asked, not an accident.
    """

    def __init__(self, reason: str, *, at: int | None) -> None:
        self.reason = reason
        self.at = at
        super().__init__(reason if at is None else f"line {at}: {reason}")


class LedgerFormatError(RecordError):
    """The file is not a Tick ledger at all — it does not even decode as text."""
