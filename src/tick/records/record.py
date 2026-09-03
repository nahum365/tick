"""One entry in the record: what it says, where its data came from, how it chains.

A `Record` is a frozen statement that something happened, and it carries its own
integrity. The hash is computed over the canonical encoding of

    {"seq", "ts", "kind", "payload", "prev_hash"}

and a `Record` whose `hash` is not that value cannot be constructed — not from
a file, not from Python. Chaining `prev_hash` to the previous record's `hash`
is what makes the file tamper-*evident*: changing anything about record 7
changes its hash, which record 8 quotes, which record 9 quotes, so a single
edited byte invalidates the whole tail (CLAUDE.md invariant 4).

Tamper-EVIDENT, not tamper-proof, and the difference is worth stating plainly:
anyone who can write the file can rewrite the whole chain from the edit
forward, because the hash uses no secret. What the chain gives is that no
*surgical* edit survives, that truncation is visible as a shortened sequence,
and that the runtime refuses to append to a file that no longer verifies. A
signed or externally-anchored chain is a later question, and it is a different
one.

**Provenance travels with the row.** Every payload carries a top-level
`source` — `fixture`, `paper` or `robinhood` — naming where the DATA in it came
from. The ledger is local-only: it lives under `TICK_HOME` on the user's own
machine and nothing in this package transmits it anywhere. `source` is what
lets a later export or support-bundle tool refuse to ship the rows derived from
Robinhood's Trading MCP, which under the 2026-08-31 terms audit never leave the
user's box. It sits INSIDE the payload rather than beside it so that it is
covered by the hash: a provenance label that could be edited without breaking
the chain would be a label nobody could rely on.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any

from pydantic import AwareDatetime, BaseModel, BeforeValidator, ConfigDict, model_validator

from tick.spec import canonical_dumps, canonical_encode, sha256_hex

from .errors import LedgerCorrupt, PayloadError
from .payload import canonical_timestamp, normalize_payload

__all__ = [
    "GENESIS_PREV_HASH",
    "DataSource",
    "Record",
    "RecordKind",
    "decode_line",
    "encode_line",
    "record_hash",
]

#: What the first record in a chain quotes as its predecessor. A fixed
#: constant, domain-tagged rather than sixty-four zeros: zeros are what a
#: half-written file or a lazy forgery contains by accident, and this value
#: also names the chain format, so a future change to the hashed shape gets a
#: new genesis and every old ledger stops verifying loudly instead of quietly
#: re-interpreting.
GENESIS_PREV_HASH = sha256_hex(b"tick.records.chain.v1.genesis")

#: The key inside every payload that names where its data came from.
SOURCE_KEY = "source"

_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")

#: Every field a line must carry to be a record at all.
RECORD_FIELDS = frozenset({"seq", "ts", "kind", "payload", "prev_hash", "hash"})


class RecordKind(StrEnum):
    """What a record is a record OF. A closed vocabulary, on purpose.

    A new kind is a change to what the runtime claims it did, so it is a code
    change and a review, never a free-form string a caller invents at the call
    site.
    """

    DECISION = "decision"
    ORDER = "order"
    FILL = "fill"
    REJECTED = "rejected"
    REFUSAL = "refusal"
    MODE_CHANGE = "mode_change"
    STOP = "stop"
    NOTE = "note"
    #: Retirement. Nothing is ever deleted or edited; a row that no longer
    #: applies is superseded by a `retired` record naming it.
    RETIRED = "retired"


class DataSource(StrEnum):
    """Where the data in a payload came from.

    Not who wrote the record — the runtime always did — but whose numbers are
    in it: a market fixture the user pointed us at, Tick's own local paper
    simulation, Robinhood's Trading MCP, or nothing outside the runtime at all.

    `RUNTIME` is that last case, and it is a source rather than an absence:
    a stop, a mode change, or a note about the ledger's own state carries no
    market or account data, and labelling one `paper` would say a simulation
    produced a number that no simulation touched.
    """

    FIXTURE = "fixture"
    PAPER = "paper"
    ROBINHOOD = "robinhood"
    RUNTIME = "runtime"

    @property
    def derived_from_robinhood(self) -> bool:
        """True for data obtained through Robinhood's Trading MCP.

        Quotes, positions, balances and orders read through the MCP stay on the
        user's machine (2026-08-31 terms audit). Nothing in this package
        transmits a record anywhere; this property is how a later export tool
        says which rows it must refuse, without re-deriving the rule.
        """
        return self is DataSource.ROBINHOOD


def _normalized_payload(value: Any) -> Any:
    """Normalize a payload on the way in, so a `Record` is always plain JSON."""
    if isinstance(value, Mapping):
        return normalize_payload(value)
    raise PayloadError(
        f"a record payload is a JSON object with named fields, not a {type(value).__name__}"
    )


RecordPayload = Annotated[dict[str, Any], BeforeValidator(_normalized_payload)]


def record_hash(
    *,
    seq: int,
    ts: datetime,
    kind: RecordKind,
    payload: dict[str, Any],
    prev_hash: str,
) -> str:
    """The hash of a record: SHA-256 over the canonical encoding of its body.

    The body excludes `hash` itself, and includes everything else. `ts` is
    hashed in its `canonical_timestamp` form, so the number in the file and the
    number in the preimage are the same characters.
    """
    return sha256_hex(
        canonical_encode(
            {
                "seq": seq,
                "ts": canonical_timestamp(ts),
                "kind": RecordKind(kind).value,
                "payload": payload,
                "prev_hash": prev_hash,
            }
        )
    )


class Record(BaseModel):
    """One line of the ledger. Frozen, closed, and self-verifying."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seq: int
    ts: AwareDatetime
    kind: RecordKind
    payload: RecordPayload
    prev_hash: str
    hash: str

    @model_validator(mode="after")
    def _check(self) -> Record:
        if self.seq < 1:
            raise ValueError(f"seq ({self.seq}) starts at 1 and only ever increases")
        for name in ("prev_hash", "hash"):
            value = getattr(self, name)
            if not _HEX64.match(value):
                raise ValueError(f"{name} ({value!r}) is not a lowercase sha256 hex digest")
        if self.seq == 1 and self.prev_hash != GENESIS_PREV_HASH:
            raise ValueError(
                "the first record chains to the genesis hash; a chain that starts "
                "anywhere else is a fragment of another ledger"
            )
        if self.seq > 1 and self.prev_hash == GENESIS_PREV_HASH:
            raise ValueError(
                f"record {self.seq} chains to the genesis hash, which only record 1 may do"
            )
        if SOURCE_KEY not in self.payload:
            raise ValueError(
                f"record {self.seq} has no {SOURCE_KEY!r}: every payload names where its "
                f"data came from, so an export can tell Robinhood-derived rows apart"
            )
        try:
            DataSource(self.payload[SOURCE_KEY])
        except ValueError:
            raise ValueError(
                f"record {self.seq}: {self.payload[SOURCE_KEY]!r} is not a data source "
                f"({', '.join(source.value for source in DataSource)})"
            ) from None
        expected = record_hash(
            seq=self.seq,
            ts=self.ts,
            kind=self.kind,
            payload=self.payload,
            prev_hash=self.prev_hash,
        )
        if self.hash != expected:
            raise ValueError(
                f"record {self.seq}: hash {self.hash} does not match its contents "
                f"(expected {expected})"
            )
        return self

    @classmethod
    def chained(
        cls,
        *,
        seq: int,
        ts: datetime,
        kind: RecordKind,
        payload: dict[str, Any],
        prev_hash: str,
    ) -> Record:
        """Build the record that follows `prev_hash`, computing its own hash."""
        normalized = normalize_payload(payload)
        return cls(
            seq=seq,
            ts=ts,
            kind=kind,
            payload=normalized,
            prev_hash=prev_hash,
            hash=record_hash(seq=seq, ts=ts, kind=kind, payload=normalized, prev_hash=prev_hash),
        )

    @property
    def source(self) -> DataSource:
        """Where the data in this record came from."""
        return DataSource(self.payload[SOURCE_KEY])

    def body(self) -> dict[str, Any]:
        """The hashed part of the record, as plain JSON."""
        return {
            "seq": self.seq,
            "ts": canonical_timestamp(self.ts),
            "kind": self.kind.value,
            "payload": self.payload,
            "prev_hash": self.prev_hash,
        }

    def as_line(self) -> str:
        """The record's one canonical line, without its newline."""
        return canonical_dumps(self.body() | {"hash": self.hash})


def encode_line(record: Record) -> str:
    """The canonical JSONL line for `record`, newline included."""
    return record.as_line() + "\n"


def decode_line(line: str, *, at: int) -> Record:
    """Parse one ledger line, refusing anything that is not exactly canonical.

    `at` is the 1-based position of the line in the file; it is used in errors
    because a `seq` read out of a line nobody has verified yet cannot be quoted
    as fact.

    Two checks, and the second is the unusual one. The record must hash to its
    own `hash` — and the line must be *byte-identical* to the canonical
    encoding of what it decoded to. Without the second check a line could be
    re-indented, have its keys reordered, or carry a duplicate key, and still
    decode to a record that verifies. The record is evidence; the bytes are
    part of it, so the byte check stays.

    What the byte check may NOT do is announce a conclusion it has not
    reached. "Not in canonical form" means the bytes are not the ones Tick
    writes; it does not distinguish a line rewritten by an editor from a torn
    write that happened to leave parseable text, and it is not the chain
    failing — so the refusal says the first and disclaims the other two. The
    chain's own failure has its own sentence in `ledger._scan`.
    """
    text = line.rstrip("\n")
    try:
        document = json.loads(text, parse_float=_refuse_float)
    except PayloadError as exc:
        raise LedgerCorrupt(str(exc), at=at) from exc
    except json.JSONDecodeError as exc:
        raise LedgerCorrupt(f"the line is not JSON: {exc}", at=at) from exc
    if not isinstance(document, dict):
        raise LedgerCorrupt(
            f"the line is a {type(document).__name__}; every ledger line is a JSON object",
            at=at,
        )
    missing = RECORD_FIELDS - set(document)
    if missing:
        raise LedgerCorrupt(
            f"the line is missing {', '.join(sorted(missing))}; it is not a record", at=at
        )
    try:
        record = Record.model_validate(document)
    except PayloadError as exc:
        raise LedgerCorrupt(str(exc), at=at) from exc
    except ValueError as exc:
        raise LedgerCorrupt(_first_problem(exc), at=at) from exc
    if record.as_line() != text:
        raise LedgerCorrupt(
            "the line is not in canonical form: its bytes are not the encoding Tick "
            "writes for the record they decode to, so the line was either rewritten "
            "in place or torn by a partial write that still parses. This is not a "
            "chain failure — the record's own hash and its link to the line before "
            "it were not what failed here",
            at=at,
        )
    return record


def _first_problem(exc: ValueError) -> str:
    """The readable half of a pydantic failure: the first message, unprefixed.

    `LedgerCorrupt` puts the line's position in front of whatever it is given,
    and the validator's own messages already name the record, so the leading
    `record 7: ` is dropped here rather than said twice.
    """
    errors = getattr(exc, "errors", None)
    message = str(exc)
    if callable(errors):
        entries = errors()
        if entries:
            message = str(entries[0]["msg"])
    for noise in ("Value error, ", "Assertion failed, "):
        if message.startswith(noise):
            message = message[len(noise) :]
    return re.sub(r"\Arecord \d+: ", "", message)


def _refuse_float(text: str) -> Any:
    """A JSON number with a decimal point has no place in a ledger line.

    Every exact number is written as a string (`"12.50"`), so a bare `12.5` in
    a ledger line either was never written by Tick or was written by an editor.
    Parsing it as a float would round it; parsing it as a Decimal would accept
    a shape the writer never produces.
    """
    raise PayloadError(
        f"a ledger line contains the bare number {text}; exact numbers are "
        f"recorded as strings, so this line was not written by Tick"
    )
