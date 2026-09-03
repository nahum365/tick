"""The record — append-only, hash-chained, and local to the user's machine.

    from tick.records import DataSource, Ledger, RecordKind, utc_clock, verify

    ledger = Ledger(agent_ledger_path(home, "dip-buyer"), clock=utc_clock)
    ledger.append(RecordKind.FILL, {"order_id": "o-1"}, source=DataSource.PAPER)
    verify(ledger.path).ok

Every decision, order, fill, rejection, refusal, mode change and stop the
runtime produces is appended here as one line of JSON, each line quoting the
hash of the line before it. Nothing is ever edited and nothing is ever deleted:
a record that no longer applies is superseded by a `retired` record naming it
(CLAUDE.md invariant 4). There is no delete function, no edit function and no
compaction in this package — the absence is the design, and
`tests/records/test_no_mutation_api.py` fails if one appears.

CLAUDE.md invariants this package carries:

- **4, append-only and tamper-evident.** The chain is `sha256` over the
  canonical encoding of `{seq, ts, kind, payload, prev_hash}`, sharing one
  encoder with `spec_id`. Any edited byte anywhere invalidates the record it is
  in and every record after it, and `append` refuses to extend a file that does
  not verify — a broken record stops the runtime instead of growing a plausible
  tail.
- **5, no number is fabricated.** `Decimal` is written as its exact string; a
  binary float is refused, going in and coming out, rather than rounded into
  permanent evidence.
- **1, and the terms audit: nothing leaves the box.** The ledger is a local
  file under `TICK_HOME`. This package opens no socket and builds no client;
  every payload names its `source` (`fixture` / `paper` / `robinhood`) so that
  a later export tool can refuse to ship Robinhood-derived rows.
- **No silent meaning-bearing defaults.** `Ledger` requires its `clock`,
  `append` requires its `source`, and `tick_home` requires the environment it
  reads.
"""

from __future__ import annotations

from .errors import (
    LedgerCorrupt,
    LedgerFormatError,
    PayloadError,
    RecordError,
)
from .export import evidence_rows, export_evidence
from .home import (
    DEFAULT_TICK_HOME,
    PRIVATE_FILE_MODE,
    TICK_HOME_ENV,
    agent_ledger_path,
    ensure_private_dir,
    tick_home,
    write_private_file,
)
from .ledger import Ledger, VerifyResult, last, ledger_lock_path, read, utc_clock, verify
from .payload import MAX_PAYLOAD_DEPTH, canonical_timestamp, normalize_payload, normalize_value
from .record import (
    GENESIS_PREV_HASH,
    RECORD_FIELDS,
    SOURCE_KEY,
    DataSource,
    Record,
    RecordKind,
    decode_line,
    encode_line,
    record_hash,
)

__all__ = [
    "DEFAULT_TICK_HOME",
    "GENESIS_PREV_HASH",
    "MAX_PAYLOAD_DEPTH",
    "RECORD_FIELDS",
    "SOURCE_KEY",
    "TICK_HOME_ENV",
    "DataSource",
    "Ledger",
    "LedgerCorrupt",
    "LedgerFormatError",
    "PayloadError",
    "Record",
    "RecordError",
    "RecordKind",
    "VerifyResult",
    "agent_ledger_path",
    "canonical_timestamp",
    "decode_line",
    "encode_line",
    "PRIVATE_FILE_MODE",
    "ensure_private_dir",
    "evidence_rows",
    "export_evidence",
    "last",
    "ledger_lock_path",
    "normalize_payload",
    "normalize_value",
    "read",
    "record_hash",
    "tick_home",
    "utc_clock",
    "verify",
    "write_private_file",
]
