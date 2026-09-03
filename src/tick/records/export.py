"""A local evidence copy that cannot disclose brokerage-derived data.

The original ledger remains the authority.  An evidence export is a review
artifact: rows sourced from the brokerage are replaced by their kind and
existing record hash, while account-shaped fields are removed from every
other row.  It is deliberately not presented as another valid hash chain.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .home import write_private_file
from .ledger import read, verify
from .record import DataSource, Record

__all__ = ["evidence_rows", "export_evidence"]

_ACCOUNT_KEY = re.compile(r"account", re.IGNORECASE)


def _without_accounts(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_accounts(item)
            for key, item in value.items()
            if not _ACCOUNT_KEY.search(str(key))
        }
    if isinstance(value, list):
        return [_without_accounts(item) for item in value]
    return value


def evidence_rows(records: list[Record]) -> list[dict[str, Any]]:
    """Return review rows with brokerage content represented only by hashes."""
    rows: list[dict[str, Any]] = []
    for record in records:
        source = DataSource(record.payload["source"])
        if source.derived_from_robinhood:
            rows.append(
                {
                    "seq": record.seq,
                    "at": record.ts.isoformat(),
                    "kind": record.kind.value,
                    "record_hash": record.hash,
                    "redacted": "derived_from_robinhood",
                }
            )
            continue
        rows.append(
            _without_accounts(
                {
                    "seq": record.seq,
                    "at": record.ts.isoformat(),
                    "kind": record.kind.value,
                    "payload": record.payload,
                    "record_hash": record.hash,
                }
            )
        )
    return rows


def export_evidence(ledger_path: Path, destination: Path) -> Path:
    """Write a private redacted copy, refusing a ledger that does not verify."""
    result = verify(ledger_path)
    if not result.ok:
        raise ValueError(
            f"{result}. No evidence copy was written; start a successor ledger and export again."
        )
    text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in evidence_rows(list(read(ledger_path)))
    )
    return write_private_file(destination, text)
