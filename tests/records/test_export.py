"""Evidence export retains hashes and removes private brokerage content."""

from __future__ import annotations

import json
import stat

from tick.records import DataSource, Ledger, RecordKind, export_evidence


def test_evidence_export_redacts_broker_rows_and_account_fields(ledger_path, clock, tmp_path):
    ledger = Ledger(ledger_path, clock=clock)
    broker = ledger.append(
        RecordKind.FILL,
        {"account_id": "private-account", "price": "17.25", "symbol": "XYZ"},
        source=DataSource.ROBINHOOD,
    )
    ledger.append(
        RecordKind.NOTE,
        {"event": "local", "account_id": "private-account", "safe": "kept"},
        source=DataSource.RUNTIME,
    )
    destination = tmp_path / "evidence" / "agent.jsonl"

    export_evidence(ledger_path, destination)

    rows = [json.loads(line) for line in destination.read_text().splitlines()]
    assert rows[0] == {
        "at": broker.ts.isoformat(),
        "kind": "fill",
        "record_hash": broker.hash,
        "redacted": "derived_from_robinhood",
        "seq": 1,
    }
    assert rows[1]["payload"]["safe"] == "kept"
    assert "account_id" not in rows[1]["payload"]
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
