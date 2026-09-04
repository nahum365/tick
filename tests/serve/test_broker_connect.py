from __future__ import annotations

import threading
import time
from pathlib import Path

from tick.serve.broker_connect import BrokerConnectManager


class FakeSession:
    """Stands in for the MCP session: announces an authorization URL, then finishes."""

    def __init__(self, loopback, *, fail: bool, stored_grant: bool = False) -> None:
        self._loopback = loopback
        self._fail = fail
        self._stored_grant = stored_grant
        self.closed = False

    def open(self) -> None:
        if not self._stored_grant:
            self._loopback.authorization_url = "https://broker.invalid/authorize?state=abc"
        if self._fail:
            raise RuntimeError("the broker refused the code")

    def list_tools(self) -> list[str]:
        return ["a", "b", "c"]

    def close(self) -> None:
        self.closed = True


def build(tmp_path: Path, *, fail: bool, stored_grant: bool = False):
    outcomes: list[tuple[str, str | None, int | None]] = []
    finished = threading.Event()

    def on_finished(state: str, reason: str | None, tools: int | None) -> None:
        outcomes.append((state, reason, tools))
        finished.set()

    manager = BrokerConnectManager(
        home=tmp_path,
        timeout_seconds=5.0,
        announce_wait_seconds=5.0,
        session_factory=lambda _server, _storage, loopback, _timeout: FakeSession(
            loopback, fail=fail, stored_grant=stored_grant
        ),
        callback_received=lambda: None,
        on_finished=on_finished,
    )
    return manager, outcomes, finished


def test_a_finished_connection_reports_its_outcome_once(tmp_path: Path):
    manager, outcomes, finished = build(tmp_path, fail=False)
    started = manager.start(None, None)
    assert started["authorization_url"].startswith("https://broker.invalid/authorize")
    assert finished.wait(5.0)
    assert outcomes == [("succeeded", None, 3)]
    deadline = time.monotonic() + 5.0
    while manager.status(started["connect_id"])["state"] == "pending":
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert manager.status(started["connect_id"])["tools_discovered"] == 3


def test_a_failed_connection_reports_the_provider_sentence(tmp_path: Path):
    manager, outcomes, finished = build(tmp_path, fail=True)
    started = manager.start(None, None)
    assert finished.wait(5.0)
    assert outcomes == [("failed", "the broker refused the code", None)]
    assert manager.status(started["connect_id"])["state"] == "failed"


def test_record_broker_outcome_writes_a_ledger_row_without_credentials(tmp_path: Path):
    from tick.records import read
    from tick.serve.handlers import record_broker_outcome

    record_broker_outcome(
        tmp_path, state="failed", reason="the token endpoint said 400", tools=None
    )
    record_broker_outcome(tmp_path, state="succeeded", reason=None, tools=7)
    rows = list(read(tmp_path / "broker/records.jsonl"))
    assert [row.payload["event"] for row in rows] == [
        "broker_connection_failed",
        "broker_connection_succeeded",
    ]
    assert rows[0].payload["reason"] == "the token endpoint said 400"
    assert "tools_discovered" not in rows[0].payload
    assert rows[1].payload["tools_discovered"] == 7
    assert rows[1].payload["via"] == "loopback"


def test_a_stored_grant_connects_without_a_ceremony(tmp_path: Path):
    """Live 2026-09-04: after a real grant, the next connect had no URL to issue and
    was refused as "the broker did not issue an authorization URL"."""
    manager, outcomes, finished = build(tmp_path, fail=False, stored_grant=True)
    started = manager.start(None, None)
    assert finished.wait(5.0)
    assert started["authorization_url"] is None
    assert started["state"] == "succeeded"
    assert started["tools_discovered"] == 3
    assert outcomes == [("succeeded", None, 3)]
