from __future__ import annotations

import threading
import time
from pathlib import Path

from tick.serve.broker_connect import BrokerConnectManager


class FakeSession:
    """Stands in for the MCP session: announces an authorization URL, then finishes."""

    def __init__(self, loopback, *, fail: bool) -> None:
        self._loopback = loopback
        self._fail = fail
        self.closed = False

    def open(self) -> None:
        self._loopback.authorization_url = "https://broker.invalid/authorize?state=abc"
        if self._fail:
            raise RuntimeError("the broker refused the code")

    def list_tools(self) -> list[str]:
        return ["a", "b", "c"]

    def close(self) -> None:
        self.closed = True


def build(tmp_path: Path, *, fail: bool):
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
            loopback, fail=fail
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
