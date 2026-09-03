"""Shared fixtures for the record tests.

Two things are true of every test in this package. It writes only under
`tmp_path`, never under a real `TICK_HOME` and never under `~/.tick` — the
`no_real_tick_home` fixture makes that a guarantee rather than a convention.
And it stamps records from an injected clock, so the timestamps in the assertions
are the ones the test chose.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tick.records import TICK_HOME_ENV, Ledger

#: The moment the first record of every test is stamped with.
START = datetime(2026, 9, 1, 13, 30, tzinfo=UTC)


class StepClock:
    """A clock that advances one minute per reading, and says so.

    Not `datetime.now`: a record's `ts` is a claim about when something
    happened, and a test that asserts on a claim it did not make is asserting on
    the machine's clock.
    """

    def __init__(self, start: datetime = START, step: timedelta = timedelta(minutes=1)) -> None:
        self.next = start
        self.step = step
        self.readings: list[datetime] = []

    def __call__(self) -> datetime:
        moment = self.next
        self.next = moment + self.step
        self.readings.append(moment)
        return moment


@pytest.fixture(autouse=True)
def no_real_tick_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Point `TICK_HOME` at a temporary directory for the whole test session.

    Autouse, because the failure it prevents — a test appending to the
    developer's own agent record — is silent, permanent and append-only.
    """
    monkeypatch.setenv(TICK_HOME_ENV, str(tmp_path / "tick-home"))
    yield


@pytest.fixture
def clock() -> StepClock:
    return StepClock()


@pytest.fixture
def ledger_path(tmp_path: Path) -> Path:
    return tmp_path / "agents" / "placeholder-agent" / "records.jsonl"


@pytest.fixture
def ledger(ledger_path: Path, clock: StepClock) -> Ledger:
    return Ledger(ledger_path, clock=clock)
