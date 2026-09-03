"""Fixtures for the connect ceremony's tests.

Nothing here reaches a network. The loopback tests bind `127.0.0.1` on an
ephemeral port and drive it from the same process, which is a socket on the
test machine and not a request to anybody; the token-store tests write only
under `tmp_path`. `TICK_HOME` is redirected for the whole package, autouse,
because the failure it prevents — a test overwriting the developer's own
Robinhood grant — is silent and unrecoverable.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from tick.records import TICK_HOME_ENV


@pytest.fixture(autouse=True)
def no_real_tick_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    home = tmp_path / "tick-home"
    monkeypatch.setenv(TICK_HOME_ENV, str(home))
    yield home


@pytest.fixture
def home(no_real_tick_home: Path) -> Path:
    return no_real_tick_home
