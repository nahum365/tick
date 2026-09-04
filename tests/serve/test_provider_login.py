from __future__ import annotations

import io
from datetime import UTC, datetime

import pytest

from tick.serve.provider_login import ProviderLoginError, ProviderLoginManager


class Help:
    returncode = 0
    stdout = "usage: codex login --device-auth"
    stderr = ""


class Process:
    stdout = io.StringIO("Open https://login.example.invalid/device\nEnter code: ABCD-WXYZ\n")
    stderr = io.StringIO("")

    def poll(self):
        return None


def test_device_login_discovers_the_flag_and_returns_only_the_user_ceremony():
    argv = []
    manager = ProviderLoginManager(
        binary="codex",
        help_run=lambda command: Help(),
        start_process=lambda command: argv.append(list(command)) or Process(),
        now=lambda: datetime(2026, 9, 3, 12, 0, tzinfo=UTC),
        capture_timeout_seconds=0.2,
    )

    started = manager.start()

    assert argv == [["codex", "login", "--device-auth"]]
    assert started["url"] == "https://login.example.invalid/device"
    assert started["code"] == "ABCD-WXYZ"
    assert manager.status(started["login_id"])["state"] == "pending"


def test_changed_login_help_refuses_with_the_exact_stderr():
    class Changed:
        returncode = 2
        stdout = ""
        stderr = "unknown login interface\nuse browser flow"

    manager = ProviderLoginManager(
        binary="codex",
        help_run=lambda _command: Changed(),
        start_process=lambda _command: Process(),
        now=lambda: datetime(2026, 9, 3, 12, 0, tzinfo=UTC),
        capture_timeout_seconds=0.2,
    )

    with pytest.raises(ProviderLoginError) as refused:
        manager.start()
    assert "unknown login interface\nuse browser flow" in refused.value.reason
    assert "update Tick" in refused.value.reason


def test_real_codex_output_with_colour_escapes_yields_a_clean_url_and_the_one_time_code():
    """Recorded from codex-cli 0.149.0 on 2026-09-04 (code replaced, same shape).

    The first live login failed because the captured URL kept a trailing colour
    escape and the code regex matched the word "authorization".
    """
    from pathlib import Path

    recorded = (
        Path(__file__).parent / "fixtures" / "codex_login_device_auth_0_149_0.txt"
    ).read_text(encoding="utf-8")

    class Recorded:
        stdout = io.StringIO(recorded)
        stderr = io.StringIO("")

        def poll(self):
            return None

    manager = ProviderLoginManager(
        binary="codex",
        help_run=lambda _command: Help(),
        start_process=lambda _command: Recorded(),
        now=lambda: datetime(2026, 9, 4, 1, 0, tzinfo=UTC),
        capture_timeout_seconds=0.5,
    )

    started = manager.start()

    assert started["url"] == "https://auth.openai.com/codex/device"
    assert started["code"] == "WXYZ-ABCDE"
