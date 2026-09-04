from __future__ import annotations

import io
from datetime import UTC, datetime

import pytest

from tick.serve.browser_bridge import BrowserBridgeError, Viewport
from tick.serve.provider_login import (
    ProviderBrowserLoginManager,
    ProviderLoginError,
    ProviderLoginManager,
)


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


def test_browser_login_runs_the_plain_flow_and_closes_when_codex_succeeds():
    argv = []

    class Completed:
        stdout = io.StringIO(
            "\x1b[34mOpen https://login.example.invalid/authorize?private=value\x1b[0m\n"
        )
        stderr = io.StringIO("")

        def poll(self):
            return 0

    class Bridge:
        def __init__(self):
            self.closed = []

        def open(self, url, viewport, purpose):
            assert url == "https://login.example.invalid/authorize?private=value"
            assert viewport == Viewport(390, 760)
            assert purpose == "provider_login"
            return {
                "session_id": "browser-1",
                "origin": "https://login.example.invalid",
            }

        def close(self, session_id, reason):
            self.closed.append((session_id, reason))

    bridge = Bridge()
    manager = ProviderBrowserLoginManager(
        binary="codex",
        start_process=lambda command: argv.append(list(command)) or Completed(),
        bridge=bridge,
        capture_timeout_seconds=0.2,
    )
    started = manager.start(Viewport(390, 760))
    for _ in range(100):
        if bridge.closed:
            break
        __import__("time").sleep(0.001)

    assert argv == [["codex", "login"]]
    assert started["session_id"] == "browser-1"
    assert manager.status(started["login_id"])["state"] == "succeeded"
    assert bridge.closed == [("browser-1", "login_succeeded")]


def test_browser_login_refusals_preserve_a_next_step_and_bridge_code():
    class NoURL:
        stdout = io.StringIO("browser flow changed")
        stderr = io.StringIO("")

        def poll(self):
            return None

    class RefusingBridge:
        def open(self, _url, _viewport, _purpose):
            raise BrowserBridgeError(
                "BROWSER_SESSION_BUSY",
                "this box already has a browser open. Close it and retry.",
            )

    unreadable = ProviderBrowserLoginManager(
        binary="codex",
        start_process=lambda _command: NoURL(),
        bridge=RefusingBridge(),
        capture_timeout_seconds=0.01,
    )
    with pytest.raises(ProviderLoginError) as output:
        unreadable.start(Viewport(390, 760))
    assert output.value.code == "CODEX_LOGIN_OUTPUT_UNREADABLE"
    assert output.value.reason.endswith("Run `codex login` on the box and retry there.")

    class HasURL(NoURL):
        stdout = io.StringIO("Open https://login.example.invalid/authorize\n")

    busy = ProviderBrowserLoginManager(
        binary="codex",
        start_process=lambda _command: HasURL(),
        bridge=RefusingBridge(),
        capture_timeout_seconds=0.1,
    )
    with pytest.raises(ProviderLoginError) as bridge:
        busy.start(Viewport(390, 760))
    assert bridge.value.code == "BROWSER_SESSION_BUSY"
    assert bridge.value.reason.endswith("Close it and retry.")

    with pytest.raises(ProviderLoginError) as missing:
        busy.status("missing")
    assert missing.value.code == "CODEX_LOGIN_NOT_FOUND"
    assert missing.value.reason.endswith("Start browser login again.")
