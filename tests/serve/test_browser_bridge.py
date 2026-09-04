from __future__ import annotations

import os
import stat
import threading
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from tick.records import read
from tick.serve import browser_bridge as module
from tick.serve.browser_bridge import (
    BROWSER_FRAME_QUEUE_SIZE,
    BROWSER_JPEG_QUALITY,
    BROWSER_LIFETIME_SECONDS,
    BrowserBridge,
    BrowserBridgeError,
    Viewport,
)


class FakeDisplay:
    def __init__(self) -> None:
        self.started: list[Viewport] = []
        self.stopped: list[int] = []

    def start(self, viewport: Viewport) -> int:
        self.started.append(viewport)
        return 91

    def stop(self, pid: int) -> None:
        self.stopped.append(pid)


class FakeBrowser:
    sandbox_mode = "suid_helper"

    def __init__(self) -> None:
        self.launched: list[tuple[str, Viewport, Path]] = []
        self.terminated: list[int] = []
        self.rss_values = iter((Decimal("82.5"), Decimal("91.25")))

    def launch(self, url: str, viewport: Viewport, profile: Path) -> int:
        self.launched.append((url, viewport, profile))
        return 92

    def rss_mb(self, pid: int):
        assert pid == 92
        return next(self.rss_values)

    def terminate(self, pid: int) -> None:
        self.terminated.append(pid)


class FakeCapture:
    def __init__(self) -> None:
        self.calls = 0

    def capture(self) -> bytes:
        self.calls += 1
        return b"\xff\xd8fixture-jpeg\xff\xd9"


class FakeInput:
    def __init__(self) -> None:
        self.events: list[tuple[object, ...]] = []

    def tap(self, x: int, y: int) -> None:
        self.events.append(("tap", x, y))

    def scroll(self, x: int, y: int, dy: int) -> None:
        self.events.append(("scroll", x, y, dy))

    def text(self, value: str) -> None:
        self.events.append(("text", value))

    def key(self, value: str) -> None:
        self.events.append(("key", value))

    def back(self) -> None:
        self.events.append(("back",))


def build_bridge(tmp_path: Path, *, monotonic=None, sleeper=None):
    display = FakeDisplay()
    browser = FakeBrowser()
    capture = FakeCapture()
    input_port = FakeInput()
    bridge = BrowserBridge(
        home=tmp_path / "tick-home",
        display=display,
        browser=browser,
        capture=capture,
        input_port=input_port,
        now=lambda: datetime(2026, 9, 4, 12, 0, tzinfo=UTC),
        monotonic=monotonic or (lambda: Decimal("100")),
        sleeper=sleeper or (lambda _seconds: threading.Event().wait(0.01)),
    )
    return bridge, display, browser, capture, input_port


def test_open_stream_input_and_close_leave_only_bounded_pixels_in_memory(tmp_path: Path):
    bridge, display, browser, capture, input_port = build_bridge(tmp_path)
    opened = bridge.open(
        "https://login.example.invalid/authorize?private=not-recorded",
        Viewport(390, 760),
        "broker_connect",
    )
    frame = next(bridge.frames(opened["session_id"]))
    assert frame == (100_000, b"\xff\xd8fixture-jpeg\xff\xd9", "https://login.example.invalid")
    assert bridge._active is not None
    assert bridge._active.frames.maxsize == BROWSER_FRAME_QUEUE_SIZE

    bridge.input(
        opened["session_id"],
        [
            {"kind": "tap", "x": 12, "y": 34},
            {"kind": "scroll", "x": 20, "y": 30, "dy": -2},
            {"kind": "text", "s": "private value"},
            {"kind": "key", "key": "Return"},
            {"kind": "back"},
        ],
    )
    bridge.close(opened["session_id"], "user_closed")

    assert input_port.events == [
        ("tap", 12, 34),
        ("scroll", 20, 30, -2),
        ("text", "private value"),
        ("key", "Return"),
        ("back",),
    ]
    assert browser.terminated == [92]
    assert display.stopped == [91]
    assert not browser.launched[0][2].exists()
    rows = list(read(tmp_path / "tick-home/browser/records.jsonl"))
    assert rows[0].payload | {"source": "runtime"} == {
        "event": "browser_session_opened",
        "purpose": "broker_connect",
        "host": "login.example.invalid",
        "rss_mb": "82.5",
        "sandbox": "suid_helper",
        "source": "runtime",
    }
    assert rows[1].payload["event"] == "browser_session_closed"
    assert rows[1].payload["rss_mb"] == "91.25"
    ledger_text = (tmp_path / "tick-home/browser/records.jsonl").read_text()
    assert "private=not-recorded" not in ledger_text
    assert "private value" not in ledger_text


def test_a_second_session_and_invalid_events_refuse_with_the_next_action(tmp_path: Path):
    bridge, *_ = build_bridge(tmp_path)
    opened = bridge.open("https://login.example.invalid", Viewport(390, 760), "provider_login")
    with pytest.raises(BrowserBridgeError) as busy:
        bridge.open("https://login.example.invalid", Viewport(390, 760), "provider_login")
    assert busy.value.code == "BROWSER_SESSION_BUSY"
    assert busy.value.reason.endswith("Close it before starting another.")

    refused = (
        ({"kind": "tap", "x": 390, "y": 0}, "latest frame coordinates"),
        ({"kind": "scroll", "x": 0, "y": 0, "dy": 0}, "send the gesture again"),
        ({"kind": "text", "s": "x" * 513}, "send the gesture again"),
        ({"kind": "key", "key": "Delete"}, "send the gesture again"),
        ({"kind": "unknown"}, "send the gesture again"),
    )
    for event, sentence in refused:
        with pytest.raises(BrowserBridgeError) as error:
            bridge.input(opened["session_id"], [event])
        assert error.value.code == "BROWSER_EVENT_INVALID"
        assert sentence.lower() in error.value.reason.lower()
    bridge.close(opened["session_id"], "test_finished")


def test_lifetime_expires_and_kills_both_processes(tmp_path: Path):
    moments = iter(
        (Decimal(0), Decimal(BROWSER_LIFETIME_SECONDS), Decimal(BROWSER_LIFETIME_SECONDS))
    )
    bridge, display, browser, capture, _ = build_bridge(tmp_path, monotonic=lambda: next(moments))
    opened = bridge.open("https://login.example.invalid", Viewport(390, 760), "broker_connect")
    assert list(bridge.frames(opened["session_id"])) == []
    assert bridge.close_reason(opened["session_id"]) == "expired"
    assert capture.calls == 0
    assert browser.terminated == [92]
    assert display.stopped == [91]


class FakeProcess:
    def __init__(self, pid: int) -> None:
        self.pid = pid

    def poll(self):
        return None

    def wait(self, timeout=None):
        return 0


def test_real_ports_build_only_the_reviewed_commands(monkeypatch, tmp_path: Path):
    processes = iter((FakeProcess(101), FakeProcess(102)))
    runs: list[list[str]] = []

    monkeypatch.setattr(module.subprocess, "Popen", lambda argv, **_kwargs: next(processes))

    def run(argv, **_kwargs):
        runs.append(list(argv))
        return module.subprocess.CompletedProcess(argv, 0, stdout=b"jpeg", stderr=b"")

    monkeypatch.setattr(module.subprocess, "run", run)
    display = module._Display(binary="Xvfb", display=":99")
    browser = module._Browser(
        binary="google-chrome-stable",
        display=":99",
        sandbox_helper=tmp_path / "missing-helper",
        version_run=lambda argv: module.subprocess.CompletedProcess(
            argv, 0, stdout="Google Chrome 152.0.0.0", stderr=""
        ),
    )
    capture = module._Capture(binary="maim", display=":99")
    input_port = module._Input(binary="xdotool", display=":99")
    viewport = Viewport(390, 760)
    display.start(viewport)
    browser.launch("https://login.example.invalid", viewport, tmp_path / "profile")
    capture.capture()
    input_port.tap(1, 2)
    input_port.scroll(3, 4, 2)
    input_port.text("value")
    input_port.key("Tab")
    input_port.back()

    assert display.argv == [["Xvfb", ":99", "-screen", "0", "390x760x24", "-nolisten", "tcp"]]
    chrome = browser.argv[0]
    assert chrome[0] == "google-chrome-stable"
    for flag in module.CHROME_FIXED_FLAGS:
        assert flag in chrome
    assert "--window-size=390,760" in chrome
    assert "--window-position=0,0" in chrome
    assert "--renderer-process-limit=1" in chrome
    assert "--process-per-site" in chrome
    assert "--js-flags=--max-old-space-size=96" in chrome
    expected_user_agent = module.CHROME_USER_AGENT_TEMPLATE.format(major="152")
    assert f"--user-agent={expected_user_agent}" in chrome
    assert "--no-sandbox" in chrome
    assert not any("remote-debugging" in value for value in chrome)
    assert capture.argv == [
        ["maim", "-u", "--format=jpg", "-m", "6", "-q", str(BROWSER_JPEG_QUALITY), "/dev/stdout"]
    ]
    assert input_port.argv == [
        ["xdotool", "mousemove", "1", "2", "click", "1"],
        ["xdotool", "mousemove", "3", "4"],
        ["xdotool", "click", "5"],
        ["xdotool", "click", "5"],
        ["xdotool", "type", "--delay", "0", "--", "value"],
        ["xdotool", "key", "Tab"],
        ["xdotool", "key", "alt+Left"],
    ]
    assert runs[0] == capture.argv[0]


def test_suid_probe_requires_root_owned_executable_setuid_helper(monkeypatch, tmp_path: Path):
    helper = tmp_path / "chrome-sandbox"
    helper.write_text("fixture")
    helper.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_ISUID)
    original_stat = Path.stat

    class RootStat:
        st_uid = 0
        st_mode = stat.S_IRUSR | stat.S_IXUSR | stat.S_ISUID

    monkeypatch.setattr(
        Path,
        "stat",
        lambda self, **kwargs: RootStat() if self == helper else original_stat(self, **kwargs),
    )
    monkeypatch.setattr(module.os, "access", lambda path, mode: path == helper and mode == os.X_OK)
    assert module._Browser._probe_sandbox(helper) == "suid_helper"


def test_browser_termination_escalates_from_term_to_kill_after_five_seconds(
    monkeypatch, tmp_path: Path
):
    class HungProcess:
        pid = 7001

        def wait(self, timeout=None):
            if timeout == module.BROWSER_CLOSE_GRACE_SECONDS:
                raise module.subprocess.TimeoutExpired("chrome", timeout)
            return -9

    browser = module._Browser(
        binary="google-chrome-stable",
        display=":99",
        sandbox_helper=tmp_path / "missing-helper",
        version_run=lambda argv: module.subprocess.CompletedProcess(
            argv, 0, stdout="Google Chrome 152.0.0.0", stderr=""
        ),
    )
    browser._processes[7001] = HungProcess()
    signals = []
    monkeypatch.setattr(module.os, "killpg", lambda pid, sent: signals.append((pid, sent)))

    browser.terminate(7001)

    assert signals == [(7001, module.signal.SIGTERM), (7001, module.signal.SIGKILL)]


def test_origin_never_reflects_url_userinfo(tmp_path: Path):
    bridge, *_ = build_bridge(tmp_path)
    with pytest.raises(BrowserBridgeError) as refused:
        bridge.open(
            "https://private@login.example.invalid/authorize",
            Viewport(390, 760),
            "broker_connect",
        )
    assert refused.value.code == "BROWSER_URL_INVALID"
    assert "Start the ceremony again" in refused.value.reason


def test_core_validation_refusals_each_name_the_available_next_step(tmp_path: Path):
    with pytest.raises(BrowserBridgeError) as viewport:
        Viewport(0, 760)
    assert viewport.value.code == "BROWSER_VIEWPORT_INVALID"
    assert viewport.value.reason.endswith("Send the phone's visible frame size and retry.")

    bridge, *_ = build_bridge(tmp_path)
    with pytest.raises(BrowserBridgeError) as purpose:
        bridge.open("https://login.example.invalid", Viewport(390, 760), "other")
    assert purpose.value.code == "BROWSER_PURPOSE_INVALID"
    assert purpose.value.reason.endswith("Start that ceremony and retry.")

    opened = bridge.open("https://login.example.invalid", Viewport(390, 760), "broker_connect")
    with pytest.raises(BrowserBridgeError) as empty:
        bridge.input(opened["session_id"], [])
    assert empty.value.code == "BROWSER_EVENTS_REQUIRED"
    assert empty.value.reason.endswith("Send the gesture again.")
    with pytest.raises(BrowserBridgeError) as reason:
        bridge.close(opened["session_id"], "")
    assert reason.value.code == "BROWSER_CLOSE_REASON_REQUIRED"
    assert reason.value.reason.endswith("Name why the session ended and close it again.")
    bridge.close(opened["session_id"], "test_finished")

    with pytest.raises(BrowserBridgeError) as missing:
        bridge.input("missing", [{"kind": "back"}])
    assert missing.value.code == "BROWSER_SESSION_NOT_FOUND"
    assert missing.value.reason.endswith("if you still need its browser.")


def test_real_port_refusals_expose_recovery_without_process_output(monkeypatch, tmp_path: Path):
    browser = module._Browser(
        binary="google-chrome-stable",
        display=":99",
        sandbox_helper=tmp_path / "missing-helper",
        version_run=lambda argv: module.subprocess.CompletedProcess(
            argv, 1, stdout="", stderr="private diagnostic"
        ),
    )
    with pytest.raises(BrowserBridgeError) as version:
        browser.launch("https://login.example.invalid", Viewport(390, 760), tmp_path / "profile")
    assert version.value.code == "BROWSER_VERSION_UNAVAILABLE"
    assert version.value.reason.endswith("Reinstall Google Chrome on the box and retry.")
    assert "private diagnostic" not in version.value.reason

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda argv, **_kwargs: module.subprocess.CompletedProcess(
            argv, 1, stdout=b"", stderr=b"private diagnostic"
        ),
    )
    with pytest.raises(BrowserBridgeError) as capture:
        module._Capture(binary="maim", display=":99").capture()
    assert capture.value.code == "BROWSER_CAPTURE_FAILED"
    assert capture.value.reason.endswith("check maim and Xvfb locally, then retry.")
    with pytest.raises(BrowserBridgeError) as input_error:
        module._Input(binary="xdotool", display=":99").tap(1, 2)
    assert input_error.value.code == "BROWSER_INPUT_FAILED"
    assert input_error.value.reason.endswith("retry the gesture or close it.")
