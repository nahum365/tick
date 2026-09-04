"""A small pixel-and-input browser bridge for ceremonies on the user's box.

The bridge deliberately has no browser automation surface.  It starts a stock
browser on a RAM-backed X display, captures JPEG pixels, and forwards a closed
set of human input events.  Frames, URLs, and typed text are never written;
the local ledger retains only session purpose, origin host, duration, and RSS.
"""

from __future__ import annotations

import os
import queue
import re
import secrets
import shutil
import signal
import stat
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Generator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

from tick.engine import Unavailable
from tick.records import DataSource, Ledger, RecordKind, ensure_private_dir

__all__ = [
    "BROWSER_CAPTURE_GRACE_SECONDS",
    "BROWSER_FPS",
    "BROWSER_JPEG_QUALITY",
    "BROWSER_LIFETIME_SECONDS",
    "BROWSER_MAX_TEXT_CHARS",
    "BROWSER_SESSION_LIMIT",
    "BrowserBridge",
    "BrowserBridgeError",
    "BrowserPort",
    "CapturePort",
    "DisplayPort",
    "InputPort",
    "Viewport",
]

BROWSER_DISPLAY = ":99"
BROWSER_FPS = 1
BROWSER_FRAME_QUEUE_SIZE = 3
# maim's quality scale is 1 through 10 (its -m flag); -q means quiet, not quality.
BROWSER_JPEG_QUALITY = 6
BROWSER_LIFETIME_SECONDS = 10 * 60
# Xvfb and Chrome start asynchronously; captures may miss until the display is
# up and the first window is mapped. Misses inside this window are retried;
# a miss that persists past it closes the session as capture_failed.
BROWSER_CAPTURE_GRACE_SECONDS = 5
BROWSER_MAX_TEXT_CHARS = 512
BROWSER_SESSION_LIMIT = 1
BROWSER_CLOSE_GRACE_SECONDS = 5

# A private session leaves no browser profile behind after close.
CHROME_INCOGNITO_FLAG = "--incognito"
# First-run UI would cover the ceremony the user is trying to complete.
CHROME_NO_FIRST_RUN_FLAG = "--no-first-run"
# Default-browser prompts likewise obscure the streamed page.
CHROME_NO_DEFAULT_BROWSER_CHECK_FLAG = "--no-default-browser-check"
# Extensions add unbounded code and memory to this single-purpose session.
CHROME_DISABLE_EXTENSIONS_FLAG = "--disable-extensions"
# Background fetches are unrelated to the page the person opened.
CHROME_DISABLE_BACKGROUND_NETWORKING_FLAG = "--disable-background-networking"
# Sync would retain browser state beyond this temporary profile.
CHROME_DISABLE_SYNC_FLAG = "--disable-sync"
# Xvfb has no useful GPU and the software path has a smaller footprint.
CHROME_DISABLE_GPU_FLAG = "--disable-gpu"
# The 64 MiB default shared-memory mount is too small on the target droplet.
CHROME_DISABLE_DEV_SHM_FLAG = "--disable-dev-shm-usage"
# One renderer bounds the browser's process and memory fan-out.
CHROME_RENDERER_LIMIT_FLAG = "--renderer-process-limit=1"
# A site shares its renderer while unrelated origins remain isolated.
CHROME_PROCESS_PER_SITE_FLAG = "--process-per-site"
# The JavaScript heap is explicitly capped for the 512 MiB host.
CHROME_JS_HEAP_FLAG = "--js-flags=--max-old-space-size=96"
# Only this flag is conditional, when the installed SUID sandbox helper fails its probe.
CHROME_NO_SANDBOX_FLAG = "--no-sandbox"
CHROME_FIXED_FLAGS = (
    CHROME_INCOGNITO_FLAG,
    CHROME_NO_FIRST_RUN_FLAG,
    CHROME_NO_DEFAULT_BROWSER_CHECK_FLAG,
    CHROME_DISABLE_EXTENSIONS_FLAG,
    CHROME_DISABLE_BACKGROUND_NETWORKING_FLAG,
    CHROME_DISABLE_SYNC_FLAG,
    CHROME_DISABLE_GPU_FLAG,
    CHROME_DISABLE_DEV_SHM_FLAG,
    CHROME_RENDERER_LIMIT_FLAG,
    CHROME_PROCESS_PER_SITE_FLAG,
    CHROME_JS_HEAP_FLAG,
)
CHROME_USER_AGENT_TEMPLATE = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/{major}.0.0.0 Safari/537.36"
)
CHROME_SANDBOX_HELPER = Path("/opt/google/chrome/chrome-sandbox")


class BrowserBridgeError(Exception):
    """A stable refusal code plus a sentence describing the available next step."""

    def __init__(self, code: str, reason: str) -> None:
        self.code = code
        self.reason = reason
        super().__init__(f"{code}: {reason}")


@dataclass(frozen=True, slots=True)
class Viewport:
    """The exact frame dimensions supplied by the phone for this session."""

    width: int
    height: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.width, int)
            or isinstance(self.width, bool)
            or not isinstance(self.height, int)
            or isinstance(self.height, bool)
            or not 1 <= self.width <= 4096
            or not 1 <= self.height <= 4096
        ):
            raise BrowserBridgeError(
                "BROWSER_VIEWPORT_INVALID",
                "viewport width and height must each be integers from 1 through 4096. "
                "Send the phone's visible frame size and retry.",
            )


class DisplayPort(Protocol):
    def start(self, viewport: Viewport) -> int: ...

    def stop(self, pid: int) -> None: ...


class BrowserPort(Protocol):
    sandbox_mode: str

    def launch(self, url: str, viewport: Viewport, profile: Path) -> int: ...

    def rss_mb(self, pid: int) -> Decimal | Unavailable: ...

    def terminate(self, pid: int) -> None: ...


class CapturePort(Protocol):
    def capture(self) -> bytes: ...


class InputPort(Protocol):
    def tap(self, x: int, y: int) -> None: ...

    def scroll(self, x: int, y: int, dy: int) -> None: ...

    def text(self, value: str) -> None: ...

    def key(self, value: str) -> None: ...

    def back(self) -> None: ...


class _Display:
    def __init__(self, *, binary: str, display: str) -> None:
        self._binary = binary
        self._display = display
        self.argv: list[list[str]] = []
        self._processes: dict[int, subprocess.Popen[bytes]] = {}

    def start(self, viewport: Viewport) -> int:
        argv = [
            self._binary,
            self._display,
            "-screen",
            "0",
            f"{viewport.width}x{viewport.height}x24",
            "-nolisten",
            "tcp",
        ]
        self.argv.append(argv)
        process = subprocess.Popen(argv, start_new_session=True)  # noqa: S603
        self._processes[process.pid] = process
        return process.pid

    def stop(self, pid: int) -> None:
        process = self._processes.pop(pid, None)
        if process is None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=BROWSER_CLOSE_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


class _Browser:
    def __init__(
        self,
        *,
        binary: str,
        display: str,
        sandbox_helper: Path,
        version_run: Callable[[Sequence[str]], subprocess.CompletedProcess[str]],
    ) -> None:
        self._binary = binary
        self._display = display
        self._processes: dict[int, subprocess.Popen[bytes]] = {}
        self.argv: list[list[str]] = []
        self.sandbox_mode = self._probe_sandbox(sandbox_helper)
        self._version_run = version_run
        self._user_agent: str | None = None

    def _read_user_agent(self) -> str:
        if self._user_agent is not None:
            return self._user_agent
        result = self._version_run([self._binary, "--version"])
        match = re.search(r"\b(\d+)\.", result.stdout)
        if result.returncode != 0 or match is None:
            raise BrowserBridgeError(
                "BROWSER_VERSION_UNAVAILABLE",
                "the installed Chrome version could not be read. Reinstall Google Chrome on "
                "the box and retry.",
            )
        self._user_agent = CHROME_USER_AGENT_TEMPLATE.format(major=match.group(1))
        return self._user_agent

    @staticmethod
    def _probe_sandbox(helper: Path) -> str:
        try:
            mode = helper.stat()
        except OSError:
            return "no_sandbox"
        if mode.st_uid == 0 and stat.S_ISUID & mode.st_mode and os.access(helper, os.X_OK):
            return "suid_helper"
        return "no_sandbox"

    def launch(self, url: str, viewport: Viewport, profile: Path) -> int:
        argv = [
            self._binary,
            *CHROME_FIXED_FLAGS,
            f"--window-size={viewport.width},{viewport.height}",
            "--window-position=0,0",
            f"--user-agent={self._read_user_agent()}",
            f"--user-data-dir={profile}",
        ]
        if self.sandbox_mode == "no_sandbox":
            argv.append(CHROME_NO_SANDBOX_FLAG)
        argv.append(url)
        self.argv.append(argv)
        environment = dict(os.environ)
        environment["DISPLAY"] = self._display
        process = subprocess.Popen(argv, env=environment, start_new_session=True)  # noqa: S603
        self._processes[process.pid] = process
        return process.pid

    def rss_mb(self, pid: int) -> Decimal | Unavailable:
        try:
            status_text = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
        except OSError as exc:
            return Unavailable(what="Chrome RSS", reason=f"/proc status could not be read: {exc}")
        match = re.search(r"^VmRSS:\s+(\d+)\s+kB$", status_text, re.MULTILINE)
        if match is None:
            return Unavailable(
                what="Chrome RSS", reason="VmRSS was absent from the Chrome process status"
            )
        return Decimal(match.group(1)) / Decimal(1024)

    def terminate(self, pid: int) -> None:
        process = self._processes.pop(pid, None)
        if process is None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=BROWSER_CLOSE_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


class _Capture:
    def __init__(self, *, binary: str, display: str) -> None:
        self._binary = binary
        self._display = display
        self.argv: list[list[str]] = []

    def capture(self) -> bytes:
        # No output path: maim writes to stdout. A stray positional argument
        # (as a misread "-q <n>" once produced) silently becomes a filename.
        argv = [
            self._binary,
            "-u",
            "--format=jpg",
            "-m",
            str(BROWSER_JPEG_QUALITY),
        ]
        self.argv.append(argv)
        environment = dict(os.environ)
        environment["DISPLAY"] = self._display
        result = subprocess.run(argv, env=environment, capture_output=True, check=False)  # noqa: S603
        if result.returncode != 0 or not result.stdout:
            raise BrowserBridgeError(
                "BROWSER_CAPTURE_FAILED",
                "the box could not capture the browser frame. Close this session, check maim "
                "and Xvfb locally, then retry.",
            )
        return result.stdout


class _Input:
    def __init__(self, *, binary: str, display: str) -> None:
        self._binary = binary
        self._display = display
        self.argv: list[list[str]] = []

    def _run(self, argv: list[str]) -> None:
        self.argv.append(argv)
        environment = dict(os.environ)
        environment["DISPLAY"] = self._display
        result = subprocess.run(argv, env=environment, capture_output=True, check=False)  # noqa: S603
        if result.returncode != 0:
            raise BrowserBridgeError(
                "BROWSER_INPUT_FAILED",
                "the box could not send that input to the browser. The session is still "
                "open; retry the gesture or close it.",
            )

    def tap(self, x: int, y: int) -> None:
        self._run([self._binary, "mousemove", str(x), str(y), "click", "1"])

    def scroll(self, x: int, y: int, dy: int) -> None:
        self._run([self._binary, "mousemove", str(x), str(y)])
        button = "5" if dy > 0 else "4"
        for _ in range(abs(dy)):
            self._run([self._binary, "click", button])

    def text(self, value: str) -> None:
        self._run([self._binary, "type", "--delay", "0", "--", value])

    def key(self, value: str) -> None:
        self._run([self._binary, "key", value])

    def back(self) -> None:
        self._run([self._binary, "key", "alt+Left"])


@dataclass(slots=True)
class _Session:
    session_id: str
    purpose: str
    host: str
    origin: str
    viewport: Viewport
    profile: Path
    display_pid: int
    browser_pid: int
    started_monotonic: Decimal
    frames: queue.Queue[tuple[int, bytes]] = field(
        default_factory=lambda: queue.Queue(maxsize=BROWSER_FRAME_QUEUE_SIZE)
    )
    closed: threading.Event = field(default_factory=threading.Event)
    close_reason: str | None = None
    producer: threading.Thread | None = None


class BrowserBridge:
    """Own the single ephemeral browser session allowed on one box."""

    def __init__(
        self,
        *,
        home: Path,
        display: DisplayPort,
        browser: BrowserPort,
        capture: CapturePort,
        input_port: InputPort,
        now: Callable[[], datetime],
        monotonic: Callable[[], Decimal],
        sleeper: Callable[[Decimal], None],
    ) -> None:
        self._home = home
        self._display = display
        self._browser = browser
        self._capture = capture
        self._input = input_port
        self._now = now
        self._monotonic = monotonic
        self._sleeper = sleeper
        self._lock = threading.RLock()
        self._active: _Session | None = None
        self._closed_reasons: dict[str, str] = {}

    @classmethod
    def for_environment(cls, *, home: Path) -> BrowserBridge:
        """Build the four subprocess ports; callers never construct argv themselves."""

        def version_run(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
            return subprocess.run(  # noqa: S603
                list(argv), capture_output=True, text=True, check=False, timeout=10
            )

        display = _Display(binary="Xvfb", display=BROWSER_DISPLAY)
        browser = _Browser(
            binary="google-chrome-stable",
            display=BROWSER_DISPLAY,
            sandbox_helper=CHROME_SANDBOX_HELPER,
            version_run=version_run,
        )
        capture = _Capture(binary="maim", display=BROWSER_DISPLAY)
        input_port = _Input(binary="xdotool", display=BROWSER_DISPLAY)
        return cls(
            home=home,
            display=display,
            browser=browser,
            capture=capture,
            input_port=input_port,
            now=lambda: datetime.now(UTC),
            monotonic=lambda: Decimal(time.monotonic_ns()) / Decimal(1_000_000_000),
            sleeper=lambda seconds: time.sleep(float(seconds)),
        )

    def open(self, url: str, viewport: Viewport, purpose: str) -> dict[str, str]:
        parsed = urlsplit(url)
        try:
            port = parsed.port
        except ValueError as exc:
            raise BrowserBridgeError(
                "BROWSER_URL_INVALID",
                "the ceremony URL has an invalid port. Start the ceremony again and use "
                "the URL the box returns.",
            ) from exc
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise BrowserBridgeError(
                "BROWSER_URL_INVALID",
                "the ceremony URL must be an http or https URL with a host. Start the "
                "ceremony again and use the URL the box returns.",
            )
        if purpose not in {"broker_connect", "provider_login"}:
            raise BrowserBridgeError(
                "BROWSER_PURPOSE_INVALID",
                "purpose must be broker_connect or provider_login. Start that ceremony and retry.",
            )
        with self._lock:
            if self._active is not None and not self._active.closed.is_set():
                raise BrowserBridgeError(
                    "BROWSER_SESSION_BUSY",
                    "this box already has a browser session open. Close it before starting "
                    "another.",
                )
            browser_root = ensure_private_dir(self._home / "browser")
            profile = Path(tempfile.mkdtemp(prefix="profile-", dir=browser_root))
            profile.chmod(0o700)
            display_pid: int | None = None
            try:
                display_pid = self._display.start(viewport)
                browser_pid = self._browser.launch(url, viewport, profile)
            except BaseException:
                if display_pid is not None:
                    self._display.stop(display_pid)
                shutil.rmtree(profile, ignore_errors=True)
                raise
            session_id = secrets.token_hex(16)
            host = parsed.hostname
            rendered_host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
            origin = f"{parsed.scheme}://{rendered_host}"
            if port is not None:
                origin += f":{port}"
            session = _Session(
                session_id=session_id,
                purpose=purpose,
                host=host,
                origin=origin,
                viewport=viewport,
                profile=profile,
                display_pid=display_pid,
                browser_pid=browser_pid,
                started_monotonic=self._monotonic(),
            )
            self._active = session
            self._ledger().append(
                RecordKind.NOTE,
                {
                    "event": "browser_session_opened",
                    "purpose": purpose,
                    "host": host,
                    "rss_mb": self._browser.rss_mb(browser_pid),
                    "sandbox": self._browser.sandbox_mode,
                },
                source=DataSource.RUNTIME,
            )
            return {"session_id": session_id, "origin": origin}

    def frames(self, session_id: str) -> Generator[tuple[int, bytes, str], None, None]:
        with self._lock:
            if session_id in self._closed_reasons:
                return
        session = self._session(session_id, allow_closed=True)
        with self._lock:
            if session.producer is None and not session.closed.is_set():
                session.producer = threading.Thread(
                    target=self._capture_loop,
                    args=(session,),
                    name=f"tick-browser-{session.session_id}",
                    daemon=True,
                )
                session.producer.start()
        while not session.closed.is_set() or not session.frames.empty():
            try:
                t_ms, jpeg = session.frames.get(timeout=1 / BROWSER_FPS)
            except queue.Empty:
                continue
            yield t_ms, jpeg, session.origin

    def input(self, session_id: str, events: Sequence[Mapping[str, Any]]) -> None:
        session = self._session(session_id)
        if not events:
            raise BrowserBridgeError(
                "BROWSER_EVENTS_REQUIRED",
                "events must contain at least one browser input. Send the gesture again.",
            )
        for event in events:
            if not isinstance(event, Mapping):
                self._event_invalid("each event must be an object")
            kind = event.get("kind")
            if kind == "tap" and set(event) == {"kind", "x", "y"}:
                x, y = self._coordinates(session, event)
                self._input.tap(x, y)
            elif kind == "scroll" and set(event) == {"kind", "x", "y", "dy"}:
                x, y = self._coordinates(session, event)
                dy = event.get("dy")
                if not isinstance(dy, int) or isinstance(dy, bool) or dy == 0:
                    self._event_invalid("scroll dy must be a non-zero integer")
                self._input.scroll(x, y, dy)
            elif kind == "text" and set(event) == {"kind", "s"}:
                value = event.get("s")
                if not isinstance(value, str) or not value or len(value) > BROWSER_MAX_TEXT_CHARS:
                    self._event_invalid(
                        f"text s must contain 1 through {BROWSER_MAX_TEXT_CHARS} characters"
                    )
                self._input.text(value)
            elif kind == "key" and set(event) == {"kind", "key"}:
                value = event.get("key")
                if value not in {"Return", "BackSpace", "Tab", "Escape"}:
                    self._event_invalid("key must be Return, BackSpace, Tab, or Escape")
                self._input.key(str(value))
            elif kind == "back" and set(event) == {"kind"}:
                self._input.back()
            else:
                self._event_invalid("the event shape is not in the browser input contract")

    def close(self, session_id: str, reason: str) -> None:
        if not reason.strip():
            raise BrowserBridgeError(
                "BROWSER_CLOSE_REASON_REQUIRED",
                "a close reason is required. Name why the session ended and close it again.",
            )
        with self._lock:
            session = self._active
            if session is None or session.session_id != session_id:
                if session_id in self._closed_reasons:
                    return
                raise BrowserBridgeError(
                    "BROWSER_SESSION_NOT_FOUND",
                    "that browser session is not active. Start the ceremony again if you still "
                    "need its browser.",
                )
            if session.closed.is_set():
                return
            session.close_reason = reason
            session.closed.set()
            rss = self._browser.rss_mb(session.browser_pid)
            self._browser.terminate(session.browser_pid)
            self._display.stop(session.display_pid)
            shutil.rmtree(session.profile, ignore_errors=True)
            elapsed = self._monotonic() - session.started_monotonic
            if elapsed < 0:
                elapsed = Decimal(0)
            self._ledger().append(
                RecordKind.NOTE,
                {
                    "event": "browser_session_closed",
                    "reason": reason,
                    "rss_mb": rss,
                    "seconds": elapsed,
                },
                source=DataSource.RUNTIME,
            )
            self._closed_reasons[session_id] = reason
            self._active = None

    def close_active(self, *, purpose: str, reason: str) -> None:
        with self._lock:
            session = self._active
            if session is None or session.purpose != purpose:
                return
            session_id = session.session_id
        self.close(session_id, reason)

    def close_reason(self, session_id: str) -> str:
        with self._lock:
            if self._active is not None and self._active.session_id == session_id:
                return self._active.close_reason or "closed"
            return self._closed_reasons.get(session_id, "closed")

    def knows(self, session_id: str) -> bool:
        """Answer whether a frames route may safely start or finish for this id."""
        with self._lock:
            return (
                self._active is not None and self._active.session_id == session_id
            ) or session_id in self._closed_reasons

    def shutdown(self) -> None:
        with self._lock:
            session_id = self._active.session_id if self._active is not None else None
        if session_id is not None:
            self.close(session_id, "serve_stopped")

    def _capture_loop(self, session: _Session) -> None:
        interval = Decimal(1) / Decimal(BROWSER_FPS)
        first_miss: Decimal | None = None
        while not session.closed.is_set():
            now = self._monotonic()
            age = now - session.started_monotonic
            if age >= BROWSER_LIFETIME_SECONDS:
                self.close(session.session_id, "expired")
                return
            try:
                frame = self._capture.capture()
            except Exception:  # noqa: BLE001 - a capture failure closes without leaking details
                if first_miss is None:
                    first_miss = now
                if now - first_miss >= BROWSER_CAPTURE_GRACE_SECONDS:
                    self.close(session.session_id, "capture_failed")
                    return
                self._sleeper(interval)
                continue
            first_miss = None
            item = (int(self._monotonic() * Decimal(1000)), frame)
            try:
                session.frames.put_nowait(item)
            except queue.Full:
                try:
                    session.frames.get_nowait()
                except queue.Empty:
                    pass
                session.frames.put_nowait(item)
            self._sleeper(interval)

    def _session(self, session_id: str, *, allow_closed: bool = False) -> _Session:
        with self._lock:
            session = self._active
            if session is not None and session.session_id == session_id:
                if allow_closed or not session.closed.is_set():
                    return session
            raise BrowserBridgeError(
                "BROWSER_SESSION_NOT_FOUND",
                "that browser session is not active. Start the ceremony again if you still need "
                "its browser.",
            )

    @staticmethod
    def _coordinates(session: _Session, event: Mapping[str, Any]) -> tuple[int, int]:
        x, y = event.get("x"), event.get("y")
        if (
            not isinstance(x, int)
            or isinstance(x, bool)
            or not isinstance(y, int)
            or isinstance(y, bool)
            or not 0 <= x < session.viewport.width
            or not 0 <= y < session.viewport.height
        ):
            BrowserBridge._event_invalid(
                "x and y must be integer pixels inside the current browser frame"
            )
        return x, y

    @staticmethod
    def _event_invalid(detail: str) -> None:
        raise BrowserBridgeError(
            "BROWSER_EVENT_INVALID",
            f"{detail}. Use the latest frame coordinates and send the gesture again.",
        )

    def _ledger(self) -> Ledger:
        return Ledger(self._home / "browser" / "records.jsonl", clock=self._now)
