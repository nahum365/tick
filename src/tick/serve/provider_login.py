"""Supervise Codex device login without reading or storing its credential."""

from __future__ import annotations

import queue
import re
import secrets
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import IO, Protocol

from .browser_bridge import BrowserBridge, BrowserBridgeError, Viewport

__all__ = ["ProviderBrowserLoginManager", "ProviderLoginError", "ProviderLoginManager"]

# Codex colours its output even when piped; strip escapes before reading it.
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_URL = re.compile("https:" + r"//[^\s]+")
# The one-time code is the only hyphenated upper-case token Codex prints
# (e.g. ABCD-EFGHJ); matching "code:" text captured the word "authorization".
_CODE = re.compile(r"\b([A-Z0-9]{4,8}-[A-Z0-9]{4,8})\b")


class ProviderLoginError(Exception):
    def __init__(self, code: str, reason: str) -> None:
        self.code = code
        self.reason = reason
        super().__init__(f"{code}: {reason}")


class Process(Protocol):
    stdout: IO[str] | None
    stderr: IO[str] | None

    def poll(self) -> int | None: ...


class HelpResult(Protocol):
    returncode: int
    stdout: str
    stderr: str


@dataclass(slots=True)
class _Login:
    process: Process
    url: str | None
    code: str | None
    expires_at: datetime
    error: str | None


class ProviderLoginManager:
    """Process registry; it retains output state, never provider credentials."""

    def __init__(
        self,
        *,
        binary: str,
        help_run: Callable[[Sequence[str]], HelpResult],
        start_process: Callable[[Sequence[str]], Process],
        now: Callable[[], datetime],
        capture_timeout_seconds: float,
    ) -> None:
        self._binary = binary
        self._help_run = help_run
        self._start_process = start_process
        self._now = now
        self._capture_timeout = capture_timeout_seconds
        self._logins: dict[str, _Login] = {}
        self._lock = threading.Lock()

    @classmethod
    def for_environment(cls, *, now: Callable[[], datetime]) -> ProviderLoginManager:
        binary = shutil.which("codex")
        if binary is None:
            raise ProviderLoginError(
                "CODEX_NOT_INSTALLED",
                "install the `codex` command on the box, then start device login again.",
            )

        def help_run(argv: Sequence[str]) -> HelpResult:
            return subprocess.run(  # noqa: S603
                list(argv), capture_output=True, text=True, check=False, timeout=10
            )

        def start(argv: Sequence[str]) -> Process:
            return subprocess.Popen(  # noqa: S603
                list(argv), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

        return cls(
            binary=binary,
            help_run=help_run,
            start_process=start,
            now=now,
            capture_timeout_seconds=10.0,
        )

    def start(self) -> dict[str, str]:
        help_result = self._help_run([self._binary, "login", "--help"])
        help_text = help_result.stdout + help_result.stderr
        flag = next(
            (value for value in ("--device-auth", "--device-code") if value in help_text), None
        )
        if help_result.returncode != 0 or flag is None:
            exact = (help_result.stderr or help_result.stdout).strip() or "no help output"
            raise ProviderLoginError(
                "CODEX_LOGIN_FLAG_CHANGED",
                f"codex login no longer advertises device authentication: {exact}. "
                "Run `codex login --help` on the box and update Tick before retrying.",
            )
        process = self._start_process([self._binary, "login", flag])
        login_id = secrets.token_hex(10)
        login = _Login(
            process=process,
            url=None,
            code=None,
            expires_at=self._now() + timedelta(minutes=15),
            error=None,
        )
        with self._lock:
            self._logins[login_id] = login
        self._capture(login)
        if login.url is None or login.code is None:
            detail = login.error or "codex did not print a verification URL and user code"
            raise ProviderLoginError(
                "CODEX_LOGIN_OUTPUT_UNREADABLE",
                f"{detail}. Run `codex login --help` on the box and retry there.",
            )
        return {
            "login_id": login_id,
            "url": login.url,
            "code": login.code,
            "expires_at": login.expires_at.isoformat(),
        }

    def _capture(self, login: _Login) -> None:
        text = ""
        lines: queue.Queue[str] = queue.Queue()

        def copy(stream: IO[str]) -> None:
            for line in stream:
                lines.put(line)

        for stream in (login.process.stdout, login.process.stderr):
            if stream is not None:
                reader = threading.Thread(target=copy, args=(stream,), daemon=True)
                reader.start()
        deadline = time.monotonic() + self._capture_timeout
        while time.monotonic() < deadline:
            try:
                text += "\n" + lines.get(timeout=0.05)
            except queue.Empty:
                pass
            clean = _ANSI.sub("", text)
            url = _URL.search(clean)
            code = _CODE.search(clean)
            if url and code:
                login.url = url.group(0).rstrip(".,")
                login.code = code.group(1)
                break
        if login.url is None or login.code is None:
            login.error = " ".join(_ANSI.sub("", text).split())[-500:]

    def status(self, login_id: str) -> dict[str, str]:
        with self._lock:
            login = self._logins.get(login_id)
        if login is None:
            raise ProviderLoginError(
                "CODEX_LOGIN_NOT_FOUND",
                f"login {login_id} is not active. Start device login again.",
            )
        result = login.process.poll()
        state = "pending" if result is None else ("succeeded" if result == 0 else "failed")
        payload = {"login_id": login_id, "state": state}
        if state == "failed":
            payload["reason"] = login.error or "codex login failed; run it locally for details."
        return payload


@dataclass(slots=True)
class _BrowserLogin:
    process: Process
    url: str
    session_id: str
    error: str | None


class ProviderBrowserLoginManager:
    """Run Codex's browser login while a person drives its page on the box."""

    def __init__(
        self,
        *,
        binary: str,
        start_process: Callable[[Sequence[str]], Process],
        bridge: BrowserBridge,
        capture_timeout_seconds: float,
    ) -> None:
        self._binary = binary
        self._start_process = start_process
        self._bridge = bridge
        self._capture_timeout = capture_timeout_seconds
        self._logins: dict[str, _BrowserLogin] = {}
        self._lock = threading.Lock()

    @classmethod
    def for_environment(cls, *, bridge: BrowserBridge) -> ProviderBrowserLoginManager:
        binary = shutil.which("codex")
        if binary is None:
            raise ProviderLoginError(
                "CODEX_NOT_INSTALLED",
                "install the `codex` command on the box, then start browser login again.",
            )

        def start(argv: Sequence[str]) -> Process:
            return subprocess.Popen(  # noqa: S603
                list(argv), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

        return cls(
            binary=binary,
            start_process=start,
            bridge=bridge,
            capture_timeout_seconds=10.0,
        )

    def start(self, viewport: Viewport) -> dict[str, str]:
        process = self._start_process([self._binary, "login"])
        url, captured_error = self._capture_url(process)
        if url is None:
            raise ProviderLoginError(
                "CODEX_LOGIN_OUTPUT_UNREADABLE",
                f"{captured_error or 'codex did not print a browser login URL'}. Run `codex "
                "login` on the box and retry there.",
            )
        login_id = secrets.token_hex(10)
        try:
            opened = self._bridge.open(url, viewport, "provider_login")
        except BrowserBridgeError as exc:
            raise ProviderLoginError(exc.code, exc.reason) from exc
        login = _BrowserLogin(
            process=process,
            url=url,
            session_id=opened["session_id"],
            error=captured_error,
        )
        with self._lock:
            self._logins[login_id] = login
        threading.Thread(
            target=self._watch,
            args=(login,),
            name=f"tick-codex-login-{login_id}",
            daemon=True,
        ).start()
        return {**opened, "login_id": login_id}

    def status(self, login_id: str) -> dict[str, str]:
        with self._lock:
            login = self._logins.get(login_id)
        if login is None:
            raise ProviderLoginError(
                "CODEX_LOGIN_NOT_FOUND",
                f"login {login_id} is not active. Start browser login again.",
            )
        result = login.process.poll()
        state = "pending" if result is None else ("succeeded" if result == 0 else "failed")
        payload = {"login_id": login_id, "state": state}
        if state == "failed":
            payload["reason"] = login.error or "codex login failed; run it locally for details."
        return payload

    def active_authorization_url(self) -> str | None:
        with self._lock:
            for login in reversed(tuple(self._logins.values())):
                if login.process.poll() is None:
                    return login.url
        return None

    def _capture_url(self, process: Process) -> tuple[str | None, str | None]:
        text = ""
        lines: queue.Queue[str] = queue.Queue()

        def copy(stream: IO[str]) -> None:
            for line in stream:
                lines.put(line)

        for stream in (process.stdout, process.stderr):
            if stream is not None:
                threading.Thread(target=copy, args=(stream,), daemon=True).start()
        deadline = time.monotonic() + self._capture_timeout
        while time.monotonic() < deadline:
            try:
                text += "\n" + lines.get(timeout=0.05)
            except queue.Empty:
                pass
            clean = _ANSI.sub("", text)
            found = _URL.search(clean)
            if found:
                return found.group(0).rstrip(".,"), None
        error = " ".join(_ANSI.sub("", text).split())[-500:]
        return None, error or None

    def _watch(self, login: _BrowserLogin) -> None:
        while (result := login.process.poll()) is None:
            time.sleep(0.1)
        reason = "login_succeeded" if result == 0 else "login_failed"
        self._bridge.close(login.session_id, reason)
