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

__all__ = ["ProviderLoginError", "ProviderLoginManager"]

_URL = re.compile("https:" + r"//[^\s]+")
_CODE = re.compile(
    r"(?:code\s*[: ]|enter\s+(?:this\s+)?code\s*[: ])\s*([A-Z0-9-]{4,})",
    re.IGNORECASE,
)


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
            url = _URL.search(text)
            code = _CODE.search(text)
            if url and code:
                login.url = url.group(0).rstrip(".,")
                login.code = code.group(1)
                break
        if login.url is None or login.code is None:
            login.error = " ".join(text.split())[-500:]

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
