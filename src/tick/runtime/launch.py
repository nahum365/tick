"""Boot-scoped launch tickets and process leases for per-launch live authority.

``--live`` is necessary but deliberately insufficient.  A launcher creates a
one-use ticket outside persistent configuration; the child atomically consumes it
before any broker session opens.  The separate lease lets status join process
liveness to historical records instead of presenting a stale mode as current.
"""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, model_validator

from tick.records import ensure_private_dir, write_private_file

from .modes import ApprovalMode, Mode

__all__ = [
    "LaunchError",
    "LaunchTicket",
    "RunLease",
    "consume_launch_ticket",
    "create_launch_ticket",
    "launch_lock",
    "load_run_lease",
    "run_state",
    "save_run_lease",
    "ticket_directory",
]

FILE_MODE = 0o600


class LaunchError(Exception):
    """A launch refusal with a stable code and actionable sentence."""

    def __init__(self, code: str, reason: str, *, status: int = 409) -> None:
        self.code = code
        self.reason = reason
        self.status = status
        super().__init__(f"{code}: {reason}")


class LaunchTicket(BaseModel):
    """One-use authority for one live child on this boot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_id: str
    run_id: str
    boot_id: str
    approval_mode: ApprovalMode
    standing_ok: bool
    created_at: AwareDatetime

    @model_validator(mode="after")
    def _standing(self) -> LaunchTicket:
        if self.approval_mode is ApprovalMode.STANDING and not self.standing_ok:
            raise ValueError("a standing live launch needs its explicit acknowledgement")
        return self


class RunLease(BaseModel):
    """The process observation status joins with the ledger."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_id: str
    run_id: str
    boot_id: str
    pid: int
    mode: Mode
    approval: ApprovalMode
    launch_source: Literal["cli", "api", "supervisor"]
    started_at: AwareDatetime
    previous_run_id: str | None
    previous_run_mode: Mode | None
    previous_run_boot_id: str | None

    @model_validator(mode="after")
    def _pid(self) -> RunLease:
        if self.pid <= 0:
            raise ValueError("a run lease pid must be positive")
        return self


def ticket_directory(home: str | os.PathLike[str], env: Mapping[str, str]) -> tuple[Path, bool]:
    """Return a boot-volatile ticket directory, plus whether fallback was necessary."""
    runtime = env.get("XDG_RUNTIME_DIR")
    if runtime and Path(runtime).is_dir():
        return Path(runtime) / "tick", False
    managed = Path("/run/tick")
    if managed.is_dir() and os.access(managed, os.W_OK):
        return managed, False
    return Path(home) / "run", True


def create_launch_ticket(
    home: str | os.PathLike[str],
    *,
    agent_id: str,
    run_id: str,
    approval_mode: ApprovalMode,
    standing_ok: bool,
    created_at: datetime,
    env: Mapping[str, str],
    current_boot_id: str,
) -> tuple[Path, bool]:
    """Create a private ticket; callers surface the persistent fallback caveat."""
    ticket = LaunchTicket(
        agent_id=agent_id,
        run_id=run_id,
        boot_id=current_boot_id,
        approval_mode=approval_mode,
        standing_ok=standing_ok,
        created_at=created_at,
    )
    directory, fallback = ticket_directory(home, env)
    ensure_private_dir(directory)
    path = directory / f"{run_id}.json"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, FILE_MODE)
    try:
        os.fchmod(descriptor, FILE_MODE)
        os.write(
            descriptor,
            (json.dumps(ticket.model_dump(mode="json"), sort_keys=True) + "\n").encode(),
        )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)
    return path, fallback


def consume_launch_ticket(
    path: str | os.PathLike[str],
    *,
    agent_id: str,
    run_id: str,
    approval_mode: ApprovalMode,
    standing_ok: bool,
    current_boot_id: str,
) -> LaunchTicket:
    """Atomically consume and validate a live ticket before broker construction."""
    source = Path(path)
    consumed = source.with_name(source.name + ".consumed")
    try:
        os.rename(source, consumed)
    except OSError as exc:
        raise LaunchError(
            "live_ticket_missing",
            "this live command has no unused launch ticket for the current boot. Nothing "
            "was connected or placed; launch live again from the box or enabled remote.",
            status=403,
        ) from exc
    try:
        try:
            ticket = LaunchTicket.model_validate_json(consumed.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise LaunchError(
                "live_ticket_invalid",
                f"the one-use live ticket is invalid ({exc}). Nothing was connected or "
                "placed; launch live again.",
                status=403,
            ) from exc
        expected = (
            agent_id,
            run_id,
            current_boot_id,
            ApprovalMode(approval_mode),
            standing_ok,
        )
        actual = (
            ticket.agent_id,
            ticket.run_id,
            ticket.boot_id,
            ticket.approval_mode,
            ticket.standing_ok,
        )
        if actual != expected:
            raise LaunchError(
                "live_ticket_mismatch",
                "the live ticket does not bind this agent, run, boot, approval mode, and "
                "standing acknowledgement. Nothing was connected or placed; launch live "
                "again.",
                status=403,
            )
        return ticket
    finally:
        try:
            consumed.unlink()
        except FileNotFoundError:
            pass


def run_lease_path(home: str | os.PathLike[str], agent_id: str) -> Path:
    return Path(home) / "agents" / agent_id / "run.json"


def save_run_lease(home: str | os.PathLike[str], lease: RunLease) -> Path:
    """Publish the current process observation without storing a live preference."""
    path = run_lease_path(home, lease.agent_id)
    write_private_file(
        path,
        json.dumps(lease.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
    )
    return path


def load_run_lease(home: str | os.PathLike[str], agent_id: str) -> RunLease | None:
    """Read a lease; malformed state yields unknown status instead of invented liveness."""
    try:
        return RunLease.model_validate_json(
            run_lease_path(home, agent_id).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None


def run_state(
    lease: RunLease | None,
    *,
    pid_alive: Callable[[int], bool],
    current_boot_id: str,
) -> Literal["running", "stopped", "unknown"]:
    """Classify current liveness; an old boot can never still be running."""
    if lease is None:
        return "unknown"
    if lease.boot_id != current_boot_id:
        return "stopped"
    try:
        return "running" if pid_alive(lease.pid) else "stopped"
    except OSError:
        return "unknown"


@contextmanager
def launch_lock(home: str | os.PathLike[str], agent_id: str) -> Iterator[None]:
    """Serialize launch checks and process creation for one agent."""
    directory = Path(home) / "agents" / agent_id
    ensure_private_dir(directory)
    descriptor = os.open(directory / "launch.lock", os.O_CREAT | os.O_RDWR, FILE_MODE)
    try:
        os.fchmod(descriptor, FILE_MODE)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
