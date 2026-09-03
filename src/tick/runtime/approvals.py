"""Durable, run-bound approval requests with one terminal resolution.

The queue is a directory per request, not a mutable JSON document.  ``request.json``
is immutable evidence of what was proposed and ``resolution.json`` is created once
under an advisory per-request lock.  That shape makes the safety rule inspectable:
there is no update operation that can turn a decline, expiry, interruption, or STOP
into an approval later.
"""

from __future__ import annotations

import fcntl
import json
import os
import secrets
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import AwareDatetime, BaseModel, ConfigDict, model_validator

from tick.records import ensure_private_dir, normalize_payload, write_private_file
from tick.spec import canonical_encode, sha256_hex

__all__ = [
    "ApprovalError",
    "ApprovalOutcome",
    "ApprovalQueue",
    "ApprovalRequest",
    "ApprovalResolution",
    "ApprovalWindow",
    "boot_id",
    "parse_approval_window",
]

FILE_MODE = 0o600
REQUEST_FILE = "request.json"
RESOLUTION_FILE = "resolution.json"
LOCK_FILE = "resolution.lock"
_THREAD_LOCKS_GUARD = threading.Lock()
_THREAD_LOCKS: dict[str, threading.Lock] = {}


class ApprovalOutcome(StrEnum):
    """Terminal queue states.  None of them can transition again."""

    APPROVED = "approved"
    DECLINED = "declined"
    EXPIRED = "expired"
    INTERRUPTED = "approval_interrupted"
    ABORTED_BY_STOP = "aborted_by_stop"


class ApprovalError(Exception):
    """A stable refusal from the queue, including its HTTP-equivalent status."""

    def __init__(
        self,
        code: str,
        reason: str,
        *,
        status: int,
        resolution: ApprovalResolution | None = None,
    ) -> None:
        self.code = code
        self.reason = reason
        self.status = status
        self.resolution = resolution
        super().__init__(f"{code}: {reason}")


class ApprovalWindow(BaseModel):
    """A required positive duration; there is deliberately no module default."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seconds: int

    @model_validator(mode="after")
    def _positive(self) -> ApprovalWindow:
        if self.seconds <= 0:
            raise ValueError("an approval window must be greater than zero seconds")
        return self


def parse_approval_window(value: str) -> ApprovalWindow:
    """Parse an explicit ``Ns``, ``Nm``, or ``Nh`` approval window."""
    if len(value) < 2 or value[-1] not in "smh" or not value[:-1].isdigit():
        raise ValueError(
            f"{value!r} is not an approval window. Use a positive duration such as 300s."
        )
    multiplier = {"s": 1, "m": 60, "h": 3600}[value[-1]]
    return ApprovalWindow(seconds=int(value[:-1]) * multiplier)


def boot_id() -> str:
    """Return this boot's kernel id, or a process-stable conservative fallback."""
    override = os.environ.get("TICK_BOOT_ID")
    if override and override.strip():
        return override.strip()
    path = Path("/proc/sys/kernel/random/boot_id")
    try:
        value = path.read_text(encoding="ascii").strip()
    except OSError:
        value = _fallback_boot_id()
    return value


def _fallback_boot_id() -> str:
    """Approximate the kernel boot epoch where Linux's boot id is unavailable.

    Wall time minus monotonic uptime is stable across processes on one boot.  A
    ten-second bucket avoids cross-process measurement jitter while still making
    a reboot produce a different id.  The ticket directory caveat remains visible
    when this fallback accompanies persistent storage.
    """
    boot_epoch = time.time() - time.monotonic()
    return f"boot-epoch-{round(boot_epoch / 10) * 10}"


class ApprovalRequest(BaseModel):
    """Immutable evidence shown locally and, only behind the explicit gate, remotely."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    approval_id: str
    agent_id: str
    run_id: str
    tick_id: str
    intent_hash: str
    evidence_hash: str
    created_at: AwareDatetime
    deadline_wall: AwareDatetime
    boot_id: str
    deadline_monotonic: Decimal
    symbol: str
    side: str
    qty: int
    est_price: Decimal | None
    price_source: str
    data_class: str
    est_notional: Decimal | None
    cage_checks: tuple[str, ...]
    proposed_by: str
    intent: dict[str, Any]
    evidence: dict[str, Any]

    @model_validator(mode="after")
    def _valid(self) -> ApprovalRequest:
        if self.deadline_wall <= self.created_at:
            raise ValueError("an approval deadline must be after its creation time")
        if self.qty <= 0:
            raise ValueError("approval quantity must be a known positive whole-share value")
        if self.est_price is not None and self.est_price <= 0:
            raise ValueError("an estimated price, where known, must be positive")
        if self.est_notional is not None and self.est_notional <= 0:
            raise ValueError("an estimated notional, where known, must be positive")
        return self


class ApprovalResolution(BaseModel):
    """The one durable terminal result for an approval request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    approval_id: str
    outcome: ApprovalOutcome
    decided_via: str | None
    decided_at: AwareDatetime
    reason: str


def _dump(model: BaseModel) -> str:
    return (
        json.dumps(
            normalize_payload(model.model_dump(mode="python")),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    )


class ApprovalQueue:
    """A fail-closed approval queue rooted in one agent directory."""

    def __init__(
        self,
        home: str | os.PathLike[str],
        agent_id: str,
        *,
        wall_clock: Callable[[], datetime],
        monotonic_clock: Callable[[], float],
        current_boot_id: Callable[[], str],
    ) -> None:
        self.home = Path(home)
        self.agent_id = agent_id
        self.directory = self.home / "agents" / agent_id / "approvals"
        self._wall_clock = wall_clock
        self._monotonic_clock = monotonic_clock
        self._current_boot_id = current_boot_id

    @classmethod
    def system(cls, home: str | os.PathLike[str], agent_id: str) -> ApprovalQueue:
        """Build the production queue with both independent box clocks."""
        return cls(
            home,
            agent_id,
            wall_clock=lambda: datetime.now(UTC),
            monotonic_clock=time.monotonic,
            current_boot_id=boot_id,
        )

    def create(
        self,
        *,
        run_id: str,
        tick_id: str,
        window: ApprovalWindow,
        symbol: str,
        side: str,
        qty: int,
        est_price: Decimal | None,
        price_source: str,
        data_class: str,
        est_notional: Decimal | None,
        cage_checks: tuple[str, ...],
        proposed_by: str,
        intent: Mapping[str, Any],
        evidence: Mapping[str, Any],
    ) -> ApprovalRequest:
        """Commit an immutable request bound to this run, tick, intent, and evidence."""
        now = self._aware_now()
        monotonic_now = Decimal(str(self._monotonic_clock()))
        normalized_intent = normalize_payload(dict(intent), where="approval intent")
        normalized_evidence = normalize_payload(dict(evidence), where="approval evidence")
        request = ApprovalRequest(
            approval_id=secrets.token_hex(16),
            agent_id=self.agent_id,
            run_id=run_id,
            tick_id=tick_id,
            intent_hash=sha256_hex(canonical_encode(normalized_intent)),
            evidence_hash=sha256_hex(canonical_encode(normalized_evidence)),
            created_at=now,
            deadline_wall=now + timedelta(seconds=window.seconds),
            boot_id=self._current_boot_id(),
            deadline_monotonic=monotonic_now + Decimal(window.seconds),
            symbol=symbol,
            side=side,
            qty=qty,
            est_price=est_price,
            price_source=price_source,
            data_class=data_class,
            est_notional=est_notional,
            cage_checks=cage_checks,
            proposed_by=proposed_by,
            intent=normalized_intent,
            evidence=normalized_evidence,
        )
        ensure_private_dir(self.directory)
        request_dir = self.directory / request.approval_id
        request_dir.mkdir(mode=0o700)
        write_private_file(request_dir / REQUEST_FILE, _dump(request))
        self._fsync_directory(request_dir)
        self._fsync_directory(self.directory)
        return request

    def get(self, approval_id: str) -> tuple[ApprovalRequest, ApprovalResolution | None]:
        """Load a request and its terminal resolution, refusing incomplete state."""
        request_dir = self._request_dir(approval_id)
        try:
            request = ApprovalRequest.model_validate_json(
                (request_dir / REQUEST_FILE).read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            raise ApprovalError(
                "approval_state_invalid",
                f"approval {approval_id} is missing or corrupt ({exc}). Nothing can be placed; "
                "wait for a newly evaluated intent.",
                status=409,
            ) from exc
        if request.agent_id != self.agent_id:
            raise ApprovalError(
                "approval_state_invalid",
                f"approval {approval_id} belongs to another agent. Nothing can be placed; "
                "wait for this agent to evaluate again.",
                status=409,
            )
        resolution_path = request_dir / RESOLUTION_FILE
        if not resolution_path.exists():
            return request, None
        try:
            resolution = ApprovalResolution.model_validate_json(
                resolution_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            raise ApprovalError(
                "approval_state_invalid",
                f"approval {approval_id}'s resolution is corrupt ({exc}). Nothing can be "
                "placed; wait for a newly evaluated intent.",
                status=409,
            ) from exc
        return request, resolution

    def pending(self) -> tuple[ApprovalRequest, ...]:
        """Return only valid pending requests, materializing old-boot interruptions."""
        if not self.directory.is_dir():
            return ()
        pending: list[ApprovalRequest] = []
        for entry in sorted(self.directory.iterdir()):
            if not entry.is_dir():
                continue
            try:
                request, resolution = self.get(entry.name)
                if resolution is None:
                    resolution = self._terminal_if_invalid(request)
                if resolution is None:
                    pending.append(request)
            except ApprovalError:
                continue
        return tuple(pending)

    def decide(self, approval_id: str, *, approve: bool, decided_via: str) -> ApprovalResolution:
        """Resolve once; a decision at either deadline is already expired."""
        if decided_via not in {"terminal", "api", "chat"}:
            raise ValueError("decided_via must be terminal, api, or chat")
        request_dir = self._request_dir(approval_id)
        with self._locked(request_dir):
            request, resolution = self.get(approval_id)
            if resolution is not None:
                self._raise_resolved(resolution)
            terminal = self._terminal_if_invalid(request, already_locked=True)
            if terminal is not None:
                if terminal.outcome is ApprovalOutcome.EXPIRED:
                    raise ApprovalError(
                        "approval_expired", terminal.reason, status=410, resolution=terminal
                    )
                self._raise_resolved(terminal)
            outcome = ApprovalOutcome.APPROVED if approve else ApprovalOutcome.DECLINED
            reason = (
                "approval accepted before the box deadline; the runner must recheck every "
                "dispatch condition before placing."
                if approve
                else "approval declined; nothing was placed, and a later tick may be evaluated."
            )
            return self._write_resolution(
                request,
                ApprovalResolution(
                    approval_id=approval_id,
                    outcome=outcome,
                    decided_via=decided_via,
                    decided_at=self._aware_now(),
                    reason=reason,
                ),
            )

    def abort_by_stop(self, approval_id: str) -> ApprovalResolution:
        """Make STOP terminal for a pending request before waking its waiter."""
        return self._resolve_system(
            approval_id,
            ApprovalOutcome.ABORTED_BY_STOP,
            "the kill switch was set while approval was pending. Nothing was placed; "
            "remove STOP only when you want a later tick to evaluate again.",
        )

    def interrupt(self, approval_id: str) -> ApprovalResolution:
        """Bind pending authority to the dead run; it is never resumed."""
        return self._resolve_system(
            approval_id,
            ApprovalOutcome.INTERRUPTED,
            "the run or boot changed while approval was pending. Nothing was placed; "
            "wait for the current run to evaluate a new intent.",
        )

    def expire(self, approval_id: str) -> ApprovalResolution:
        """Materialize expiration; useful to a waiter whose timer reaches the boundary."""
        return self._resolve_system(
            approval_id,
            ApprovalOutcome.EXPIRED,
            "the approval window expired without a decision. Nothing was placed; "
            "wait for a later tick to evaluate again.",
        )

    def wait(
        self,
        approval_id: str,
        *,
        run_id: str,
        stop_requested: Callable[[], bool],
        terminal_decision: Callable[[], bool] | None,
        wait_for_change: Callable[[float], None],
    ) -> ApprovalResolution:
        """Wait for API, terminal, deadline, restart, or STOP—whichever commits first."""
        if terminal_decision is not None:

            def ask() -> None:
                try:
                    answer = bool(terminal_decision())
                    self.decide(approval_id, approve=answer, decided_via="terminal")
                except ApprovalError:
                    return

            threading.Thread(target=ask, daemon=True, name=f"approval-{approval_id}").start()
        while True:
            request, resolution = self.get(approval_id)
            if resolution is not None:
                return resolution
            if request.run_id != run_id or request.boot_id != self._current_boot_id():
                return self.interrupt(approval_id)
            if stop_requested():
                return self.abort_by_stop(approval_id)
            terminal = self._terminal_if_invalid(request)
            if terminal is not None:
                return terminal
            remaining_wall = (request.deadline_wall - self._aware_now()).total_seconds()
            remaining_mono = float(request.deadline_monotonic) - self._monotonic_clock()
            wait_for_change(max(0.0, min(0.1, remaining_wall, remaining_mono)))

    def _resolve_system(
        self, approval_id: str, outcome: ApprovalOutcome, reason: str
    ) -> ApprovalResolution:
        request_dir = self._request_dir(approval_id)
        with self._locked(request_dir):
            request, existing = self.get(approval_id)
            if existing is not None:
                return existing
            return self._write_resolution(
                request,
                ApprovalResolution(
                    approval_id=approval_id,
                    outcome=outcome,
                    decided_via=None,
                    decided_at=self._aware_now(),
                    reason=reason,
                ),
            )

    def _terminal_if_invalid(
        self, request: ApprovalRequest, *, already_locked: bool = False
    ) -> ApprovalResolution | None:
        if request.boot_id != self._current_boot_id():
            if already_locked:
                return self._write_resolution(
                    request,
                    ApprovalResolution(
                        approval_id=request.approval_id,
                        outcome=ApprovalOutcome.INTERRUPTED,
                        decided_via=None,
                        decided_at=self._aware_now(),
                        reason=(
                            "the box restarted while approval was pending. Nothing was placed; "
                            "wait for the current run to evaluate a new intent."
                        ),
                    ),
                )
            return self.interrupt(request.approval_id)
        now = self._aware_now()
        monotonic_now = Decimal(str(self._monotonic_clock()))
        if now >= request.deadline_wall or monotonic_now >= request.deadline_monotonic:
            if already_locked:
                return self._write_resolution(
                    request,
                    ApprovalResolution(
                        approval_id=request.approval_id,
                        outcome=ApprovalOutcome.EXPIRED,
                        decided_via=None,
                        decided_at=now,
                        reason=(
                            "the approval window expired before this decision. Nothing was "
                            "placed; wait for a later tick to evaluate again."
                        ),
                    ),
                )
            return self.expire(request.approval_id)
        return None

    def _write_resolution(
        self, request: ApprovalRequest, resolution: ApprovalResolution
    ) -> ApprovalResolution:
        if request.approval_id != resolution.approval_id:
            raise ValueError("a resolution must name its request")
        path = self._request_dir(request.approval_id) / RESOLUTION_FILE
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, FILE_MODE)
        try:
            os.fchmod(descriptor, FILE_MODE)
            os.write(descriptor, _dump(resolution).encode("utf-8"))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._fsync_directory(path.parent)
        return resolution

    @contextmanager
    def _locked(self, request_dir: Path) -> Iterator[None]:
        key = str(request_dir.resolve())
        with _THREAD_LOCKS_GUARD:
            thread_lock = _THREAD_LOCKS.setdefault(key, threading.Lock())
        try:
            descriptor = os.open(request_dir / LOCK_FILE, os.O_CREAT | os.O_RDWR, FILE_MODE)
        except OSError as exc:
            raise ApprovalError(
                "approval_state_invalid",
                f"approval state cannot be locked ({exc}). Nothing can be placed; wait for "
                "a newly evaluated intent.",
                status=409,
            ) from exc
        try:
            with thread_lock:
                os.fchmod(descriptor, FILE_MODE)
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def _request_dir(self, approval_id: str) -> Path:
        if not approval_id or Path(approval_id).name != approval_id or approval_id in {".", ".."}:
            raise ApprovalError(
                "approval_not_found",
                f"{approval_id!r} is not an approval id. Refresh the pending approvals list.",
                status=404,
            )
        path = self.directory / approval_id
        if not path.is_dir():
            raise ApprovalError(
                "approval_not_found",
                f"approval {approval_id} does not exist. Refresh the pending approvals list.",
                status=404,
            )
        return path

    def _raise_resolved(self, resolution: ApprovalResolution) -> None:
        raise ApprovalError(
            "already_resolved",
            f"approval {resolution.approval_id} is already {resolution.outcome.value}. "
            "Nothing changed; refresh the approval list.",
            status=409,
            resolution=resolution,
        )

    def _aware_now(self) -> datetime:
        now = self._wall_clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("the approval wall clock must return a timezone-aware datetime")
        return now

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
