"""Ordered readiness checks shared by ``tick doctor`` and the box API.

All effects are injected.  The runtime package stays local-only while the
transport layer supplies the two observations that can touch the operating
system: provider command output and an authenticated loopback request.
"""

from __future__ import annotations

import json
import stat
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from tick.auth import FileTokenStorage
from tick.broker import BrokerError, Category, ProfileState, load_profile
from tick.records import write_private_file

from .approvals import boot_id
from .launch import load_run_lease
from .modes import ApprovalMode
from .state import AgentRun
from .status import joined_agent_status

__all__ = ["DoctorCheck", "DoctorReport", "acknowledge_demotion", "run_doctor"]


@dataclass(frozen=True, slots=True)
class DoctorCheck:
    name: str
    status: Literal["ok", "refuse"]
    sentence: str
    detail: Mapping[str, Any] | None

    def line(self) -> str:
        return f"{self.status}: {self.name} — {self.sentence}"


@dataclass(frozen=True, slots=True)
class DoctorReport:
    checks: tuple[DoctorCheck, ...]

    @property
    def ready(self) -> bool:
        return all(check.status == "ok" for check in self.checks)

    def json(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "checks": [
                {
                    "name": item.name,
                    "status": item.status,
                    "sentence": item.sentence,
                    "detail": item.detail,
                }
                for item in self.checks
            ],
        }


def _check(name: str, good: bool, yes: str, no: str, detail=None) -> DoctorCheck:
    return DoctorCheck(
        name=name, status="ok" if good else "refuse", sentence=yes if good else no, detail=detail
    )


def _mode(path: Path) -> int | None:
    try:
        return stat.S_IMODE(path.stat().st_mode)
    except OSError:
        return None


def _acks_path(home: Path) -> Path:
    return home / "doctor" / "demotion-acks.json"


def _acks(home: Path) -> set[str]:
    try:
        values = json.loads(_acks_path(home).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    return (
        {value for value in values if isinstance(value, str)} if isinstance(values, list) else set()
    )


def acknowledge_demotion(home: Path, run_id: str, statuses: Sequence[Mapping[str, Any]]) -> Path:
    """Acknowledge one observed reboot transition, never store a preference."""
    observed = any(
        status.get("transition") == "reboot_demoted_live_to_paper"
        and status.get("run_id") == run_id
        for status in statuses
    )
    if not observed:
        raise ValueError(
            f"run {run_id} has no observed reboot demotion. Refresh doctor and "
            "acknowledge a listed run id."
        )
    values = sorted(_acks(home) | {run_id})
    return write_private_file(_acks_path(home), json.dumps(values, indent=2) + "\n")


def run_doctor(
    home: Path,
    *,
    now: datetime,
    provider_status: Callable[[], tuple[bool, str]],
    loopback_status: Callable[[], tuple[bool, str]],
    tunnel_status: Callable[[], tuple[bool, str]],
    unit_fragments: Callable[[], tuple[bool, str, Sequence[str]]],
    pid_alive: Callable[[int], bool],
) -> DoctorReport:
    """Check every live dependency in the required stable display order."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("doctor needs a timezone-aware current time")
    checks: list[DoctorCheck] = []
    fresh_home = not home.exists() or not any(home.iterdir())

    home_mode = _mode(home)
    children = list(home.rglob("*")) if home.exists() else []
    bad_directories = [str(path) for path in children if path.is_dir() and _mode(path) != 0o700]
    # Agent documents are deliberately read-only (0400); mutable state and credentials are 0600.
    bad_files = [
        str(path) for path in children if path.is_file() and _mode(path) not in {0o400, 0o600}
    ]
    modes_ok = home_mode == 0o700 and not bad_directories and not bad_files
    checks.append(
        _check(
            "TICK_HOME modes",
            modes_ok,
            f"{home} and its directories are 0700; private files are 0600 or read-only 0400.",
            f"make {home} and its directories mode 0700 and private files 0600 "
            "(agent documents may be 0400), then run doctor again.",
            {
                "home_mode": f"{home_mode:04o}" if home_mode is not None else None,
                "bad_directories": bad_directories,
                "bad_files": bad_files,
            },
        )
    )

    provider_ok, provider_sentence = provider_status()
    checks.append(_check("provider codex", provider_ok, provider_sentence, provider_sentence))

    grant = FileTokenStorage(home).connected()
    checks.append(
        _check(
            "broker grant",
            grant,
            "the local broker grant is present.",
            "run `tick connect robinhood`; paper rule agents can still run without it.",
        )
    )

    try:
        profile = load_profile(home)
        profile_error = None
    except BrokerError as exc:
        profile, profile_error = None, str(exc)
    profile_ok = profile is not None and profile.state is ProfileState.CONFIRMED
    if profile is None:
        profile_sentence = (
            f"{profile_error}. Propose and confirm the profile again."
            if profile_error
            else "run `tick broker propose`, review it, and confirm each required tool."
        )
        profile_detail = None
    elif profile.state is ProfileState.DRIFTED:
        profile_sentence = (
            "the profile drifted; inspect this diff and reconfirm only changed dependencies."
        )
        profile_detail = {
            "profile_hash": profile.profile_hash,
            "diff": [item.model_dump(mode="json") for item in profile.drift],
        }
    else:
        profile_sentence = f"confirmed profile {profile.profile_hash}."
        profile_detail = {"profile_hash": profile.profile_hash}
    checks.append(
        _check("broker profile", profile_ok, profile_sentence, profile_sentence, profile_detail)
    )

    proof_required = {
        Category.READ_QUOTE,
        Category.READ_POSITIONS,
        Category.READ_BALANCES,
        Category.READ_ORDERS,
    }
    if profile is not None:
        proof_required.update(
            tool.category
            for tool in profile.tools.values()
            if tool.category is Category.ORDER_PREFLIGHT
        )
    proof_tools = {
        category: next((tool for tool in profile.tools.values() if tool.category is category), None)
        if profile is not None
        else None
        for category in proof_required
    }
    missing_proof = [
        category.value
        for category, tool in sorted(proof_tools.items(), key=lambda item: item[0].value)
        if tool is None or not tool.proved
    ]
    proved_at = [tool.proved_at for tool in proof_tools.values() if tool and tool.proved_at]
    oldest = min(proved_at) if proved_at else None
    age = now.astimezone(UTC) - oldest.astimezone(UTC) if oldest is not None else None
    proof_ok = profile is not None and not missing_proof
    proof_yes = (
        f"all required read proofs pass; oldest proof is {int(age.total_seconds())} seconds old."
        if age is not None
        else "all required read proofs pass."
    )
    proof_no = (
        f"proof is missing for {', '.join(missing_proof)}. Run `tick broker prove` with "
        "your probe values; paper runs remain available."
        if missing_proof
        else "run `tick broker prove` with your probe values; no proof profile is present."
    )
    checks.append(
        _check(
            "broker proof",
            proof_ok,
            proof_yes,
            proof_no,
            {
                "oldest_proof_age_seconds": int(age.total_seconds()) if age else None,
                "missing": missing_proof,
            },
        )
    )

    agent_ids = AgentRun.list_ids(home)
    ledger_failures: list[str] = []
    statuses: list[dict[str, Any]] = []
    for agent_id in agent_ids:
        try:
            agent = AgentRun.load(home, agent_id)
            verification = agent.verify_ledger()
            if not verification.ok:
                ledger_failures.append(f"{agent_id}: {verification}")
            joined = joined_agent_status(agent, pid_alive=pid_alive, current_boot_id=boot_id())
            lease = load_run_lease(home, agent_id)
            joined["run_id"] = lease.run_id if lease else None
            statuses.append(joined)
        except Exception as exc:  # noqa: BLE001 - every unreadable agent must be named
            ledger_failures.append(f"{agent_id}: {exc}")
    ledgers_ok = bool(agent_ids) and not ledger_failures
    checks.append(
        _check(
            "agent ledgers",
            ledgers_ok,
            f"verified {len(agent_ids)} agent ledger(s).",
            "create an agent, or start a successor ledger for each listed failure.",
            {"failures": ledger_failures},
        )
    )

    pairing = home / "pairing" / "secret"
    pairing_ok = pairing.exists() and _mode(pairing) == 0o600
    checks.append(
        _check(
            "pairing secret",
            pairing_ok,
            "the pairing secret is present with mode 0600.",
            "run `tick pair new`, then pair the app again.",
        )
    )

    reachable, reach_sentence = loopback_status()
    checks.append(_check("serve loopback", reachable, reach_sentence, reach_sentence))

    tunnel_ok, tunnel_sentence = tunnel_status()
    checks.append(_check("direct tunnel", tunnel_ok, tunnel_sentence, tunnel_sentence))

    units_read, units_sentence, fragments = unit_fragments()
    live_fragments = [fragment for fragment in fragments if "--live" in fragment]
    units_ok = units_read and not live_fragments
    units_no = (
        "remove persistent --live from the enabled unit fragments before any live launch. "
        "Doctor cannot detect every root-owned script."
        if live_fragments
        else units_sentence
    )
    checks.append(
        _check(
            "persistent launch units",
            units_ok,
            units_sentence + " Doctor cannot detect every root-owned script.",
            units_no,
            {"live_fragments": live_fragments},
        )
    )

    approval_ok = bool(agent_ids) and all(
        status.get("approval") == ApprovalMode.STANDING.value
        or status.get("run_state") == "running"
        for status in statuses
    )
    checks.append(
        _check(
            "approval window",
            approval_ok,
            "every per-order run has an active launch window, or the agent is standing.",
            "launch each-approval agents with `--approval-window <duration>`; "
            "paper runs remain available.",
        )
    )

    state_ok = bool(statuses) and all(status.get("current_mode") != "live" for status in statuses)
    checks.append(
        _check(
            "agent paper/live state",
            state_ok,
            "every agent is paper or stopped; live remains per launch.",
            "create an agent and verify its paper/live state before launching live.",
            {"agents": statuses},
        )
    )

    unacked = [
        status
        for status in statuses
        if status.get("transition") == "reboot_demoted_live_to_paper"
        and status.get("run_id") not in _acks(home)
    ]
    checks.append(
        _check(
            "reboot demotion",
            bool(statuses) and not unacked,
            "no unacknowledged live-to-paper reboot demotion is present.",
            (
                "a live run was demoted to paper after reboot; inspect it, then run "
                "`tick doctor --ack-demotion <run_id>`."
                if unacked
                else "create an agent so doctor can assess reboot demotion state; "
                "paper setup can continue."
            ),
            {"observations": unacked},
        )
    )
    if fresh_home:
        checks = [
            check
            if check.status == "refuse"
            else DoctorCheck(
                name=check.name,
                status="refuse",
                sentence=(
                    f"{check.sentence} This TICK_HOME has no configured owner run; "
                    "complete the walkthrough setup, then run doctor again."
                ),
                detail=check.detail,
            )
            for check in checks
        ]
    return DoctorReport(tuple(checks))
