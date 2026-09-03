"""One process/ledger status join shared by CLI, box API, and the future doctor."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .launch import load_run_lease, run_state
from .modes import Mode
from .state import AgentRun, state_summary

__all__ = ["joined_agent_status"]


def joined_agent_status(
    run: AgentRun,
    *,
    pid_alive: Callable[[int], bool],
    current_boot_id: str,
) -> dict[str, Any]:
    """Join declared agent state to observed process liveness and boot history."""
    summary = state_summary(run)
    lease = load_run_lease(run.home, run.agent_id)
    observed = run_state(lease, pid_alive=pid_alive, current_boot_id=current_boot_id)
    current_mode = lease.mode.value if lease is not None and observed == "running" else None
    previous_mode = (
        lease.previous_run_mode.value if lease is not None and lease.previous_run_mode else None
    )
    transition = None
    attention = False
    if lease is not None and lease.boot_id != current_boot_id and lease.mode is Mode.LIVE:
        transition = "reboot_demoted_live_to_paper"
        previous_mode = Mode.LIVE.value
        attention = True
    elif (
        lease is not None
        and lease.mode is Mode.PAPER
        and lease.previous_run_mode is Mode.LIVE
        and lease.previous_run_boot_id != current_boot_id
    ):
        transition = "reboot_demoted_live_to_paper"
        previous_mode = Mode.LIVE.value
        attention = True
    return {
        **summary,
        # ``state.mode`` is historical compatibility, never current authority.
        "mode": current_mode or Mode.PAPER.value,
        "run_state": observed,
        "current_mode": current_mode,
        "previous_run_mode": previous_mode,
        "transition": transition,
        "attention_required": attention,
        "last_contact": lease.started_at.isoformat() if lease else summary["last_tick"],
    }
