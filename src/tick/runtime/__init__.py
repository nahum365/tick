"""The runtime: the clock, the schedule, the agent's state, and one tick.

    from tick.runtime import MarketClock, Scheduler

    clock = MarketClock.for_2026()
    when = Scheduler(clock).next_tick(spec.cadence, now)

Everything here runs on the user's own machine and reaches no network. The
market calendar is hand-entered and partial (`clock.py`), which is why the
clock refuses a year it does not cover rather than assuming every weekday is a
session.
"""

from __future__ import annotations

from . import notify
from .approvals import (
    ApprovalError,
    ApprovalOutcome,
    ApprovalQueue,
    ApprovalRequest,
    ApprovalResolution,
    ApprovalWindow,
    boot_id,
    parse_approval_window,
)
from .clock import (
    EARLY_CLOSE_TIME,
    EARLY_CLOSES_2026,
    EASTERN,
    HOLIDAYS_2026,
    REGULAR_CLOSE,
    REGULAR_OPEN,
    MarketClock,
)
from .doctor import DoctorCheck, DoctorReport, acknowledge_demotion, run_doctor
from .errors import (
    CalendarUnavailable,
    LedgerQuarantined,
    ModeNotWired,
    NotificationRefused,
    RuntimeStateError,
    TickRuntimeError,
)
from .launch import (
    LaunchError,
    LaunchTicket,
    RunLease,
    consume_launch_ticket,
    create_launch_ticket,
    launch_lock,
    load_run_lease,
    run_state,
    save_run_lease,
    ticket_directory,
)
from .live import (
    LIVE_CAPABILITIES,
    LiveReadiness,
    check_live_ready,
    check_local_live_ready,
    local_actor,
    record_first_live_place_proof,
)
from .modes import ApprovalMode, Mode
from .notify import FORBIDDEN_PHRASES
from .reconcile import reconcile_unknown_orders, unknown_order_ids
from .runner import ApprovalDecision, IntentSource, Runner, TickOutcome, rule_id_of
from .schedule import DAILY_CLOSE_OFFSET, DAILY_OPEN_OFFSET, MIN_CADENCE_MINUTES, Scheduler
from .shutdown import stop_by_signal
from .state import (
    AGENT_ID_LENGTH,
    INSTRUCTIONS_FILE,
    SPEC_FILE,
    STATE_FILE,
    STOP_FILE,
    AgentRun,
    AgentState,
    agent_id_for,
    agents_dir,
    state_summary,
)
from .status import joined_agent_status

__all__ = [
    "AGENT_ID_LENGTH",
    "INSTRUCTIONS_FILE",
    "LIVE_CAPABILITIES",
    "FORBIDDEN_PHRASES",
    "SPEC_FILE",
    "STATE_FILE",
    "STOP_FILE",
    "DAILY_CLOSE_OFFSET",
    "DAILY_OPEN_OFFSET",
    "MIN_CADENCE_MINUTES",
    "EARLY_CLOSES_2026",
    "EARLY_CLOSE_TIME",
    "EASTERN",
    "HOLIDAYS_2026",
    "REGULAR_CLOSE",
    "REGULAR_OPEN",
    "AgentRun",
    "AgentState",
    "ApprovalError",
    "ApprovalDecision",
    "ApprovalMode",
    "ApprovalOutcome",
    "ApprovalQueue",
    "ApprovalRequest",
    "ApprovalResolution",
    "ApprovalWindow",
    "IntentSource",
    "CalendarUnavailable",
    "DoctorCheck",
    "DoctorReport",
    "LedgerQuarantined",
    "LiveReadiness",
    "LaunchError",
    "LaunchTicket",
    "MarketClock",
    "Mode",
    "ModeNotWired",
    "NotificationRefused",
    "Runner",
    "RunLease",
    "RuntimeStateError",
    "Scheduler",
    "TickOutcome",
    "TickRuntimeError",
    "agent_id_for",
    "acknowledge_demotion",
    "agents_dir",
    "boot_id",
    "check_live_ready",
    "check_local_live_ready",
    "local_actor",
    "joined_agent_status",
    "consume_launch_ticket",
    "create_launch_ticket",
    "launch_lock",
    "load_run_lease",
    "parse_approval_window",
    "run_state",
    "run_doctor",
    "reconcile_unknown_orders",
    "record_first_live_place_proof",
    "save_run_lease",
    "ticket_directory",
    "notify",
    "rule_id_of",
    "state_summary",
    "stop_by_signal",
    "unknown_order_ids",
]
