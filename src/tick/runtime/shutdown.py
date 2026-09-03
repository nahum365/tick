"""The durable part of signal shutdown, isolated for deterministic tests."""

from __future__ import annotations

from datetime import datetime

from tick.records import DataSource, RecordKind

from .state import AgentRun

__all__ = ["stop_by_signal"]


def stop_by_signal(agent: AgentRun, *, signal_name: str, at: datetime) -> None:
    """Set STOP first, then append the human-readable shutdown observation."""
    if not signal_name.strip():
        raise ValueError("signal shutdown must name the received signal")
    agent.request_stop(reason=f"stopped by {signal_name}", at=at)
    agent.ledger(clock=lambda: at).append(
        RecordKind.NOTE,
        {
            "event": "stopped_by_signal",
            "signal": signal_name,
            "reason": (
                f"the run stopped after {signal_name}. The broker session is closing; "
                f"remove {agent.stop_path} only when a later run should start."
            ),
            "at": at,
        },
        source=DataSource.RUNTIME,
    )
