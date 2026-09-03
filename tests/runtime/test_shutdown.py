"""Signal shutdown sets the durable switch and leaves readable evidence."""

from datetime import UTC, datetime

from tick.records import RecordKind, read
from tick.runtime import stop_by_signal


def test_signal_shutdown_sets_stop_before_recording_the_observation(agent):
    at = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)

    stop_by_signal(agent, signal_name="SIGTERM", at=at)

    assert agent.stop_requested()
    assert agent.stop_reason() == "stopped by SIGTERM"
    row = list(read(agent.ledger_path))[-1]
    assert row.kind is RecordKind.NOTE
    assert row.payload["event"] == "stopped_by_signal"
    assert "remove" in row.payload["reason"]
