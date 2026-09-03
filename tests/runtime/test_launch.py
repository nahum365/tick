"""Live authority is a one-use ticket bound to one boot."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from tick.runtime import (
    ApprovalMode,
    LaunchError,
    consume_launch_ticket,
    create_launch_ticket,
)


def test_live_ticket_is_consumed_once(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    path, fallback = create_launch_ticket(
        tmp_path / "home",
        agent_id="0123456789ab",
        run_id="run-one",
        approval_mode=ApprovalMode.EACH,
        standing_ok=False,
        created_at=datetime(2026, 9, 3, tzinfo=UTC),
        env={"XDG_RUNTIME_DIR": str(runtime)},
        current_boot_id="boot-one",
    )
    assert fallback is False
    ticket = consume_launch_ticket(
        path,
        agent_id="0123456789ab",
        run_id="run-one",
        approval_mode=ApprovalMode.EACH,
        standing_ok=False,
        current_boot_id="boot-one",
    )
    assert ticket.boot_id == "boot-one"

    with pytest.raises(LaunchError) as caught:
        consume_launch_ticket(
            path,
            agent_id="0123456789ab",
            run_id="run-one",
            approval_mode=ApprovalMode.EACH,
            standing_ok=False,
            current_boot_id="boot-one",
        )
    assert caught.value.code == "live_ticket_missing"
    assert "Nothing was connected or placed" in caught.value.reason


def test_copied_live_command_cannot_cross_a_boot(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    path, _ = create_launch_ticket(
        tmp_path / "home",
        agent_id="0123456789ab",
        run_id="run-one",
        approval_mode=ApprovalMode.EACH,
        standing_ok=False,
        created_at=datetime(2026, 9, 3, tzinfo=UTC),
        env={"XDG_RUNTIME_DIR": str(runtime)},
        current_boot_id="boot-one",
    )

    with pytest.raises(LaunchError) as caught:
        consume_launch_ticket(
            path,
            agent_id="0123456789ab",
            run_id="run-one",
            approval_mode=ApprovalMode.EACH,
            standing_ok=False,
            current_boot_id="boot-two",
        )

    assert caught.value.code == "live_ticket_mismatch"
    assert not path.exists()
