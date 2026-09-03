"""The approval queue is one durable, deadline-bounded authority decision."""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from tick.runtime import ApprovalError, ApprovalOutcome, ApprovalQueue, ApprovalWindow


class Clocks:
    def __init__(self) -> None:
        self.wall = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)
        self.monotonic = 100.0
        self.boot = "boot-one"


@pytest.fixture
def clocks() -> Clocks:
    return Clocks()


@pytest.fixture
def queue(tmp_path, clocks: Clocks) -> ApprovalQueue:
    return ApprovalQueue(
        tmp_path,
        "0123456789ab",
        wall_clock=lambda: clocks.wall,
        monotonic_clock=lambda: clocks.monotonic,
        current_boot_id=lambda: clocks.boot,
    )


def create(queue: ApprovalQueue):
    return queue.create(
        run_id="run-one",
        tick_id="tick-one",
        window=ApprovalWindow(seconds=300),
        symbol="XYZ",
        side="buy",
        qty=2,
        est_price=Decimal("10.25"),
        price_source="fixture",
        data_class="local_fixture",
        est_notional=Decimal("20.50"),
        cage_checks=("session", "order_notional"),
        proposed_by="rule:one",
        intent={"symbol": "XYZ", "qty": 2},
        evidence={"price": Decimal("10.25")},
    )


def test_decision_at_deadline_expires(queue: ApprovalQueue, clocks: Clocks):
    request = create(queue)
    clocks.wall = request.deadline_wall

    with pytest.raises(ApprovalError) as caught:
        queue.decide(request.approval_id, approve=True, decided_via="api")

    assert caught.value.status == 410
    assert caught.value.code == "approval_expired"
    assert queue.get(request.approval_id)[1].outcome is ApprovalOutcome.EXPIRED
    assert "Nothing was placed" in caught.value.reason


def test_backward_wall_jump_does_not_extend_approval(queue: ApprovalQueue, clocks: Clocks):
    request = create(queue)
    clocks.wall -= timedelta(days=1)
    clocks.monotonic = float(request.deadline_monotonic)

    with pytest.raises(ApprovalError) as caught:
        queue.decide(request.approval_id, approve=True, decided_via="terminal")

    assert caught.value.code == "approval_expired"


def test_old_boot_approval_is_not_reused(queue: ApprovalQueue, clocks: Clocks):
    request = create(queue)
    clocks.boot = "boot-two"

    assert queue.pending() == ()
    resolution = queue.get(request.approval_id)[1]
    assert resolution is not None
    assert resolution.outcome is ApprovalOutcome.INTERRUPTED


def test_first_terminal_resolution_wins(queue: ApprovalQueue):
    request = create(queue)
    barrier = threading.Barrier(3)
    outcomes: list[str] = []

    def decide(approve: bool) -> None:
        barrier.wait()
        try:
            outcomes.append(
                queue.decide(request.approval_id, approve=approve, decided_via="api").outcome
            )
        except ApprovalError as exc:
            outcomes.append(exc.code)

    threads = [threading.Thread(target=decide, args=(choice,)) for choice in (True, False)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert "already_resolved" in outcomes
    resolution = queue.get(request.approval_id)[1]
    assert resolution is not None
    assert outcomes.count(resolution.outcome) == 1


def test_stop_is_terminal_and_late_answer_is_rejected(queue: ApprovalQueue):
    request = create(queue)
    resolution = queue.abort_by_stop(request.approval_id)
    assert resolution.outcome is ApprovalOutcome.ABORTED_BY_STOP

    with pytest.raises(ApprovalError) as caught:
        queue.decide(request.approval_id, approve=True, decided_via="api")

    assert caught.value.status == 409
    assert "already aborted_by_stop" in caught.value.reason


def test_incomplete_queue_state_fails_closed(queue: ApprovalQueue):
    request = create(queue)
    (queue.directory / request.approval_id / "request.json").write_text("{}")

    with pytest.raises(ApprovalError) as caught:
        queue.get(request.approval_id)

    assert caught.value.code == "approval_state_invalid"
    assert "Nothing can be placed" in caught.value.reason
