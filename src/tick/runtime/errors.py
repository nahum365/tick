"""Errors the runtime raises — the conditions under which a tick must not run.

Everything here stops the runtime rather than shaping a decision. That is the
difference from the engine's `Unavailable`/`Refusal` values: a missing quote is
a fact about one rule, while a calendar we do not have, a ledger that does not
verify, or a mode nobody wired is a fact about whether the runtime may act at
all.

Every one of them answers the fail-safe question the same way: after the
failure, the user can still stop the agent, still read the record, and still
see which record failed and what to do next. None of them is recovered from by
guessing.
"""

from __future__ import annotations

__all__ = [
    "CalendarUnavailable",
    "LedgerQuarantined",
    "ModeNotWired",
    "NotificationRefused",
    "RuntimeStateError",
    "TickRuntimeError",
]


class TickRuntimeError(Exception):
    """Base class for every runtime failure."""


class CalendarUnavailable(TickRuntimeError):
    """A date falls outside the years this clock has a market calendar for.

    The holiday list is hand-entered and partial (`clock.py`). Asked about a
    year it does not cover, the clock refuses instead of assuming every weekday
    is a session — assuming would place orders on a day the market was shut,
    which is invariant 5's fabrication wearing a calendar.
    """


class LedgerQuarantined(TickRuntimeError):
    """The agent's ledger does not verify, so the runtime will not trade.

    A runtime that cannot record must not act: an order placed against a broken
    record is an order with no evidence. Carries the failing `seq`, the reason,
    and the exact next step (`tick ledger new <agent_id>`), because a refusal
    with no next step is half a fail-safe.
    """

    def __init__(self, agent_id: str, *, seq: int | None, reason: str, next_step: str) -> None:
        self.agent_id = agent_id
        self.seq = seq
        self.reason = reason
        self.next_step = next_step
        where = "the ledger" if seq is None else f"record {seq}"
        super().__init__(
            f"agent {agent_id}: {where} does not verify ({reason}). Nothing was placed "
            f"and nothing was recorded. Start a successor ledger with: {next_step}"
        )


class ModeNotWired(TickRuntimeError):
    """A run's mode and its broker do not agree about where orders are going.

    Live orders go to the brokerage adapter and paper orders go to the local
    simulation; there is no third pairing. Either mismatch refuses loudly and
    places nothing, because paper is the default and live is an explicit,
    logged act (invariant 2) — and a `--live` that quietly ran paper, or a
    paper run that reached a real account, would make the record a lie about
    what happened to somebody's money.
    """


class RuntimeStateError(TickRuntimeError):
    """The on-disk state of an agent is missing, unreadable, or not this agent's."""


class NotificationRefused(TickRuntimeError):
    """A sentence the notification grammar will not send.

    Raised by `notify.py` when a composed notification carries a phrase the
    product may not say (invariant 6). The runner catches it, withholds the
    text, and records the reason — it never sends the sentence anyway and never
    silently drops the notification.
    """
