"""How a run executes, and who approves each order.

Both are recorded on every run and both are deliberately small closed sets. A
mode is not a flag with a truthy default: `Mode.PAPER` is what a fresh agent
gets, `Mode.LIVE` only ever arrives from an affirmative `--live` on the command
line, and the switch is written into the record before anything else happens
(CLAUDE.md invariant 2).
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["ApprovalMode", "Mode"]


class Mode(StrEnum):
    """Where an order goes: Tick's local simulation, or a real brokerage."""

    PAPER = "paper"
    LIVE = "live"

    @property
    def tag(self) -> str:
        """The word every notification ends with, so a simulation always says so."""
        return "simulated" if self is Mode.PAPER else "live"

    @property
    def is_simulated(self) -> bool:
        return self is Mode.PAPER


class ApprovalMode(StrEnum):
    """Whether the user is asked before each order.

    Robinhood's Agentic account offers both, and so does Tick: `EACH` asks per
    order, `STANDING` does not ask at all. There is no third state where the
    runtime decides for itself which orders are worth asking about.
    """

    #: Ask the user before every order; a decline places nothing and is recorded.
    EACH = "each"
    #: The user has already permitted the agent's orders for this run.
    STANDING = "standing"
