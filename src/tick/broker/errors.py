"""What can go wrong between Tick and a broker's MCP, as types callers act on.

The distinctions matter because the runtime's response differs. A transport
that dropped is a **stop**: invariant 8 says the runtime halts and notifies
rather than retrying an order it cannot see the outcome of. A tool whose
result does not fit the mapping is **invariant 7** refusing to guess. A
capability nobody has mapped is a configuration gap the user can close, and
saying which of the three happened is the difference between a message someone
can act on and one they can only forward.
"""

from __future__ import annotations

__all__ = [
    "BrokerError",
    "BrokerUnavailable",
    "CapabilityUnmapped",
    "ToolResultUnreadable",
]


class BrokerError(Exception):
    """Base for every failure of the broker adapter."""


class BrokerUnavailable(BrokerError):
    """The MCP session could not be opened, or dropped, or timed out.

    The runtime stops on this. It never re-places an order whose fate it does
    not know: a blind retry after a timeout is how one intent becomes two fills.
    """


class ToolResultUnreadable(BrokerError):
    """A tool answered in a shape the mapping does not describe (invariant 7)."""


class CapabilityUnmapped(BrokerError):
    """Tick needs a capability that no discovered tool has been mapped to."""
