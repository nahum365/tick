"""One-release import compatibility for the verified profile broker.

The prototype adapter accepted a raw ``ToolMap``. Keeping that constructor
would preserve a route around fresh contract verification, so the old public
name now has exactly the same verified-session boundary as ``ProfileBroker``.
Prototype files are migrated by :mod:`tick.broker.profile`; this module does
not interpret or call through them.
"""

from __future__ import annotations

from .profile_broker import ProfileBroker

__all__ = ["RobinhoodMCPBroker"]


class RobinhoodMCPBroker(ProfileBroker):
    """Deprecated alias for :class:`ProfileBroker`; use that name instead."""
