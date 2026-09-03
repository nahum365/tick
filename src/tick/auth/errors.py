"""What can go wrong while connecting, as types a caller can act on.

A connect ceremony fails in ways that call for different things from the
person running it — a token file someone else can read is fixed by changing
permissions, a mismatched `state` is fixed by starting the flow again and not
clicking an old link — so the failures are distinguished here rather than
flattened into one message.
"""

from __future__ import annotations

__all__ = ["AuthError", "CallbackError", "TokenStoreError"]


class AuthError(Exception):
    """Base for every failure in the connect ceremony."""


class TokenStoreError(AuthError):
    """The local token store could not be read, written, or trusted."""


class CallbackError(AuthError):
    """The browser's redirect did not carry an authorization Tick can use."""
