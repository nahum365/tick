"""OAuth for Robinhood's Trading MCP — on the user's machine, and nowhere else.

    from tick.auth import FileTokenStorage, LoopbackAuthorization, build_oauth_provider

    storage = FileTokenStorage(tick_home(os.environ))
    with LoopbackAuthorization(
        port=0, timeout_seconds=300.0, open_browser=True, announce=print,
        redirect_uri_override=None, on_callback=None
    ) as loopback:
        provider = build_oauth_provider(
            server_url=ROBINHOOD_MCP_URL, storage=storage, loopback=loopback
        )

This package is the whole of CLAUDE.md invariant 1. The grant is obtained by a
browser the user drives, the code comes back to `127.0.0.1`, the token is
written to `TICK_HOME/robinhood/` mode 0600, and there is no Tick-operated
endpoint in the flow at any point — the only remote host named anywhere here
is Robinhood's own. `tests/test_product_constraints.py` scans for exactly that
rather than trusting this paragraph.

The disclosure in `disclosure.py` is part of the ceremony, not documentation:
Robinhood's grant reads *every* account the user has, Tick narrows itself to
one, and the user is told both before the browser opens.
"""

from __future__ import annotations

from .disclosure import DISCLOSURE_LINES, disclosure_text
from .errors import AuthError, CallbackError, TokenStoreError
from .loopback import CALLBACK_PATH, LOOPBACK_HOST, LoopbackAuthorization, state_of
from .provider import (
    ROBINHOOD_MCP_URL,
    ROBINHOOD_SCOPE,
    TICK_CLIENT_NAME,
    build_oauth_provider,
    client_metadata,
)
from .storage import (
    CLIENT_FILE,
    ROBINHOOD_DIR,
    TOKEN_FILE,
    TOKEN_FILE_MODE,
    FileTokenStorage,
    robinhood_dir,
)

__all__ = [
    "AuthError",
    "CALLBACK_PATH",
    "CLIENT_FILE",
    "CallbackError",
    "DISCLOSURE_LINES",
    "FileTokenStorage",
    "LOOPBACK_HOST",
    "LoopbackAuthorization",
    "ROBINHOOD_DIR",
    "ROBINHOOD_MCP_URL",
    "ROBINHOOD_SCOPE",
    "TICK_CLIENT_NAME",
    "TOKEN_FILE",
    "TOKEN_FILE_MODE",
    "TokenStoreError",
    "build_oauth_provider",
    "client_metadata",
    "disclosure_text",
    "robinhood_dir",
    "state_of",
]
