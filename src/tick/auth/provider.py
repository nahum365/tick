"""Assembling the OAuth client: what Tick registers as, and where it points.

Robinhood's Trading MCP publishes, in its protected-resource metadata,
`authorization_code` + `refresh_token` grants, PKCE with `S256`, open dynamic
client registration, a single scope (`internal`), and
`token_endpoint_auth_methods: ["none"]`. Everything in this file follows from
that: Tick registers itself as a **native public client** whose redirect URI
is a loopback address, holds no client secret, and asks for the one scope that
exists.

The registration is per machine. Nothing here is provisioned by Tick, held by
Tick, or shared between users — which is the same property invariant 1 states
about the token, applied to the client identity that redeems it. The consent
screen therefore shows whatever client name Robinhood's server assigns the
registration, which is why `auth/disclosure.py` warns the user in advance.

The server URL is a parameter with a named default rather than a literal
buried in a call, so a reviewer can see the one host this package talks to and
so a test can point the whole ceremony at a local mock without monkeypatching.
"""

from __future__ import annotations

from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.shared.auth import OAuthClientMetadata

from .loopback import LoopbackAuthorization

__all__ = [
    "ROBINHOOD_MCP_URL",
    "ROBINHOOD_SCOPE",
    "TICK_CLIENT_NAME",
    "build_oauth_provider",
    "client_metadata",
]

#: Robinhood's official Trading MCP — the one host this package speaks to.
ROBINHOOD_MCP_URL = "https://agent.robinhood.com/mcp/trading"

#: The only scope their metadata advertises.
ROBINHOOD_SCOPE = "internal"

#: What Tick asks to be registered as. The authorization server may assign a
#: different display name, and the disclosure says so.
TICK_CLIENT_NAME = "Tick (local agent runtime)"


def client_metadata(redirect_uri: str) -> OAuthClientMetadata:
    """The dynamic-registration document for a native public client.

    `token_endpoint_auth_method="none"` is not a weakening: it is the shape
    Robinhood's metadata advertises, and it is correct for a client that runs
    on the user's own machine, where a "secret" shipped in software is not one.
    PKCE is what binds the code to this client, and the SDK sends `S256`.
    """
    return OAuthClientMetadata(
        client_name=TICK_CLIENT_NAME,
        redirect_uris=[redirect_uri],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        token_endpoint_auth_method="none",
        application_type="native",
        scope=ROBINHOOD_SCOPE,
    )


def build_oauth_provider(
    *,
    server_url: str,
    storage: TokenStorage,
    loopback: LoopbackAuthorization,
) -> OAuthClientProvider:
    """The provider that authorises requests to `server_url`.

    Every argument is required and named. In particular `storage` is injected
    rather than constructed here: the file that holds the token is the whole
    of invariant 1, and a provider that could quietly pick its own store would
    be a second place credentials can land.
    """
    return OAuthClientProvider(
        server_url,
        client_metadata=client_metadata(loopback.redirect_uri),
        storage=storage,
        redirect_handler=loopback.redirect_handler,
        callback_handler=loopback.callback_handler,
    )
