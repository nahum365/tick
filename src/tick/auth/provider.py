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
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken

from .loopback import LoopbackAuthorization

__all__ = [
    "RedirectBoundStorage",
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
        storage=RedirectBoundStorage(storage, redirect_uri=loopback.redirect_uri),
        redirect_handler=loopback.redirect_handler,
        callback_handler=loopback.callback_handler,
    )


class RedirectBoundStorage:
    """The injected store, with the registered client bound to this run's redirect.

    A dynamically registered client carries the exact redirect URI it was
    registered with. The loopback listener takes a fresh port each run, so a
    client registered on an earlier port would authorize and then fail the code
    exchange with a redirect mismatch the person cannot see. When the stored
    client was registered for a different redirect, it reads as absent and the
    provider registers again. Tokens are untouched: they pass straight through
    to the one store, which stays the only place credentials land.
    """

    def __init__(self, storage: TokenStorage, *, redirect_uri: str) -> None:
        self.storage = storage
        self._redirect_uri = redirect_uri

    async def get_tokens(self) -> OAuthToken | None:
        return await self.storage.get_tokens()

    async def set_tokens(self, tokens: OAuthToken) -> None:
        await self.storage.set_tokens(tokens)

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        info = await self.storage.get_client_info()
        if info is None:
            return None
        registered = [str(uri) for uri in info.redirect_uris or []]
        if self._redirect_uri not in registered:
            return None
        return info

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        await self.storage.set_client_info(client_info)
