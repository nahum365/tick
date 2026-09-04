"""What Tick registers as, and where the flow points. No request is made here.

Building an `OAuthClientProvider` opens nothing: the flow starts on the first
401 from the resource server, and no resource server is contacted in this file.
What is asserted is the shape of the registration — public client, loopback
redirect, the one scope Robinhood's metadata advertises — and that the token
store the provider was handed is the local one and not something it chose.
"""

from __future__ import annotations

from pathlib import Path

from mcp.client.auth import OAuthClientProvider

from tick.auth import (
    ROBINHOOD_MCP_URL,
    ROBINHOOD_SCOPE,
    FileTokenStorage,
    LoopbackAuthorization,
    build_oauth_provider,
    client_metadata,
)


def a_loopback() -> LoopbackAuthorization:
    return LoopbackAuthorization(
        port=0,
        timeout_seconds=5.0,
        open_browser=False,
        announce=lambda line: None,
        redirect_uri_override=None,
    )


def test_the_registration_is_a_native_public_client_on_a_loopback_redirect():
    """`token_endpoint_auth_method: none` is the shape their metadata advertises."""
    metadata = client_metadata("http://127.0.0.1:8765/tick/callback")

    assert metadata.token_endpoint_auth_method == "none"
    assert metadata.application_type == "native"
    assert metadata.response_types == ["code"]
    assert sorted(metadata.grant_types) == ["authorization_code", "refresh_token"]
    assert metadata.scope == ROBINHOOD_SCOPE
    assert [str(uri) for uri in metadata.redirect_uris or []] == [
        "http://127.0.0.1:8765/tick/callback"
    ]


def test_the_only_remote_host_named_is_robinhoods_own():
    assert ROBINHOOD_MCP_URL.startswith("https://agent.robinhood.com/")


def test_the_provider_is_built_over_the_local_token_store(home: Path):
    """Invariant 1: the store is injected, so there is exactly one place tokens land."""
    storage = FileTokenStorage(home)
    with a_loopback() as loopback:
        provider = build_oauth_provider(
            server_url=ROBINHOOD_MCP_URL, storage=storage, loopback=loopback
        )

        assert isinstance(provider, OAuthClientProvider)
        assert provider.context.storage is storage
        assert [str(uri) for uri in provider.context.client_metadata.redirect_uris or []] == [
            loopback.redirect_uri
        ]


def test_building_a_provider_contacts_nothing(home: Path):
    """The flow begins on a 401, so construction is inert — and stays that way."""
    storage = FileTokenStorage(home)
    with a_loopback() as loopback:
        build_oauth_provider(server_url=ROBINHOOD_MCP_URL, storage=storage, loopback=loopback)

    assert not storage.connected()
    assert list(home.rglob("*")) == [] or not storage.token_path.exists()
