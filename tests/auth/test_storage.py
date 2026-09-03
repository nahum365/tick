"""The token store: private on disk, and the only copy of the credential.

CLAUDE.md invariant 1 says credentials never leave the user's box. On a
multi-user machine "the box" is not fine enough — a token file the `staff`
group can read has left the user, whatever it says in the terms — so the mode
is asserted here on every file the store writes, on creation and on rewrite.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest
from mcp.client.auth import TokenStorage
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

from tick.auth import (
    CLIENT_FILE,
    TOKEN_FILE,
    TOKEN_FILE_MODE,
    FileTokenStorage,
    TokenStoreError,
)


def a_token(access: str = "access-placeholder") -> OAuthToken:
    return OAuthToken(access_token=access, refresh_token="refresh-placeholder", expires_in=3600)


def a_client() -> OAuthClientInformationFull:
    return OAuthClientInformationFull(
        client_id="client-placeholder",
        redirect_uris=["http://127.0.0.1:8765/tick/callback"],
    )


def mode_of(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


async def test_the_token_file_is_created_readable_by_nobody_else(home: Path):
    """Mode 0600, from the moment the file exists."""
    storage = FileTokenStorage(home)
    await storage.set_tokens(a_token())

    assert storage.token_path == home / "robinhood" / TOKEN_FILE
    assert mode_of(storage.token_path) == TOKEN_FILE_MODE


async def test_the_registered_client_file_is_private_too(home: Path):
    """Registration is per machine, so the client identity is as local as the token."""
    storage = FileTokenStorage(home)
    await storage.set_client_info(a_client())

    assert storage.client_path == home / "robinhood" / CLIENT_FILE
    assert mode_of(storage.client_path) == TOKEN_FILE_MODE


async def test_a_rewrite_reasserts_the_mode(home: Path):
    """A refresh rewrites this file; the second write must be as private as the first."""
    storage = FileTokenStorage(home)
    await storage.set_tokens(a_token("first"))
    storage.token_path.chmod(0o644)

    await storage.set_tokens(a_token("second"))

    assert mode_of(storage.token_path) == TOKEN_FILE_MODE
    assert json.loads(storage.token_path.read_text())["access_token"] == "second"


async def test_the_directory_is_private(home: Path):
    """The list of what is connected is not public either."""
    storage = FileTokenStorage(home)
    await storage.set_tokens(a_token())

    assert mode_of(storage.directory) == 0o700


async def test_a_stored_grant_round_trips(home: Path):
    storage = FileTokenStorage(home)
    await storage.set_tokens(a_token())
    await storage.set_client_info(a_client())

    tokens = await storage.get_tokens()
    client = await storage.get_client_info()

    assert tokens is not None and tokens.access_token == "access-placeholder"
    assert tokens.refresh_token == "refresh-placeholder"
    assert client is not None and client.client_id == "client-placeholder"


async def test_nothing_stored_reads_as_nothing_not_as_an_error(home: Path):
    """A machine that has never connected is a normal state, not a failure."""
    storage = FileTokenStorage(home)

    assert await storage.get_tokens() is None
    assert await storage.get_client_info() is None
    assert storage.connected() is False


async def test_a_widened_token_file_is_refused_rather_than_used(home: Path):
    """A credential the machine has already shown to someone else is not used."""
    storage = FileTokenStorage(home)
    await storage.set_tokens(a_token())
    storage.token_path.chmod(0o644)

    with pytest.raises(TokenStoreError) as caught:
        await storage.get_tokens()

    assert "0644" in str(caught.value)
    assert "chmod 600" in str(caught.value)


async def test_an_unreadable_token_file_refuses_and_says_what_to_do(home: Path):
    storage = FileTokenStorage(home)
    await storage.set_tokens(a_token())
    storage.token_path.write_text("{not json", encoding="utf-8")
    storage.token_path.chmod(TOKEN_FILE_MODE)

    with pytest.raises(TokenStoreError) as caught:
        await storage.get_tokens()

    assert "tick connect robinhood" in str(caught.value)


async def test_forget_deletes_every_local_copy(home: Path):
    """The user can end the grant locally without editing files by hand."""
    storage = FileTokenStorage(home)
    await storage.set_tokens(a_token())
    await storage.set_client_info(a_client())

    removed = storage.forget()

    assert sorted(path.name for path in removed) == [CLIENT_FILE, TOKEN_FILE]
    assert storage.connected() is False
    assert storage.forget() == []


def test_the_store_satisfies_the_sdks_token_storage_protocol(home: Path):
    """The SDK's protocol is what the OAuth provider calls; this is that shape."""
    storage = FileTokenStorage(home)
    for method in TokenStorage.__protocol_attrs__:
        assert callable(getattr(storage, method)), method


async def test_the_store_writes_nowhere_but_its_own_directory(home: Path):
    """One directory holds the credential, so one directory is what a reader checks."""
    storage = FileTokenStorage(home)
    await storage.set_tokens(a_token())
    await storage.set_client_info(a_client())

    written = sorted(str(path.relative_to(home)) for path in home.rglob("*") if path.is_file())
    assert written == [f"robinhood/{CLIENT_FILE}", f"robinhood/{TOKEN_FILE}"]
