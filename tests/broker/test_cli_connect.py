"""The connect and broker-profile ceremonies, driven end to end in memory.

The broker session and OAuth listener are injected in-memory fakes. Everything
else runs for real — disclosure ordering, refusal to print a token, proposal,
per-tool confirmation, proof, drift status, and private file modes.

The fake opener drives the loopback object's own `redirect_handler`, so ordering
assertion is about the real ceremony: the disclosure must appear before the
authorization URL, not merely somewhere in the output.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from mcp.server.mcpserver import MCPServer
from typer.testing import CliRunner

from tests.test_product_constraints import REAL_TICKERS
from tick import cli
from tick.auth import DISCLOSURE_LINES, FileTokenStorage
from tick.broker import Category, MCPSession, load_profile
from tick.broker.mock_mcp import MockBrokerage

from .conftest import TIMEOUT, memory_opener

runner = CliRunner()

AUTH_URL = "https://agent.robinhood.com/oauth/trading/authorize?state=state-placeholder"

ACCOUNT_OPTION = "agentic-0001"


@pytest.fixture(autouse=True)
def local_broker(monkeypatch: pytest.MonkeyPatch, mock_server: MCPServer):
    """Point the CLI's one network seam at the in-memory mock."""

    def fake_open(server_url, storage, loopback, timeout_seconds):
        import asyncio

        # The real provider announces the authorization URL through this
        # handler on the first 401, so the fake does the same: the ordering
        # test is then about the ceremony, not about the fake.
        asyncio.run(loopback.redirect_handler(AUTH_URL))
        return MCPSession(memory_opener(mock_server), timeout_seconds=TIMEOUT)

    monkeypatch.setattr(cli, "_open_robinhood_session", fake_open)
    return fake_open


@pytest.fixture
def connected(home: Path) -> FileTokenStorage:
    """A machine that has already been through the ceremony."""
    storage = FileTokenStorage(home)
    from tick.records import write_private_file

    write_private_file(storage.token_path, json.dumps({"access_token": "secret-placeholder"}))
    return storage


def invoke(*args: str, **kwargs):
    return runner.invoke(cli.app, list(args), **kwargs)


# ----------------------------------------------------------------------
# connect
# ----------------------------------------------------------------------


def test_the_disclosure_is_printed_before_the_authorization_url():
    """The audit's requirement, asserted as an ordering and not as a presence."""
    result = invoke("connect", "robinhood", "--yes", "--no-open-browser")

    assert result.exit_code == 0, result.output
    for line in DISCLOSURE_LINES:
        assert line in result.output
    assert result.output.index(DISCLOSURE_LINES[1]) < result.output.index(AUTH_URL)


def test_connecting_reports_success_without_printing_a_token(home: Path):
    result = invoke("connect", "robinhood", "--yes", "--no-open-browser")

    assert "Connected to mock-brokerage." in result.output
    assert "6 tool(s) discovered" in result.output
    assert "access_token" not in result.output
    assert "secret" not in result.output


def test_declining_the_disclosure_authorises_nothing(home: Path):
    """A ceremony you can walk out of. Nothing is written and it says so."""
    result = invoke("connect", "robinhood", "--no-open-browser", input="n\n")

    assert result.exit_code == 1
    assert "Nothing was authorised and nothing was written." in result.output
    assert not FileTokenStorage(home).connected()


def test_the_disclosure_names_the_wider_grant_and_ticks_own_narrowing():
    result = invoke("connect", "robinhood", "--yes", "--no-open-browser")

    lowered = result.output.lower()
    assert "all of your robinhood accounts" in lowered
    assert "tick reads only the one agentic account you configure" in lowered


# ----------------------------------------------------------------------
# disconnect
# ----------------------------------------------------------------------


def test_disconnect_removes_the_local_copy_and_says_it_is_only_local(
    connected: FileTokenStorage,
):
    result = invoke("disconnect", "robinhood")

    assert "removed" in result.output
    assert "Revoke the grant at Robinhood too" in result.output
    assert not connected.connected()


def test_disconnect_on_a_machine_with_no_grant_is_not_an_error(home: Path):
    result = invoke("disconnect", "robinhood")

    assert result.exit_code == 0
    assert "nothing to remove" in result.output


# ----------------------------------------------------------------------
# broker tools / profile ceremony
# ----------------------------------------------------------------------


def test_the_broker_commands_refuse_before_a_connection_exists(home: Path):
    result = invoke("broker", "tools")

    assert result.exit_code == 1
    assert "tick connect robinhood" in result.output


def test_tools_lists_what_the_broker_declares(connected: FileTokenStorage):
    """Invariant 7 made visible: what it really offers, not what Tick expected."""
    result = invoke("broker", "tools")

    assert result.exit_code == 0, result.output
    for name in ("get_quote", "get_positions", "place_order", "cancel_order", "list_orders"):
        assert name in result.output
    assert "account_id: string (required)" in result.output
    assert "6 tool(s)" in result.output


def test_status_says_what_is_connected_and_what_refuses(
    connected: FileTokenStorage, home: Path, brokerage: MockBrokerage
):
    invoke("broker", "propose", "--account", brokerage.agentic_account)
    invoke("broker", "confirm", "--yes-all-reads")

    result = invoke("broker", "status")

    assert "grant: stored" in result.output
    assert f"account: {brokerage.agentic_account}" in result.output
    assert "unmapped (these refuse): cancel_order, place_order" in result.output


def test_status_on_a_bare_machine_says_what_to_run(home: Path):
    result = invoke("broker", "status")

    assert "grant: none" in result.output
    assert "tick broker propose --account" in result.output


def test_profile_ceremony_never_bulk_confirms_an_order(connected: FileTokenStorage, home: Path):
    proposed = invoke("broker", "propose", "--account", ACCOUNT_OPTION)
    assert proposed.exit_code == 0, proposed.output
    assert "Nothing is callable yet" in proposed.output

    reads = invoke("broker", "confirm", "--yes-all-reads")
    assert reads.exit_code == 0, reads.output
    profile = load_profile(home)
    assert profile is not None
    with pytest.raises(Exception, match="order.place"):
        profile.mapping_for(Category.ORDER_PLACE)

    order = invoke("broker", "confirm", "--tool", "place_order", input="y\n")
    assert order.exit_code == 0, order.output
    profile = load_profile(home)
    assert profile is not None
    assert profile.mapping_for(Category.ORDER_PLACE).confirmed_by == "terminal"


def test_prove_reports_each_resolved_path_and_writes_a_note(
    connected: FileTokenStorage, home: Path
):
    invoke("broker", "propose", "--account", ACCOUNT_OPTION)
    invoke("broker", "confirm", "--yes-all-reads")

    result = invoke("broker", "prove", "--probe", "symbol=XYZ")

    assert result.exit_code == 0, result.output
    assert "get_quote: proved" in result.output
    assert "price: resolved" in result.output
    assert "profile_proven" in result.output


def test_prove_reports_unresolved_paths_as_unavailable(
    connected: FileTokenStorage,
):
    invoke("broker", "propose", "--account", ACCOUNT_OPTION)
    invoke("broker", "confirm", "--yes-all-reads")

    result = invoke("broker", "prove", "--probe", "symbol=UNKNOWN")

    assert result.exit_code == 0, result.output
    assert "get_quote: UNRESOLVED" in result.output
    assert "Unavailable" in result.output
    assert "price" in result.output


def test_a_community_server_requires_unsanctioned_and_is_recorded(
    connected: FileTokenStorage, home: Path
):
    server = "https://community.invalid/mcp"
    refused = invoke("broker", "propose", "--account", ACCOUNT_OPTION, "--server-url", server)
    assert refused.exit_code == 2
    assert "--unsanctioned" in refused.output

    proposed = invoke(
        "broker",
        "propose",
        "--account",
        ACCOUNT_OPTION,
        "--server-url",
        server,
        "--unsanctioned",
    )
    assert proposed.exit_code == 0, proposed.output
    confirmed = invoke("broker", "confirm", "--yes-all-reads", "--unsanctioned")
    assert confirmed.exit_code == 0, confirmed.output
    profile = load_profile(home)
    assert profile is not None
    assert profile.sanction == "community"


# ----------------------------------------------------------------------
# The product rules the CLI carries
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        ["connect"],
        ["connect", "robinhood"],
        ["disconnect", "robinhood"],
        ["broker"],
        ["broker", "tools"],
        ["broker", "propose"],
        ["broker", "confirm"],
        ["broker", "prove"],
        ["broker", "status"],
    ],
    ids=lambda p: " ".join(p),
)
def test_no_new_help_text_names_a_security_or_calls_a_rule_agent_an_ai(path: list[str]):
    """Tick authors no strategies, and a deterministic agent is a rule agent."""
    output = invoke(*path, "--help").output

    assert "ai agent" not in output.lower()
    assert REAL_TICKERS.findall(output) == []
