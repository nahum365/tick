"""`tick run --live` — what it refuses, what it records, and what it stops for.

Only one thing is replaced: `_open_robinhood_session`, the single seam in the
CLI where a socket could be opened, which returns a session over the in-memory
mock brokerage instead. Everything else runs for real — the readiness check
against the files on disk, the warning, the mode-change record, the standing
approval gate, the adapter's read scoping, the cage, and the record.

The mock brokerage is not a model of Robinhood's real tools (invariant 7 says
their shapes are unverified). It is a plausible brokerage that lets the wiring
be exercised without a socket, a token, or an account.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from mcp.server.mcpserver import MCPServer
from typer.testing import CliRunner

from tests.test_product_constraints import REAL_TICKERS
from tick import cli
from tick.agents import EMIT_TOOL_NAME, read_model_reply
from tick.broker import (
    BrokerUnavailable,
    Category,
    DriftDifference,
    MCPSession,
    ProfileState,
    ProfileTool,
    confirm_profile,
    inventory_hash,
    load_profile,
    profile_path,
    propose_profile,
    save_profile,
)
from tick.broker.mock_mcp import MockBrokerage
from tick.broker.profile import (
    CANONICALIZER_VERSION,
    CATEGORIZER_VERSION,
    CATEGORY_REGISTRY_VERSION,
    PROFILE_FORMAT_VERSION,
    ProofResult,
    build_profile,
    mapping_hash,
)
from tick.records import RecordKind, read, write_private_file
from tick.runtime import AgentRun, Mode

from .conftest import TIMEOUT, memory_opener

runner = CliRunner()

ACCOUNT = "agentic-0001"


# ----------------------------------------------------------------------
# The two agent documents these tests run
# ----------------------------------------------------------------------

CAGE: dict[str, Any] = {
    "max_position_pct": "100.00",
    "max_positions": 5,
    "max_order_notional": "1000000.00",
    "max_daily_drawdown_pct": "50.00",
    "allowed_session": "regular_hours",
}

#: A rule an account's own cash decides, so it needs no price history — the
#: trading connection maps no history capability and honestly says so.
RULE_DOCUMENT: dict[str, Any] = {
    "name": "Live rule agent",
    "version": 1,
    "universe": ["XYZ"],
    "cadence": {"kind": "daily_close"},
    "rules": [
        {
            "id": "buy-one",
            "when": {
                "kind": "compare",
                "left": {"kind": "cash"},
                "op": ">",
                "right": {"kind": "number", "value": "0"},
            },
            "then": {
                "side": "buy",
                "size": {"kind": "shares", "shares": 1},
                "order_type": "market",
            },
        }
    ],
    "cage": CAGE,
}

MODEL_DOCUMENT: dict[str, Any] = {
    "kind": "model_agent",
    "name": "Live model agent",
    "version": 1,
    "universe": ["XYZ"],
    "cadence": {"kind": "daily_close"},
    "provider": "anthropic",
    "model": "claude-opus-5",
    "cage": CAGE,
}

INSTRUCTIONS = "My own words. Buy one XYZ when I say so.\n"

#: A moment inside the 2026-09-01 session, in ET.
AT = "2026-09-01T11:00:00-04:00"

#: The placeholder market series, for the one test that runs the same agent in
#: paper afterwards. Absolute, so the test does not depend on the caller's cwd.
MARKET_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "market"


def invoke(*args: str, **kwargs):
    return runner.invoke(cli.app, list(args), **kwargs)


@pytest.fixture
def granted(home: Path):
    """A machine that has been through the connect ceremony."""
    write_private_file(
        home / "robinhood" / "tokens.json", json.dumps({"access_token": "secret-placeholder"})
    )
    return home


@pytest.fixture
def mapped(granted: Path, mock_server: MCPServer, brokerage: MockBrokerage) -> Path:
    """…and has confirmed and proven a profile for its account."""
    session = MCPSession(memory_opener(mock_server), timeout_seconds=TIMEOUT)
    session.open()
    try:
        proposal = propose_profile(
            session.list_tools(),
            server="https://agent.robinhood.com/mcp/trading",
            account_id=brokerage.agentic_account,
            proposed_at=datetime(2026, 9, 1, tzinfo=UTC),
        )
    finally:
        session.close()
    tools = {}
    for name, proposed in proposal.tools.items():
        assert proposed.category is not None
        category = proposed.category
        digest = mapping_hash(category, proposed.arguments, proposed.result)
        proof = ProofResult(
            success=True,
            resolved=tuple(proposed.result),
            unresolved={},
            detail="fixture proof",
        )
        tools[name] = ProfileTool(
            category=category,
            contract=proposed.contract,
            arguments=proposed.arguments,
            result=proposed.result,
            confirmed_contract_hash=proposed.contract.contract_hash,
            mapping_hash=digest,
            confirmed_at=datetime(2026, 9, 1, tzinfo=UTC),
            confirmed_by="terminal",
            categorizer_version=CATEGORIZER_VERSION,
            proved_contract_hash=proposed.contract.contract_hash,
            proved_mapping_hash=digest,
            proved_at=datetime(2026, 9, 1, tzinfo=UTC),
            proof=proof,
        )
    profile = build_profile(
        server=proposal.server,
        account_id=proposal.account_id,
        tools=tools,
        inventory_hash=inventory_hash(tuple(tool.contract for tool in tools.values())),
        data_class="display_only",
        sanction="official",
        profile_format_version=PROFILE_FORMAT_VERSION,
        canonicalizer_version=CANONICALIZER_VERSION,
        category_registry_version=CATEGORY_REGISTRY_VERSION,
        state=ProfileState.CONFIRMED,
        observed_inventory_hash=proposal.inventory_hash,
        drift=(),
    )
    confirm_profile(
        granted,
        profile,
        actor="test-terminal",
        at=datetime(2026, 9, 1, tzinfo=UTC),
    )
    return granted


@pytest.fixture
def live_session(monkeypatch: pytest.MonkeyPatch, mock_server: MCPServer):
    """Point the CLI's one network seam at the in-memory mock."""
    opened: list[MCPSession] = []

    def fake_open(server_url, storage, loopback, timeout_seconds):
        session = MCPSession(memory_opener(mock_server), timeout_seconds=TIMEOUT)
        opened.append(session)
        return session

    monkeypatch.setattr(cli, "_open_robinhood_session", fake_open)
    return opened


def add_rule_agent(tmp_path: Path, *extra: str) -> str:
    # These tests predate the per-order default (2026-09-02) and exercise the
    # standing path deliberately; a test that wants the default passes nothing.
    if "--approve" not in extra:
        extra = ("--approve", "standing", *extra)
    path = tmp_path / "rule.json"
    path.write_text(json.dumps(RULE_DOCUMENT), encoding="utf-8")
    result = invoke("agent", "add", str(path), "--max-cancels", "2", *extra)
    assert result.exit_code == 0, result.output
    return result.output.splitlines()[0].strip()


def add_model_agent(tmp_path: Path, *extra: str) -> str:
    # These tests predate the per-order default (2026-09-02) and exercise the
    # standing path deliberately; a test that wants the default passes nothing.
    if "--approve" not in extra:
        extra = ("--approve", "standing", *extra)
    document = tmp_path / "model.json"
    document.write_text(json.dumps(MODEL_DOCUMENT), encoding="utf-8")
    words = tmp_path / "words.md"
    words.write_text(INSTRUCTIONS, encoding="utf-8")
    result = invoke(
        "agent",
        "add",
        str(document),
        "--max-cancels",
        "2",
        "--instructions",
        str(words),
        *extra,
    )
    assert result.exit_code == 0, result.output
    return result.output.splitlines()[0].strip()


def run_live(agent_id: str, *extra: str, **kwargs):
    return invoke("run", agent_id, "--once", "--live", "--at", AT, *extra, **kwargs)


def records(home: Path, agent_id: str):
    return list(read(AgentRun(home, agent_id).ledger_path))


def install_model(monkeypatch: pytest.MonkeyPatch, intents: list[Any]) -> list[Any]:
    """Answer the model agent from a fake client. Nothing here reaches a provider."""
    requests: list[Any] = []

    class FakeClient:
        def propose(self, request):
            requests.append(request)
            return read_model_reply(
                SimpleNamespace(
                    model="claude-opus-5-20260401",
                    stop_reason="tool_use",
                    stop_details=None,
                    content=[
                        SimpleNamespace(
                            type="tool_use", name=EMIT_TOOL_NAME, input={"intents": intents}
                        )
                    ],
                )
            )

    monkeypatch.setattr(cli, "client_for", lambda provider: FakeClient())
    return requests


# ----------------------------------------------------------------------
# What live refuses, before anything opens
# ----------------------------------------------------------------------


def test_live_refuses_on_a_machine_with_no_grant_and_names_the_command(
    home: Path, tmp_path: Path, live_session: list[MCPSession]
):
    agent_id = add_rule_agent(tmp_path)
    result = run_live(agent_id, "--live-standing-ok")

    assert result.exit_code == 2
    assert "holds no Robinhood grant" in result.output
    assert "Run: tick connect robinhood" in result.output
    assert live_session == []  # nothing was opened
    assert AgentRun(home, agent_id).state.mode is Mode.PAPER


def test_live_refuses_without_a_confirmed_profile_and_names_the_commands(
    granted: Path, tmp_path: Path, live_session: list[MCPSession]
):
    agent_id = add_rule_agent(tmp_path)
    result = run_live(agent_id, "--live-standing-ok")

    assert result.exit_code == 2
    assert "no broker profile has been confirmed" in result.output
    assert "tick broker propose --account" in result.output
    assert live_session == []


def test_live_refuses_a_profile_whose_account_cannot_be_read(
    granted: Path, tmp_path: Path, live_session: list[MCPSession]
):
    """A map with no account is not a map: every read is scoped to that id."""
    write_private_file(
        profile_path(granted),
        json.dumps(
            {
                "account_id": "  ",
                "server": "https://agent.robinhood.com/mcp/trading",
                "tools": {},
            }
        ),
    )
    agent_id = add_rule_agent(tmp_path)
    result = run_live(agent_id, "--live-standing-ok")

    assert result.exit_code == 2
    assert "could not be read" in result.output
    assert "tick broker propose --account" in result.output
    assert live_session == []


def test_live_refuses_a_profile_that_cannot_place_an_order(
    mapped: Path, tmp_path: Path, live_session: list[MCPSession], brokerage: MockBrokerage
):
    """Finding out at 09:31, with a fired rule in hand, is the wrong time."""
    full = load_profile(mapped)
    assert full is not None
    without_place = build_profile(
        server=full.server,
        account_id=full.account_id,
        tools={
            name: tool
            for name, tool in full.tools.items()
            if tool.category is not Category.ORDER_PLACE
        },
        inventory_hash=full.inventory_hash,
        data_class=full.data_class,
        sanction=full.sanction,
        profile_format_version=full.profile_format_version,
        canonicalizer_version=full.canonicalizer_version,
        category_registry_version=full.category_registry_version,
        state=ProfileState.CONFIRMED,
        observed_inventory_hash=full.observed_inventory_hash,
        drift=(),
    )
    confirm_profile(
        mapped,
        without_place,
        actor="test-terminal",
        at=datetime(2026, 9, 1, tzinfo=UTC),
    )
    agent_id = add_rule_agent(tmp_path)
    result = run_live(agent_id, "--live-standing-ok")

    assert result.exit_code == 2
    assert "depends on order.place" in result.output
    assert live_session == []


def test_the_next_agent_run_records_a_stored_inventory_difference_once(
    mapped: Path, tmp_path: Path
):
    stored = load_profile(mapped)
    assert stored is not None
    observed = build_profile(
        server=stored.server,
        account_id=stored.account_id,
        tools=stored.tools,
        inventory_hash=stored.inventory_hash,
        data_class=stored.data_class,
        sanction=stored.sanction,
        profile_format_version=stored.profile_format_version,
        canonicalizer_version=stored.canonicalizer_version,
        category_registry_version=stored.category_registry_version,
        state=ProfileState.DRIFTED,
        observed_inventory_hash="sha256:" + "1" * 64,
        drift=(DriftDifference(tool="get_quote", changes=("changed description",)),),
    )
    save_profile(mapped, observed)
    agent = AgentRun(mapped, add_rule_agent(tmp_path))
    at = datetime(2026, 9, 2, tzinfo=UTC)

    cli._record_pending_profile_observation(agent, now=at)
    cli._record_pending_profile_observation(agent, now=at)

    notes = [
        record
        for record in read(agent.ledger_path)
        if record.kind is RecordKind.NOTE
        and record.payload.get("event") == "broker_profile_drifted"
    ]
    assert len(notes) == 1
    assert notes[0].payload["inventory_hash"] == observed.observed_inventory_hash
    assert "changed description" in repr(notes[0].payload["diff"])


def test_a_stopped_agent_never_connects(mapped: Path, tmp_path: Path, live_session):
    """The kill switch reaches further back than the tick loop."""
    agent_id = add_rule_agent(tmp_path)
    invoke("stop", agent_id, "--reason", "not today")

    result = run_live(agent_id, "--live-standing-ok")

    assert result.exit_code == 1
    assert "is stopped: not today" in result.output
    assert live_session == []
    assert AgentRun(mapped, agent_id).state.mode is Mode.PAPER


# ----------------------------------------------------------------------
# Standing approval takes a second flag
# ----------------------------------------------------------------------


def test_standing_approval_live_needs_a_second_explicit_flag(
    mapped: Path, tmp_path: Path, live_session
):
    agent_id = add_rule_agent(tmp_path, "--approve", "standing")
    result = run_live(agent_id)

    assert result.exit_code == 2
    assert "--live-standing-ok" in result.output
    assert live_session == []
    assert AgentRun(mapped, agent_id).state.mode is Mode.PAPER


def test_the_standing_acknowledgement_is_recorded(mapped: Path, tmp_path: Path, live_session):
    agent_id = add_rule_agent(tmp_path, "--approve", "standing")
    result = run_live(agent_id, "--live-standing-ok")

    assert result.exit_code == 0, result.output
    armed = next(r for r in records(mapped, agent_id) if r.payload.get("event") == "live_armed")
    assert armed.payload["standing_approval_acknowledged"] is True
    assert armed.payload["approval"] == "standing"
    assert armed.payload["account_id"] == ACCOUNT


def test_per_order_approval_needs_no_second_flag_and_asks(
    mapped: Path, tmp_path: Path, live_session
):
    agent_id = add_rule_agent(tmp_path, "--approve", "each")
    result = run_live(agent_id, input="y\n")

    assert result.exit_code == 0, result.output
    assert "Place buy 1 XYZ at $184.20?" in result.output
    assert "you will be asked before every order" in result.output


def test_declining_a_live_order_places_nothing(
    mapped: Path, tmp_path: Path, live_session, brokerage: MockBrokerage
):
    agent_id = add_rule_agent(tmp_path, "--approve", "each")
    before = brokerage.held(ACCOUNT, "XYZ")
    result = run_live(agent_id, input="n\n")

    assert result.exit_code == 0, result.output
    assert brokerage.held(ACCOUNT, "XYZ") == before
    assert "you declined" in result.output


# ----------------------------------------------------------------------
# The flip: warned, recorded, and never inherited
# ----------------------------------------------------------------------


def test_the_flip_prints_a_plain_warning_before_anything_is_placed(
    mapped: Path, tmp_path: Path, live_session
):
    agent_id = add_rule_agent(tmp_path)
    result = run_live(agent_id, "--live-standing-ok")

    assert "LIVE MODE — real orders, real money." in result.output
    assert f"will trade account {ACCOUNT} and no other" in result.output
    assert f"tick stop {agent_id}" in result.output
    assert "long orders only" in result.output
    assert result.output.index("LIVE MODE") < result.output.index("fired: bought")


def test_the_flip_is_recorded_with_who_and_when(
    mapped: Path, tmp_path: Path, live_session, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("TICK_ACTOR", "the-person-at-this-terminal")
    agent_id = add_rule_agent(tmp_path)
    run_live(agent_id, "--live-standing-ok")

    change = next(r for r in records(mapped, agent_id) if r.kind is RecordKind.MODE_CHANGE)
    assert change.payload["from"] == "paper"
    assert change.payload["to"] == "live"
    assert change.payload["by"] == "the-person-at-this-terminal"
    assert change.payload["at"].startswith("2026-09-01T11:00:00")
    assert change.payload["account_id"] == ACCOUNT


def test_the_flip_is_recorded_before_the_first_order(mapped: Path, tmp_path: Path, live_session):
    agent_id = add_rule_agent(tmp_path)
    run_live(agent_id, "--live-standing-ok")

    kinds = [record.kind for record in records(mapped, agent_id)]
    assert kinds[0] is RecordKind.MODE_CHANGE
    assert kinds.index(RecordKind.MODE_CHANGE) < kinds.index(RecordKind.FILL)


def test_a_second_live_run_records_the_arming_but_not_a_second_flip(
    mapped: Path, tmp_path: Path, live_session
):
    agent_id = add_rule_agent(tmp_path)
    run_live(agent_id, "--live-standing-ok")
    run_live(agent_id, "--live-standing-ok")

    written = records(mapped, agent_id)
    assert [r.kind for r in written].count(RecordKind.MODE_CHANGE) == 1
    assert sum(1 for r in written if r.payload.get("event") == "live_armed") == 2


def test_a_run_without_the_flag_puts_a_live_agent_back_in_paper_and_records_it(
    mapped: Path, tmp_path: Path, live_session
):
    """Paper is the default on EVERY path: live is an act, not a setting."""
    agent_id = add_rule_agent(tmp_path)
    run_live(agent_id, "--live-standing-ok")
    assert AgentRun(mapped, agent_id).state.mode is Mode.LIVE

    result = invoke(
        "run",
        agent_id,
        "--once",
        "--market",
        f"fixture:{MARKET_FIXTURES}",
        "--paper-cash",
        "10000.00",
        "--at",
        AT,
    )

    assert result.exit_code == 0, result.output
    assert AgentRun(mapped, agent_id).state.mode is Mode.PAPER
    changes = [r for r in records(mapped, agent_id) if r.kind is RecordKind.MODE_CHANGE]
    assert [(c.payload["from"], c.payload["to"]) for c in changes] == [
        ("paper", "live"),
        ("live", "paper"),
    ]


# ----------------------------------------------------------------------
# The order actually reaches the broker, scoped to one account
# ----------------------------------------------------------------------


def test_a_live_rule_agent_places_through_the_broker_and_records_the_fill(
    mapped: Path, tmp_path: Path, live_session, brokerage: MockBrokerage
):
    agent_id = add_rule_agent(tmp_path)
    before = brokerage.held(ACCOUNT, "XYZ")

    result = run_live(agent_id, "--live-standing-ok")

    assert result.exit_code == 0, result.output
    assert brokerage.held(ACCOUNT, "XYZ") == before + 1
    assert "Your rule 'buy-one' fired: bought 1 XYZ at $184.20 — live." in result.output
    fill = next(r for r in records(mapped, agent_id) if r.kind is RecordKind.FILL)
    assert fill.payload["source"] == "robinhood"
    assert AgentRun(mapped, agent_id).verify_ledger().ok


def test_a_live_model_agent_places_what_the_cage_allows(
    mapped: Path, tmp_path: Path, live_session, brokerage: MockBrokerage, monkeypatch
):
    install_model(
        monkeypatch,
        [{"symbol": "XYZ", "side": "buy", "qty": 1, "reason": "it fits my instructions"}],
    )
    agent_id = add_model_agent(tmp_path)
    before = brokerage.held(ACCOUNT, "XYZ")

    result = run_live(agent_id, "--live-standing-ok")

    assert result.exit_code == 0, result.output
    assert brokerage.held(ACCOUNT, "XYZ") == before + 1
    assert "Your model agent (claude-opus-5) bought 1 XYZ at $184.20 — live." in result.output


def test_a_live_model_agents_out_of_universe_intent_never_reaches_the_broker(
    mapped: Path, tmp_path: Path, live_session, brokerage: MockBrokerage, monkeypatch
):
    install_model(
        monkeypatch, [{"symbol": "WXY", "side": "buy", "qty": 1, "reason": "outside the universe"}]
    )
    agent_id = add_model_agent(tmp_path)
    before = dict(brokerage.cash)

    result = run_live(agent_id, "--live-standing-ok")

    assert result.exit_code == 0, result.output
    assert brokerage.cash == before
    assert "proposed an order that was rejected" in result.output
    assert "place_order" not in brokerage.argument_text()


def test_a_live_run_reads_and_trades_one_account_and_no_other(
    mapped: Path, tmp_path: Path, live_session, brokerage: MockBrokerage
):
    """The grant reads every account; Tick asks about one and keeps one."""
    agent_id = add_rule_agent(tmp_path)
    run_live(agent_id, "--live-standing-ok")

    assert brokerage.other_account not in brokerage.argument_text()
    written = json.dumps([record.payload for record in records(mapped, agent_id)], default=str)
    assert brokerage.other_account not in written
    assert "WXY" not in written  # the other account's holding


def test_a_live_run_prices_from_the_broker_it_trades_through(
    mapped: Path, tmp_path: Path, live_session
):
    agent_id = add_rule_agent(tmp_path)
    run_live(agent_id, "--live-standing-ok")

    decision = next(r for r in records(mapped, agent_id) if r.kind is RecordKind.DECISION)
    assert decision.payload["source"] == "robinhood"
    assert decision.payload["prices"]["XYZ"] == "184.20"
    assert decision.payload["profile_hash"].startswith("sha256:")
    assert decision.payload["inventory_hash"].startswith("sha256:")
    assert decision.payload["profile_sanction"] == "official"
    assert decision.payload["data_class"] == "display_only"


# ----------------------------------------------------------------------
# Fail safe: the connection drops, the run stops, nothing is re-sent
# ----------------------------------------------------------------------


class FailingSession:
    """Wraps a real session and drops the connection after `after` tool calls."""

    def __init__(self, inner: MCPSession, *, after: int) -> None:
        self._inner = inner
        self._after = after
        self.calls = 0

    def open(self) -> None:
        self._inner.open()

    def close(self) -> None:
        self._inner.close()

    @property
    def server_name(self):
        return self._inner.server_name

    def list_tools(self):
        return self._inner.list_tools()

    def call_tool(self, name: str, arguments):
        self.calls += 1
        if self.calls > self._after:
            raise BrokerUnavailable(
                "the broker's MCP session failed mid-call. Tick stops rather than "
                "reconnecting; nothing was retried."
            )
        return self._inner.call_tool(name, arguments)


def test_an_mcp_failure_mid_run_stops_the_run_and_places_nothing_further(
    mapped: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_server: MCPServer,
    brokerage: MockBrokerage,
):
    """A dropped connection is a stop, never a retry: one intent must not become two."""
    sessions: list[FailingSession] = []

    def fake_open(server_url, storage, loopback, timeout_seconds):
        session = FailingSession(
            MCPSession(memory_opener(mock_server), timeout_seconds=TIMEOUT), after=1
        )
        sessions.append(session)
        return session

    monkeypatch.setattr(cli, "_open_robinhood_session", fake_open)
    agent_id = add_rule_agent(tmp_path)
    before = brokerage.held(ACCOUNT, "XYZ")

    result = run_live(agent_id, "--live-standing-ok")

    assert result.exit_code == 1
    assert brokerage.held(ACCOUNT, "XYZ") == before
    assert "place_order" not in brokerage.argument_text()
    note = next(r for r in records(mapped, agent_id) if r.payload.get("event") == "broker_failed")
    assert "was not retried" in note.payload["reason"]
    assert "stopped" in result.output


def test_after_an_mcp_failure_the_agent_is_still_stoppable_and_its_record_verifies(
    mapped: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mock_server: MCPServer
):
    """A failure that locks the protective act is still a failure."""

    def fake_open(server_url, storage, loopback, timeout_seconds):
        return FailingSession(
            MCPSession(memory_opener(mock_server), timeout_seconds=TIMEOUT), after=0
        )

    monkeypatch.setattr(cli, "_open_robinhood_session", fake_open)
    agent_id = add_rule_agent(tmp_path)
    run_live(agent_id, "--live-standing-ok")

    assert invoke("stop", agent_id).exit_code == 0
    assert AgentRun(mapped, agent_id).stop_requested()
    assert AgentRun(mapped, agent_id).verify_ledger().ok
    assert invoke("ledger", agent_id, "--verify").exit_code == 0


def test_a_session_that_will_not_open_refuses_before_any_order(
    mapped: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def refuse(server_url, storage, loopback, timeout_seconds):
        raise BrokerUnavailable("the broker did not complete the MCP handshake.")

    monkeypatch.setattr(cli, "_open_robinhood_session", refuse)
    agent_id = add_rule_agent(tmp_path)

    result = run_live(agent_id, "--live-standing-ok")

    assert result.exit_code == 2
    assert "handshake" in result.output


# ----------------------------------------------------------------------
# The product rules the live path carries
# ----------------------------------------------------------------------


def test_the_run_help_names_no_security_and_no_ai_agent():
    output = invoke("run", "--help").output
    assert "ai agent" not in output.lower()
    assert REAL_TICKERS.findall(output) == []


def test_the_run_help_says_live_is_an_explicit_act():
    lines = invoke("run", "--help").output.split("\n")
    output = " ".join(line.strip("│ ").strip() for line in lines)
    assert "without this flag every run is paper" in output
    assert "--live-standing-ok" in output
