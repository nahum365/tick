from __future__ import annotations

from datetime import UTC, datetime

from tests.runtime.conftest import build_spec
from tick.mcpbox import BoxTools
from tick.mcpbox.definitions import chat_tool_definitions
from tick.records import read
from tick.runtime import AgentRun, ApprovalMode
from tick.serve.handlers import ServeContext


def test_proposal_tool_records_intent_but_never_executes_it(tmp_path):
    home = tmp_path / "tick-home"
    agent = AgentRun.create(
        home,
        build_spec(),
        max_cancels_per_session=2,
        approval=ApprovalMode.EACH,
        created_at=datetime(2026, 9, 3, 12, 0, tzinfo=UTC),
        instructions=None,
    )
    before = agent.state_path.read_bytes()
    context = ServeContext(
        home=home,
        env={},
        now=lambda: datetime(2026, 9, 3, 12, 0, tzinfo=UTC),
        pid_alive=lambda _pid: False,
        start_process=lambda _argv: 7001,
        signal_process=lambda _pid: None,
        provider_status=lambda: (True, "ok"),
        loopback_status=lambda: (True, "ok"),
        tunnel_status=lambda: (True, "ok"),
        unit_fragments=lambda: (True, "ok", ("paper",)),
        codex_chat_identity=lambda model: {
            "model": model or "fixture-model",
            "codex_cli_version": "0.149.0",
        },
        chat_adapter=lambda _provider, _model, _transcript, _frame, _thread=None: (),
        setup_chat_adapter=(
            lambda _provider, _model, _transcript, _frame, _session, _thread=None: ()
        ),
        provider_login_start=lambda: {},
        provider_browser_login_start=lambda _viewport: {},
        provider_login_status=lambda _login_id: {},
        codex_install=lambda: {
            "code": "CODEX_INSTALLED",
            "path": "/fixture/bin/codex",
            "release": "rust-v0.0.0",
            "sha256": "f" * 64,
            "reason": "fixture installed",
        },
        broker_connect_start=lambda _server, _scheme: {},
        broker_connect_complete=lambda _connect_id, _url: {},
        broker_connect_status=lambda _connect_id: {},
        browser_ceremony_url=lambda _purpose: None,
        browser_bridge=object(),
        broker_profile_operation=lambda _action, _body: {},
        commons_client=lambda: object(),  # type: ignore[return-value]
        metadata=type("FixtureMetadata", (), {"tags": lambda self: frozenset()})(),
    )

    result = BoxTools(context, setup_session_id=None).proposal(
        "stop", agent_id=agent.agent_id, arguments={}, transcript_hash="sha256:turn"
    )

    assert result["executed"] is False
    assert result["action"] == "stop"
    assert result["arguments"] == {"agent_id": agent.agent_id}
    assert result["transcript_hash"] == "sha256:turn"
    assert agent.state_path.read_bytes() == before
    assert not agent.stop_requested()
    row = list(read(agent.ledger_path))[-1]
    assert row.payload["event"] == "proposal"
    assert row.payload["via"] == "chat"
    assert row.payload["transcript_hash"] == "sha256:turn"


def test_proposal_schemas_require_every_value_the_confirmation_route_needs():
    definitions = {value["name"]: value for value in chat_tool_definitions()}
    for definition in definitions.values():
        schema = definition["input_schema"]
        assert set(schema["required"]) == set(schema["properties"])
        assert schema["additionalProperties"] is False
    assert set(definitions["propose_adopt_draft"]["input_schema"]["required"]) == {
        "draft_id",
        "name",
        "max_cancels",
        "approval",
        "transcript_hash",
    }
    assert "model" in definitions["start_interview"]["input_schema"]["required"]
