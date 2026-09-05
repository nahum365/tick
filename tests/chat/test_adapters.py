from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from tick.agents import AnthropicChatClient, Provider, ProviderUnavailable
from tick.agents.codex_client import CodexChatClient, ModelReplyError
from tick.broker import DiscoveredTool, contract_for, inventory_hash
from tick.chat import MAX_REPLAY_CHARACTERS, ChatSession

THREAD = "01a07380-76bc-7f30-8c76-6a382dbdaef9"
TURN = "01a07380-7709-7050-9b01-7ab17f1b4fcc"


def notification(method: str, **params):
    return {"method": method, "params": {"threadId": THREAD, "turnId": TURN, **params}}


def item(method: str, **fields):
    return notification(method, item=fields)


class FakeAppServer:
    """Scripted `codex app-server`: answers requests by method, streams a turn's events.

    Records every request so a test can read the isolation boundary apart. A
    server-to-client request in `turn_events` must be refused by the client.
    """

    def __init__(self, *, model: str = "provider-model", turn_events=(), start_error=None):
        self.model = model
        self.turn_events = list(turn_events)
        self.start_error = start_error
        self.sent: list[dict] = []
        self.argv: list[str] | None = None
        self.closed = False
        self._pending: list[dict] = []

    def __call__(self, argv):
        self.argv = list(argv)
        return self

    def requests(self, method: str) -> list[dict]:
        return [m["params"] for m in self.sent if m.get("method") == method and "id" in m]

    def send(self, message):
        self.sent.append(dict(message))
        method = message.get("method")
        if "id" not in message or method is None:
            return
        if method == "initialize":
            self._pending.append({"id": message["id"], "result": {"userAgent": "tick/0.149.0"}})
        elif method == "thread/start":
            if self.start_error is not None:
                self._pending.append({"id": message["id"], "error": self.start_error})
                return
            self._pending.append(
                {"id": message["id"], "result": {"thread": {"id": THREAD}, "model": self.model}}
            )
        elif method == "turn/start":
            self._pending.append({"id": message["id"], "result": {"turn": {"id": TURN}}})
            self._pending.extend(self.turn_events)
        else:
            self._pending.append(
                {"id": message["id"], "error": {"code": -32601, "message": method}}
            )

    def receive(self, timeout):
        assert timeout > 0
        if not self._pending:
            return None
        return self._pending.pop(0)

    def close(self):
        self.closed = True


def client(fake: FakeAppServer, *, run=None) -> CodexChatClient:
    return CodexChatClient(
        run=run
        or (
            lambda _argv, _prompt, _timeout: SimpleNamespace(
                returncode=0, stdout="codex-cli 0.149.0\n", stderr=""
            )
        ),
        binary="codex",
        tick_command="/opt/tick/.venv/bin/tick",
        timeout_seconds=20,
        transport=fake,
    )


COMPLETED_TURN = [
    item("item/started", type="userMessage", id="u1"),
    item("item/agentMessage/delta", id="m1", delta="hi"),
    notification("item/agentMessage/delta", itemId="m1", delta="hi"),
    item("item/completed", type="agentMessage", id="m1", text="hi"),
    item("item/started", type="mcpToolCall", id="t1", server="tick", tool="status", arguments={}),
    item(
        "item/completed",
        type="mcpToolCall",
        id="t1",
        server="tick",
        tool="status",
        arguments={},
        status="completed",
        result={
            "content": [{"type": "text", "text": json.dumps({"ok": True, "evidence": ["checked"]})}]
        },
    ),
    notification("thread/tokenUsage/updated", tokenUsage={"output_tokens": 422}),
    notification("turn/completed", turn={"id": TURN, "status": "completed", "error": None}),
]


def test_codex_thread_registers_only_tick_and_runs_read_only_without_approvals():
    fake = FakeAppServer(turn_events=COMPLETED_TURN)

    chunks = tuple(
        client(fake).turn((), "structural frame", setup_session_id=None, model="provider-model")
    )

    assert fake.argv == ["codex", "app-server", "--listen", "stdio://"]
    (params,) = fake.requests("thread/start")
    assert params["config"] == {
        "mcp_servers": {
            "tick": {
                "command": "/opt/tick/.venv/bin/tick",
                "args": ["mcp"],
                "default_tools_approval_mode": "approve",
            }
        }
    }
    assert params["sandbox"] == "read-only"
    assert params["approvalPolicy"] == "never"
    assert params["ephemeral"] is True
    assert params["model"] == "provider-model"
    assert params["developerInstructions"] == "structural frame"
    assert [m.get("method") for m in fake.sent[:3]] == ["initialize", "initialized", "thread/start"]
    assert fake.closed
    assert chunks == (
        {"kind": "text_delta", "text": "hi"},
        {"kind": "text", "text": "hi"},
        {"kind": "tool_call", "name": "status", "server": "tick", "arguments": {}},
        {
            "kind": "tool_result",
            "name": "status",
            "result": {"ok": True, "evidence": ["checked"]},
            "evidence": ["checked"],
        },
        {"kind": "done", "model": "provider-model", "usage": {"output_tokens": 422}},
    )


def test_codex_setup_thread_scopes_the_box_server_to_the_session():
    fake = FakeAppServer(turn_events=COMPLETED_TURN)

    tuple(client(fake).turn((), "frame", setup_session_id="setup-1", model="m"))

    (params,) = fake.requests("thread/start")
    assert params["config"]["mcp_servers"]["tick"]["args"] == ["mcp", "--setup-session", "setup-1"]


def test_codex_chat_prompt_stays_under_replay_bound_for_recorded_inventory(tmp_path):
    fixture = (
        Path(__file__).parents[1] / "broker" / "fixtures" / "robinhood_tools_list_2026-09-04.json"
    )
    advertised = json.loads(fixture.read_text(encoding="utf-8"))["tools"]
    contracts = [
        contract_for(DiscoveredTool.model_validate(tool)).model_dump(mode="json")
        for tool in advertised
    ]
    session = ChatSession.create_setup(
        tmp_path,
        provider=Provider.CODEX,
        model="fixture-model",
        codex_cli_version="0.149.0",
        scope="broker_profile",
        at=datetime(2026, 9, 4, 12, 0, tzinfo=UTC),
    )
    session.append(
        "tool_result",
        {
            "name": "broker_inventory",
            "result": {
                "server_url": "https://broker.example.invalid/mcp",
                "inventory_hash": inventory_hash(
                    [contract_for(DiscoveredTool.model_validate(tool)) for tool in advertised]
                ),
                "contracts": contracts,
            },
        },
        at=datetime(2026, 9, 4, 12, 0, tzinfo=UTC),
    )
    fake = FakeAppServer(turn_events=COMPLETED_TURN)

    tuple(
        client(fake).turn(
            session.turns_for_replay(),
            "frame",
            setup_session_id=session.session_id,
            model="fixture-model",
        )
    )

    (params,) = fake.requests("turn/start")
    prompt = params["input"][0]["text"]
    assert params["input"][0]["type"] == "text"
    assert len(prompt) < MAX_REPLAY_CHARACTERS
    assert "input_schema" not in prompt
    assert "contract_hash" in prompt


def test_codex_failed_tool_and_provider_error_remain_unsourced_refusals():
    events = [
        item(
            "item/completed",
            type="mcpToolCall",
            id="t1",
            server="tick",
            tool="broker_draft",
            arguments={},
            status="failed",
            error={"message": "tool refused"},
        ),
        {"method": "error", "params": {"threadId": THREAD, "error": {"message": "rate limited"}}},
        notification("turn/completed", turn={"id": TURN, "status": "completed", "error": None}),
    ]
    chunks = tuple(
        client(FakeAppServer(turn_events=events)).turn((), "f", setup_session_id=None, model="m")
    )

    assert chunks[0] == {"kind": "tool_error", "name": "broker_draft", "message": "tool refused"}
    assert chunks[1] == {"kind": "text", "text": "rate limited", "source": "provider"}
    assert chunks[-1]["kind"] == "done"


def test_codex_failed_turn_stops_with_the_servers_reason():
    events = [
        notification(
            "turn/completed",
            turn={"id": TURN, "status": "failed", "error": {"message": "model requires upgrade"}},
        )
    ]
    with pytest.raises(ModelReplyError, match="model requires upgrade"):
        tuple(
            client(FakeAppServer(turn_events=events)).turn(
                (), "f", setup_session_id=None, model="m"
            )
        )


def test_codex_refuses_every_server_request_and_keeps_streaming():
    events = [
        {
            "id": 77,
            "method": "item/commandExecution/requestApproval",
            "params": {"threadId": THREAD},
        },
        {"id": 78, "method": "mcpServer/elicitation/request", "params": {"threadId": THREAD}},
        item("item/completed", type="agentMessage", id="m1", text="done"),
        notification("turn/completed", turn={"id": TURN, "status": "completed", "error": None}),
    ]
    fake = FakeAppServer(turn_events=events)

    chunks = tuple(client(fake).turn((), "f", setup_session_id=None, model="m"))

    refusals = [m for m in fake.sent if m.get("id") in (77, 78)]
    assert [m["error"]["code"] for m in refusals] == [-32601, -32601]
    assert all("method" not in m for m in refusals)
    assert chunks[0] == {"kind": "text", "text": "done"}


def test_codex_proposal_results_keep_their_kind():
    events = [
        item(
            "item/completed",
            type="mcpToolCall",
            id="t1",
            server="tick",
            tool="propose_stop",
            arguments={},
            status="completed",
            result={"structured_content": {"executed": False, "proposal_id": "p1"}},
        ),
        notification("turn/completed", turn={"id": TURN, "status": "completed", "error": None}),
    ]
    chunks = tuple(
        client(FakeAppServer(turn_events=events)).turn((), "f", setup_session_id=None, model="m")
    )
    assert chunks[0] == {"kind": "proposal", "executed": False, "proposal_id": "p1"}


def test_codex_identity_takes_the_model_from_the_thread_not_a_header():
    fake = FakeAppServer(model="provider-resolved-model")

    identity = client(fake).identify(None)

    assert identity.model == "provider-resolved-model"
    assert identity.cli_version == "0.149.0"
    assert fake.requests("turn/start") == []
    (params,) = fake.requests("thread/start")
    assert "model" not in params and params["ephemeral"] is True
    assert fake.closed


def test_a_person_chosen_model_skips_the_thread_probe():
    seen = []

    def run(argv, prompt, timeout):
        seen.append((list(argv), prompt, timeout))
        return SimpleNamespace(returncode=0, stdout="codex-cli 0.149.0\n", stderr="")

    fake = FakeAppServer()
    identity = client(fake, run=run).identify("person-chosen-model")

    assert identity.model == "person-chosen-model"
    assert identity.cli_version == "0.149.0"
    assert seen == [(["codex", "--version"], "", 20)]
    assert fake.argv is None


def test_a_refused_thread_start_surfaces_the_fix():
    fake = FakeAppServer(start_error={"code": -32000, "message": "not logged in"})

    with pytest.raises(ProviderUnavailable, match="not logged in"):
        client(fake).identify(None)


def test_codex_chat_refuses_without_the_binary(monkeypatch):
    monkeypatch.setattr("tick.agents.codex_client.shutil.which", lambda _command: None)

    with pytest.raises(ProviderUnavailable, match="connect Codex from the app"):
        CodexChatClient.for_environment(tick_command="tick")


def test_anthropic_chat_runs_a_bounded_in_process_tool_loop():
    replies = iter(
        (
            SimpleNamespace(
                model="chosen-model",
                content=[SimpleNamespace(type="tool_use", id="call-1", name="status", input={})],
            ),
            SimpleNamespace(
                model="chosen-model",
                content=[SimpleNamespace(type="text", text="done")],
            ),
        )
    )
    messages = SimpleNamespace(create=lambda **_kwargs: next(replies))
    client = AnthropicChatClient(
        client=SimpleNamespace(messages=messages), max_steps=3, max_tokens=512
    )

    chunks = tuple(
        client.turn(
            model="chosen-model",
            transcript=(),
            frame="structural frame",
            tools=(),
            call_tool=lambda name, arguments: {
                "name": name,
                "arguments": arguments,
                "evidence": ["checked"],
            },
        )
    )

    assert [chunk["kind"] for chunk in chunks] == [
        "tool_call",
        "tool_result",
        "text",
        "done",
    ]
    assert chunks[1]["evidence"] == ["checked"]
