from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from tick.agents import AnthropicChatClient
from tick.agents.codex_client import CodexChatClient


class Completed:
    returncode = 0
    stderr = ""
    stdout = "\n".join(
        (
            json.dumps({"type": "item.completed", "item": {"type": "message", "text": "hi"}}),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {
                        "type": "tool_result",
                        "name": "status",
                        "result": {"ok": True, "evidence": ["checked"]},
                    },
                }
            ),
        )
    )


def test_codex_chat_argv_ignores_user_servers_and_registers_only_tick(tmp_path):
    seen = []

    def run(argv, prompt, timeout):
        seen.append((list(argv), prompt, timeout))
        return Completed()

    client = CodexChatClient(
        run=run,
        binary="codex",
        tick_command="/opt/tick/.venv/bin/tick",
        timeout_seconds=20,
    )

    chunks = tuple(
        client.turn((), "structural frame", setup_session_id=None, model="provider-model")
    )

    argv = seen[0][0]
    assert "--ignore-user-config" in argv
    assert [value for value in argv if value.startswith("mcp_servers.")] == [
        'mcp_servers.tick.default_tools_approval_mode="approve"',
        'mcp_servers.tick.command="/opt/tick/.venv/bin/tick"',
        'mcp_servers.tick.args=["mcp"]',
    ]
    assert argv[argv.index("-s") + 1] == "read-only"
    assert argv[argv.index("-m") + 1] == "provider-model"
    assert chunks == (
        {"kind": "text", "text": "hi"},
        {
            "kind": "tool_result",
            "name": "status",
            "result": {"ok": True, "evidence": ["checked"]},
            "evidence": ["checked"],
        },
        {"kind": "done", "model": "provider-model"},
    )


def test_codex_0_149_fixture_decodes_completed_tools_and_usage():
    fixture = Path(__file__).with_name("codex_chat_events_0_149_0.jsonl")
    completed = SimpleNamespace(
        returncode=0,
        stdout=fixture.read_text(encoding="utf-8"),
        stderr="",
    )
    client = CodexChatClient(
        run=lambda _argv, _prompt, _timeout: completed,
        binary="codex",
        tick_command="tick",
        timeout_seconds=20,
    )

    chunks = tuple(client.turn((), "frame", setup_session_id="setup-1", model="shown-model"))

    assert [chunk["kind"] for chunk in chunks] == [
        "text",
        "tool_call",
        "tool_result",
        "tool_call",
        "tool_result",
        "text",
        "done",
    ]
    assert chunks[1] == {
        "kind": "tool_call",
        "name": "broker_draft",
        "server": "tick",
        "arguments": {},
    }
    assert chunks[2]["name"] == "broker_draft"
    assert chunks[-1]["model"] == "shown-model"
    assert chunks[-1]["usage"]["output_tokens"] == 422


def test_codex_0_149_failed_tool_and_provider_error_remain_unsourced_refusals():
    events = (
        {
            "type": "item.completed",
            "item": {
                "type": "mcp_tool_call",
                "tool": "broker_draft",
                "status": "failed",
                "error": {"message": "tool approval was refused. Retry after fixing policy."},
            },
        },
        {
            "type": "item.completed",
            "item": {"type": "error", "message": "provider transport failed. Retry the turn."},
        },
    )
    completed = SimpleNamespace(
        returncode=0,
        stdout="\n".join(json.dumps(event) for event in events),
        stderr="",
    )
    client = CodexChatClient(
        run=lambda _argv, _prompt, _timeout: completed,
        binary="codex",
        tick_command="tick",
        timeout_seconds=20,
    )

    chunks = tuple(client.turn((), "frame", setup_session_id=None, model="shown-model"))

    assert chunks[0] == {
        "kind": "tool_error",
        "name": "broker_draft",
        "message": "tool approval was refused. Retry after fixing policy.",
    }
    assert chunks[1] == {
        "kind": "text",
        "text": "provider transport failed. Retry the turn.",
        "source": "provider",
    }
    assert "evidence" not in chunks[0] and "evidence" not in chunks[1]


def test_codex_mcp_text_result_is_decoded_for_evidence_badges():
    result = {
        "content": [
            {
                "type": "text",
                "text": json.dumps(
                    {"value": "available", "evidence": ["checked"]},
                    separators=(",", ":"),
                ),
            }
        ],
        "structured_content": None,
    }
    event = {
        "type": "item.completed",
        "item": {
            "type": "mcp_tool_call",
            "server": "tick",
            "tool": "status",
            "arguments": {},
            "result": result,
            "error": None,
            "status": "completed",
        },
    }
    completed = SimpleNamespace(
        returncode=0,
        stdout=json.dumps(event),
        stderr="",
    )
    client = CodexChatClient(
        run=lambda _argv, _prompt, _timeout: completed,
        binary="codex",
        tick_command="tick",
        timeout_seconds=20,
    )

    chunks = tuple(client.turn((), "frame", setup_session_id=None, model="shown-model"))

    assert chunks[0] == {
        "kind": "tool_result",
        "name": "status",
        "result": {"value": "available", "evidence": ["checked"]},
        "evidence": ["checked"],
    }


def test_codex_chat_probes_the_effective_model_once_without_json(tmp_path):
    seen = []

    def run(argv, prompt, timeout):
        seen.append((list(argv), prompt, timeout))
        return SimpleNamespace(
            returncode=0,
            stdout="ready\n",
            stderr=("OpenAI Codex v0.149.0\n--------\nmodel: provider-resolved-model\n"),
        )

    client = CodexChatClient(
        run=run,
        binary="codex",
        tick_command="/opt/tick/.venv/bin/tick",
        timeout_seconds=20,
    )

    identity = client.identify(None)

    assert identity.model == "provider-resolved-model"
    assert identity.cli_version == "0.149.0"
    argv, prompt, _timeout = seen[0]
    assert argv[:2] == ["codex", "exec"]
    assert "--json" not in argv
    assert "-m" not in argv
    assert argv[argv.index("-s") + 1] == "read-only"
    assert prompt == "Reply with ready."


def test_a_person_chosen_model_skips_the_header_probe():
    seen = []

    def run(argv, prompt, timeout):
        seen.append((list(argv), prompt, timeout))
        return SimpleNamespace(
            returncode=0,
            stdout="codex-cli 0.149.0\n",
            stderr="",
        )

    client = CodexChatClient(
        run=run,
        binary="codex",
        tick_command="tick",
        timeout_seconds=20,
    )

    identity = client.identify("person-chosen-model")

    assert identity.model == "person-chosen-model"
    assert identity.cli_version == "0.149.0"
    assert seen == [(["codex", "--version"], "", 20)]


def test_a_probe_without_a_model_refuses_session_creation_with_the_fix():
    client = CodexChatClient(
        run=lambda _argv, _prompt, _timeout: SimpleNamespace(
            returncode=0,
            stdout="ready\n",
            stderr="OpenAI Codex v0.149.0\n",
        ),
        binary="codex",
        tick_command="tick",
        timeout_seconds=20,
    )

    from tick.agents import ProviderUnavailable

    try:
        client.identify(None)
    except ProviderUnavailable as refused:
        assert "name a model when creating the chat" in str(refused)
    else:
        raise AssertionError("a model-less probe must refuse")


def test_codex_chat_refuses_without_the_release_matched_host(monkeypatch):
    monkeypatch.setattr(
        "tick.agents.codex_client.shutil.which",
        lambda command: "/tick/bin/codex" if command == "codex" else None,
    )

    from tick.agents import ProviderUnavailable

    try:
        CodexChatClient.for_environment(tick_command="tick")
    except ProviderUnavailable as refused:
        assert "Run `tick provider install codex`, then create the chat again" in str(refused)
    else:
        raise AssertionError("chat without the Code Mode host must refuse")


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
