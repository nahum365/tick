from __future__ import annotations

import json
from types import SimpleNamespace

from tick.agents import AnthropicChatClient
from tick.agents.codex_client import CodexChatClient


class Completed:
    returncode = 0
    stderr = "model: provider-model\n"
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

    chunks = client.turn((), "structural frame", setup_session_id=None)

    argv = seen[0][0]
    assert "--ignore-user-config" in argv
    assert [value for value in argv if value.startswith("mcp_servers.")] == [
        'mcp_servers.tick.command="/opt/tick/.venv/bin/tick"',
        'mcp_servers.tick.args=["mcp"]',
    ]
    assert argv[argv.index("-s") + 1] == "read-only"
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
