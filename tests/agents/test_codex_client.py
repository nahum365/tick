"""The Codex adapter: one `codex exec` per tick, read apart without a process.

A fake runner stands in for `subprocess.run`. It records the argv and the
prompt it was handed, writes the final-message file a real run writes, and
returns the stderr header a real run prints. No test here starts a process;
`for_environment` is exercised only for its refusal when the binary is absent.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from tick.agents import (
    CODEX_BINARY,
    CodexModelClient,
    ModelReplyError,
    ModelRequest,
    ProviderUnavailable,
    intents_schema,
    tool_definitions,
)

from .conftest import agent_for, market, portfolio

HEADER = (
    "OpenAI Codex v0.149.0\n--------\nworkdir: /tmp/x\nmodel: gpt-5.6-terra\n"
    "provider: openai\napproval: on-request\nsandbox: read-only\n--------\n"
)


@dataclass
class Completed:
    returncode: int
    stdout: str = ""
    stderr: str = HEADER


@dataclass
class FakeCodex:
    """Behaves like `codex exec` from the outside: files in, files out."""

    final: Any = None
    final_text: str | None = None
    returncode: int = 0
    stderr: str = HEADER
    write_final: bool = True
    calls: list[tuple[list[str], str, float]] = field(default_factory=list)
    schema_seen: dict[str, Any] | None = None

    def __call__(self, argv: Sequence[str], prompt: str, timeout: float) -> Completed:
        argv = list(argv)
        self.calls.append((argv, prompt, timeout))
        schema_path = Path(argv[argv.index("--output-schema") + 1])
        self.schema_seen = json.loads(schema_path.read_text(encoding="utf-8"))
        assert Path(argv[argv.index("-C") + 1]).is_dir()
        if self.write_final:
            last = Path(argv[argv.index("-o") + 1])
            text = self.final_text if self.final_text is not None else json.dumps(self.final)
            last.write_text(text, encoding="utf-8")
        return Completed(self.returncode, stderr=self.stderr)


def request(model: str = "gpt-5.6-terra") -> ModelRequest:
    return ModelRequest(
        model=model,
        messages=({"role": "user", "content": 'My own words.\n\n{"snapshot": 1}'},),
        tools=tool_definitions(),
        max_tokens=100,
    )


def client(fake: FakeCodex) -> CodexModelClient:
    return CodexModelClient(run=fake, binary="/opt/bin/codex", timeout_seconds=42.0)


def test_a_conforming_answer_becomes_intents_and_names_the_model_from_the_header():
    fake = FakeCodex(final={"intents": [{"symbol": "XYZ", "side": "buy", "qty": 1, "reason": "r"}]})

    reply = client(fake).propose(request())

    assert reply.model == "gpt-5.6-terra"
    assert reply.intents == ({"symbol": "XYZ", "side": "buy", "qty": 1, "reason": "r"},)


def test_the_command_line_is_the_bounded_one():
    fake = FakeCodex(final={"intents": []})
    client(fake).propose(request(model="gpt-5.6-sol"))

    argv, prompt, timeout = fake.calls[0]
    assert argv[0] == "/opt/bin/codex" and argv[1] == "exec"
    for flag in ("--ignore-user-config", "--ignore-rules", "--ephemeral", "--skip-git-repo-check"):
        assert flag in argv, (
            f"{flag} missing: the user's MCP servers, hooks and rules must not load"
        )
    assert argv[argv.index("-s") + 1] == "read-only"
    assert argv[argv.index("-m") + 1] == "gpt-5.6-sol"
    assert argv[-1] == "-", "the prompt goes on stdin, never on the command line"
    assert timeout == 42.0
    assert prompt == request().composed_text()
    assert "system" not in prompt.lower().split("my own words")[0]


def test_the_schema_offered_is_the_intents_schema_and_nothing_else():
    fake = FakeCodex(final={"intents": []})
    client(fake).propose(request())
    assert fake.schema_seen == intents_schema()


def test_an_empty_list_is_a_complete_answer():
    reply = client(FakeCodex(final={"intents": []})).propose(request())
    assert reply.intents == ()


@pytest.mark.parametrize(
    ("fake", "fragment"),
    [
        (FakeCodex(returncode=1, stderr="boom\nauth failed"), "status 1"),
        (FakeCodex(final={"intents": []}, stderr="no model line here"), "which model"),
        (FakeCodex(final={"intents": []}, write_final=False), "final message"),
        (FakeCodex(final_text="not json"), "not JSON"),
        (FakeCodex(final=[1, 2]), "object with an 'intents' list"),
        (FakeCodex(final={"orders": []}), "no 'intents' key"),
        (FakeCodex(final={"intents": "XYZ"}), "list of intents"),
    ],
    ids=[
        "nonzero-exit",
        "no-model-header",
        "no-final-file",
        "not-json",
        "not-object",
        "no-key",
        "not-list",
    ],
)
def test_every_unreadable_run_stops_the_tick_with_a_reason(fake: FakeCodex, fragment: str):
    with pytest.raises(ModelReplyError, match=fragment):
        client(fake).propose(request())


def test_a_timeout_stops_the_tick_rather_than_asking_again():
    calls = 0

    def slow(argv: Sequence[str], prompt: str, timeout: float) -> Completed:
        nonlocal calls
        calls += 1
        raise subprocess.TimeoutExpired(cmd=list(argv), timeout=timeout)

    with pytest.raises(ModelReplyError, match="did not answer within 42s"):
        CodexModelClient(run=slow, timeout_seconds=42.0).propose(request())
    assert calls == 1


def test_no_intents_schema_in_the_request_asks_nothing():
    fake = FakeCodex(final={"intents": []})
    bare = ModelRequest(model="m", messages=(), tools=(), max_tokens=1)
    with pytest.raises(ModelReplyError, match="no intents schema"):
        client(fake).propose(bare)
    assert fake.calls == []


def test_for_environment_refuses_without_the_binary(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("tick.agents.codex_client.shutil.which", lambda name: None)
    with pytest.raises(ProviderUnavailable, match=f"no `{CODEX_BINARY}` command"):
        CodexModelClient.for_environment()


def test_a_model_agent_ticks_end_to_end_on_the_codex_adapter():
    """The adapter satisfies the port: the agent's own cage still judges every intent."""
    fake = FakeCodex(
        final={
            "intents": [
                {"symbol": "XYZ", "side": "buy", "qty": 2, "reason": "mine"},
                {"symbol": "NOPE", "side": "buy", "qty": 1, "reason": "outside"},
            ]
        }
    )
    agent = agent_for(client(fake))
    from .conftest import NOW

    evaluation = agent.evaluate_tick(agent.spec, market(), portfolio(), NOW)

    accepted = [d for d in evaluation.decisions if d.refusal is None]
    refused = [d for d in evaluation.decisions if d.refusal is not None]
    assert [d.intent.symbol for d in accepted if d.intent] == ["XYZ"]
    assert len(refused) == 1 and "universe" in refused[0].refusal.reason
    assert agent.provenance()["model_reported"] == "gpt-5.6-terra"
    assert agent.provenance()["provider"] == "anthropic"  # the document's provider, unchanged
