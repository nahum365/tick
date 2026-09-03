"""Both shipped adapters carry the slot schema through their existing port."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from tick.agents import (
    AnthropicModelClient,
    CodexModelClient,
    ModelReplyError,
    ModelRequest,
    StructuredReply,
)
from tick.interview import EXTRACT_TOOL_NAME, SLOTS


def request(*, model: str) -> ModelRequest:
    tool = {
        "name": EXTRACT_TOOL_NAME,
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["value"],
            "properties": {"value": SLOTS[0].schema},
        },
    }
    return ModelRequest(
        model=model,
        messages=({"role": "user", "content": "Use XYZ."},),
        tools=(tool,),
        max_tokens=100,
    )


class Messages:
    def __init__(self) -> None:
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            model="model-reported",
            stop_reason="tool_use",
            stop_details=None,
            content=[
                SimpleNamespace(
                    type="tool_use",
                    name=EXTRACT_TOOL_NAME,
                    input={"value": ["XYZ"]},
                )
            ],
        )


def test_anthropic_offers_the_slot_schema_as_the_tool_and_adds_no_system_text():
    messages = Messages()
    reply = AnthropicModelClient(SimpleNamespace(messages=messages)).propose(
        request(model="model-selected")
    )

    assert isinstance(reply, StructuredReply)
    assert reply.payload == {"value": ["XYZ"]}
    assert messages.kwargs["tools"] == [dict(request(model="model-selected").tools[0])]
    assert messages.kwargs["messages"] == [{"role": "user", "content": "Use XYZ."}]
    assert "system" not in messages.kwargs


def test_anthropic_refuses_an_unnamed_interviewer_model_before_a_call():
    messages = Messages()
    with pytest.raises(ModelReplyError, match="TICK_INTERVIEW_MODEL.*question again"):
        AnthropicModelClient(SimpleNamespace(messages=messages)).propose(request(model=""))
    assert messages.kwargs is None


@dataclass
class Completed:
    returncode: int = 0
    stdout: str = ""
    stderr: str = "model: model-reported\n"


class CodexRun:
    def __init__(self) -> None:
        self.argv = None
        self.prompt = None
        self.schema = None

    def __call__(self, argv, prompt, timeout):
        self.argv = list(argv)
        self.prompt = prompt
        schema_path = Path(self.argv[self.argv.index("--output-schema") + 1])
        self.schema = json.loads(schema_path.read_text(encoding="utf-8"))
        last_path = Path(self.argv[self.argv.index("-o") + 1])
        last_path.write_text(json.dumps({"value": ["XYZ"]}), encoding="utf-8")
        return Completed()


def test_codex_uses_output_schema_and_the_transcript_is_the_whole_prompt():
    fake = CodexRun()
    reply = CodexModelClient(run=fake).propose(request(model=""))

    assert isinstance(reply, StructuredReply)
    assert reply.tool_name == EXTRACT_TOOL_NAME
    assert fake.schema == request(model="").tools[0]["input_schema"]
    assert fake.prompt == "Use XYZ."
    assert "-m" not in fake.argv
