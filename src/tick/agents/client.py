"""The seam between a caged model agent and whoever runs the model.

`ModelClient` is a one-method port. The agent composes a `ModelRequest`, hands
it over, and gets back a `ModelReply`. It never touches an SDK, a socket, or a
key.

**`ModelRequest` has no field for text of Tick's own.** There is no `system`,
no preamble, no examples list — not "we leave it empty", but no place to put
one. The messages are the user's instructions and the snapshot; the tools are
the generated schema. That is the audit claim ("Tick contributes only the JSON
schema, the snapshot and the cage") expressed as a shape rather than as a
promise, and `tests/agents/test_prompt.py` reads the composed request apart to
prove nothing else got in.

**The reply must say which model produced it.** `ModelReply.model` is required
and non-empty. A decision record that cannot name the model that made the
decision is not a record, so a reply without one raises rather than being
recorded against the id the user configured — which may be an alias that
resolved to something else.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .errors import ModelReplyError

__all__ = [
    "ModelClient",
    "ModelReply",
    "ModelRequest",
    "StructuredReply",
    "first_tool_call",
    "intents_of",
]


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """Everything that would go on the wire, as data a test can read apart.

    `messages` carries the user's own instructions and this tick's snapshot.
    `tools` carries the generated intent schema. There is nothing else, and
    there is nowhere else for anything to be.
    """

    model: str
    messages: tuple[Mapping[str, Any], ...]
    tools: tuple[Mapping[str, Any], ...]
    max_tokens: int

    def composed_text(self) -> str:
        """Every character of prompt text in this request, in order.

        The audit test uses it: strip the user's instructions and the snapshot
        out of this string and what remains must be whitespace.
        """
        return "\n".join(str(message.get("content", "")) for message in self.messages)


@dataclass(frozen=True, slots=True)
class ModelReply:
    """What came back: which model answered, and the raw intents it emitted.

    `intents` is unvalidated — the model's own JSON, as it arrived, element
    types included. Validating it is `model_agent.py`'s job, and it produces
    recorded refusals rather than exceptions, because a badly-shaped intent is
    a fact about one order and not about the agent.
    """

    model: str
    intents: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ModelReplyError(
                "the reply does not say which model produced it. Tick will not record a "
                "decision against a model id it was told rather than shown; nothing was "
                "placed."
            )


@dataclass(frozen=True, slots=True)
class StructuredReply:
    """A schema-shaped reply for a request other than order intents.

    The interview shares the model port with a running model agent, but offers
    a different single schema. Keeping that payload named and uninterpreted
    here lets each caller validate its own document without adding a second
    provider method or any provider-specific parsing to the interview.
    """

    model: str
    tool_name: str
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ModelReplyError(
                "the reply does not say which model produced it. Tick cannot record "
                "model provenance without the provider-reported id; answer the current "
                "question again after the provider can report it."
            )
        if not self.tool_name.strip():
            raise ModelReplyError(
                "the structured reply names no tool. Tick cannot tell which schema it "
                "answered; answer the current question again."
            )


@runtime_checkable
class ModelClient(Protocol):
    """Anything that can turn a `ModelRequest` into its one schema-shaped reply.

    Implementations raise `ModelReplyError` when the reply is not the one tool
    call the request offered, and `ModelAgentError` for anything that stops the
    tick. They never retry: a second call after an unreadable answer is a
    machine arguing with itself on somebody's money.
    """

    def propose(
        self, request: ModelRequest
    ) -> ModelReply | StructuredReply:  # pragma: no cover - protocol
        ...


def first_tool_call(
    blocks: Sequence[Any],
    *,
    known: frozenset[str],
) -> tuple[str, Mapping[str, Any]] | None:
    """The first recognised tool call in a reply's content blocks, or `None`.

    Shared by the SDK adapter and the test fake so both read a reply the same
    way. An unrecognised tool name is treated as no call at all: the agent
    refuses a reply it does not understand rather than guessing at its shape.
    """
    for block in blocks:
        if getattr(block, "type", None) != "tool_use":
            continue
        name = getattr(block, "name", None)
        if name in known:
            return name, getattr(block, "input", {}) or {}
    return None


def intents_of(payload: Mapping[str, Any], *, source: str) -> tuple[Any, ...]:
    """The intents list inside a schema-shaped answer, as it arrived.

    Shared by every adapter, so "an empty list means place nothing and a
    missing one is unreadable" is one rule rather than one per provider. Shape
    checking of each element happens downstream, per intent, as refusals.
    """
    raw = payload.get("intents")
    if raw is None:
        raise ModelReplyError(
            f"{source} carries no 'intents' key. An empty list is how a model agent says "
            f"it is placing nothing; a missing one is unreadable."
        )
    if not isinstance(raw, list):
        raise ModelReplyError(
            f"{source} answered with a {type(raw).__name__} where the schema declares a "
            f"list of intents. Nothing was placed."
        )
    return tuple(raw)
