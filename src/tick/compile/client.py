"""The seam between the compiler and whoever runs the model.

`SpecProposer` is a one-method port. The compiler builds a `ProposalRequest`,
hands it over, and gets back either a proposed spec payload or a list of
questions. It never touches an SDK, a socket, or a key.

That seam exists for three reasons, in order of importance:

1. **The user's model account is theirs.** The proposer is injected, so the
   thing holding the user's API key is chosen by the caller and lives for the
   length of one call. There is no client Tick constructs on its own and no
   endpoint Tick operates.
2. **Tests are offline by construction, not by mocking a library.** The fake in
   `tests/compile` implements this protocol and replays recorded fixtures; no
   test needs the SDK's internals, and none can reach the network by accident.
3. **The request is inspectable.** `ProposalRequest` is a plain frozen record,
   so a test can assert exactly what would have been sent — which is how the
   "only the user's words go up" claim is checked rather than asserted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "Proposal",
    "ProposalRequest",
    "QuestionsProposal",
    "SpecProposal",
    "SpecProposer",
]


@dataclass(frozen=True, slots=True)
class ProposalRequest:
    """Everything that would go on the wire, as data.

    Nothing here is derived from the user's account: `system` is a constant,
    `tools` is generated from the spec models, and `messages` carries the
    user's own text plus — on a retry — the validator's complaint about the
    model's previous attempt.
    """

    model: str
    system: str
    messages: tuple[Mapping[str, Any], ...]
    tools: tuple[Mapping[str, Any], ...]
    max_tokens: int


@dataclass(frozen=True, slots=True)
class SpecProposal:
    """The model called `emit_strategy_spec`. `payload` is unvalidated JSON."""

    payload: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class QuestionsProposal:
    """The model called `ask_for_missing_details` rather than inventing."""

    questions: tuple[str, ...]


Proposal = SpecProposal | QuestionsProposal


@runtime_checkable
class SpecProposer(Protocol):
    """Anything that can turn a `ProposalRequest` into a `Proposal`.

    Implementations raise `ModelReplyError` when the reply is not one of the
    two tool calls the request offered.
    """

    def propose(self, request: ProposalRequest) -> Proposal:  # pragma: no cover - protocol
        ...


def first_tool_call(
    blocks: Sequence[Any],
    *,
    known: frozenset[str],
) -> tuple[str, Mapping[str, Any]] | None:
    """The first recognised tool call in a reply's content blocks, or `None`.

    Shared by the SDK adapter and the test fake so both read a reply the same
    way. An unrecognised tool name is treated as no call at all: the compiler
    refuses a reply it does not understand rather than guessing at its shape.
    """
    for block in blocks:
        if getattr(block, "type", None) != "tool_use":
            continue
        name = getattr(block, "name", None)
        if name in known:
            return name, getattr(block, "input", {}) or {}
    return None
