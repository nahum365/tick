"""The one file in Tick that talks to a model provider, and how it is bounded.

This adapter is the only place the `anthropic` SDK is imported. Everything it
is allowed to do is visible in forty lines:

- **The key is the user's, from the user's environment.** `ANTHROPIC_API_KEY`,
  or a key the caller passes for the length of one call. Tick never writes a
  key to disk, never puts one in a record, and never reads one from `TICK_HOME`.
- **There is no Tick endpoint.** No `base_url`, no proxy, no gateway, no
  default host of our own. The client is the SDK's, pointed where the SDK
  points it, so the traffic is between the user and the provider they pay.
- **Nothing about the account goes up.** The adapter sends the `ProposalRequest`
  it is given and nothing else, and that request is built from a constant
  system prompt, the generated grammar, and the user's own words.

The SDK client itself is injected too (`client=`), so this module can be
exercised against a stand-in that has the SDK's shape without a socket
existing. `for_environment()` is the one function that constructs a real one.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import anthropic

from .client import Proposal, ProposalRequest, QuestionsProposal, SpecProposal, first_tool_call
from .errors import MissingApiKey, ModelReplyError
from .schema import ASK_TOOL_NAME, EMIT_TOOL_NAME, TOOL_NAMES

__all__ = ["API_KEY_ENV", "AnthropicSpecProposer", "read_reply"]

#: The only place a key is read from. Tick stores none.
API_KEY_ENV = "ANTHROPIC_API_KEY"

_NO_KEY = (
    f"no model API key. The compiler uses YOUR account: set {API_KEY_ENV} in this "
    f"shell, or pass one in. Tick operates no model endpoint of its own, stores no "
    f"key, and will not compile without one."
)


class AnthropicSpecProposer:
    """Turns a `ProposalRequest` into one Messages API call.

    The model is forced to call one of the two tools (`tool_choice: any`), so a
    reply is either a proposed spec or a list of questions. A reply that is
    neither — prose, an unknown tool, a provider-side refusal — raises
    `ModelReplyError`; nothing is parsed and nothing is written.
    """

    def __init__(self, client: Any) -> None:
        self._client = client

    @classmethod
    def for_environment(cls, *, api_key: str | None = None) -> AnthropicSpecProposer:
        """Build a proposer from the user's own key. Never from anything of ours."""
        key = api_key if api_key is not None else os.environ.get(API_KEY_ENV)
        if not key:
            raise MissingApiKey([_NO_KEY], summary="the compiler needs your model API key.")
        return cls(anthropic.Anthropic(api_key=key))

    def propose(self, request: ProposalRequest) -> Proposal:
        """Make the call and read the reply as one of the two tools."""
        try:
            reply = self._client.messages.create(
                model=request.model,
                max_tokens=request.max_tokens,
                system=request.system,
                messages=[dict(message) for message in request.messages],
                tools=[dict(tool) for tool in request.tools],
                tool_choice={"type": "any"},
            )
        except anthropic.APIError as exc:  # pragma: no cover - needs a live client
            raise ModelReplyError(
                [f"the model provider refused the request: {exc}"],
                summary="the compiler could not reach your model provider.",
            ) from exc
        return read_reply(reply)


def read_reply(reply: Any) -> Proposal:
    """Turn a Messages reply into a proposal, or refuse to read it.

    Public because the test fake replays recorded replies through it: the way a
    reply is read is part of the adapter, and a fake that re-implemented it
    would be testing itself.
    """
    if getattr(reply, "stop_reason", None) == "refusal":
        details = getattr(reply, "stop_details", None)
        category = getattr(details, "category", None)
        raise ModelReplyError(
            [f"the model declined to answer (category: {category})"],
            summary="the model refused this request; nothing was compiled.",
        )
    call = first_tool_call(getattr(reply, "content", []) or [], known=TOOL_NAMES)
    if call is None:
        raise ModelReplyError(
            [
                f"the reply called neither {EMIT_TOOL_NAME} nor {ASK_TOOL_NAME}; "
                f"stop_reason was {getattr(reply, 'stop_reason', None)!r}"
            ],
            summary="the model answered with something that is not a strategy spec.",
        )
    name, payload = call
    if name == ASK_TOOL_NAME:
        return QuestionsProposal(_questions(payload))
    return SpecProposal(dict(payload))


def _questions(payload: Mapping[str, Any]) -> tuple[str, ...]:
    raw = payload.get("questions")
    questions = tuple(str(item).strip() for item in raw or () if str(item).strip())
    if not questions:
        raise ModelReplyError(
            [f"{ASK_TOOL_NAME} was called with no questions in it"],
            summary="the model refused but said nothing you could answer.",
        )
    return questions
