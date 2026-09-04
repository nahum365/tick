"""The one file where a model agent talks to a provider, and how it is bounded.

Everything this adapter may do is visible in forty lines, and it is the same
bounding the compiler's adapter has:

- **The key is the user's, from the user's environment.** `ANTHROPIC_API_KEY`,
  or a key the caller passes for the length of one call. Tick never writes a
  key to disk, never puts one in a record, and never reads one from `TICK_HOME`.
- **There is no Tick endpoint.** No `base_url`, no proxy, no gateway, no
  default host of our own. The traffic is between the user and the provider
  they pay, which is what "bring your own model" has to mean if the account
  data in the snapshot is to stay the user's business.
- **Nothing is sent but the request it is handed.** The request is the user's
  instructions, this tick's snapshot, and the generated schema; there is no
  system prompt to append, because `ModelRequest` has no field for one.
- **No system prompt is passed.** Not "an empty one" — the parameter is not
  supplied at all, so a future edit that wanted to add Tick-authored steering
  would have to add it here, in the file a reviewer reads first.

The SDK client itself is injected (`client=`), so this module is exercised
against a stand-in with the SDK's shape and no socket exists in a test.
`for_environment()` is the one function that constructs a real one.
"""

from __future__ import annotations

import json
import os
from typing import Any

import anthropic

from .client import (
    ModelClient,
    ModelReply,
    ModelRequest,
    StructuredReply,
    first_tool_call,
    intents_of,
)
from .errors import MissingApiKey, ModelReplyError
from .schema import EMIT_TOOL_NAME, TOOL_NAMES

__all__ = [
    "API_KEY_ENV",
    "AnthropicChatClient",
    "AnthropicModelClient",
    "read_model_reply",
    "read_structured_reply",
]

#: The only place a key is read from. Tick stores none.
API_KEY_ENV = "ANTHROPIC_API_KEY"

_NO_KEY = (
    f"no model API key. A model agent runs on YOUR account: set {API_KEY_ENV} in this "
    f"shell, or pass one in. Tick operates no model endpoint of its own, stores no key, "
    f"and will not tick a model agent without one."
)


class AnthropicModelClient:
    """Turns a `ModelRequest` into one Messages API call.

    The model is forced to call the one tool it is offered (`tool_choice:
    any`), so a reply is a list of intents or nothing readable. A reply that is
    neither — prose, an unknown tool, a provider-side refusal, or one that does
    not say which model answered — raises `ModelReplyError` and the tick stops.
    """

    def __init__(self, client: Any) -> None:
        self._client = client

    @classmethod
    def for_environment(cls, *, api_key: str | None = None) -> AnthropicModelClient:
        """Build a client from the user's own key. Never from anything of ours."""
        key = api_key if api_key is not None else os.environ.get(API_KEY_ENV)
        if not key:
            raise MissingApiKey(_NO_KEY)
        return cls(anthropic.Anthropic(api_key=key))

    def propose(self, request: ModelRequest) -> ModelReply | StructuredReply:
        """Make the call and read the reply as the one tool it was offered."""
        if not request.model.strip():
            raise ModelReplyError(
                "the Anthropic request names no model. Set TICK_INTERVIEW_MODEL to the "
                "model you chose and answer the current interview question again."
            )
        try:
            reply = self._client.messages.create(
                model=request.model,
                max_tokens=request.max_tokens,
                messages=[dict(message) for message in request.messages],
                tools=[dict(tool) for tool in request.tools],
                tool_choice={"type": "any"},
            )
        except anthropic.APIError as exc:  # pragma: no cover - needs a live client
            raise ModelReplyError(
                f"the model provider refused the request: {exc}. The tick stopped; "
                f"nothing was placed and nothing was asked again."
            ) from exc
        names = frozenset(str(tool.get("name", "")) for tool in request.tools)
        if names == TOOL_NAMES:
            return read_model_reply(reply)
        return read_structured_reply(reply, known=names)


class AnthropicChatClient:
    """Bounded in-process tool loop against the user's own Anthropic client."""

    def __init__(self, *, client: Any, max_steps: int, max_tokens: int) -> None:
        if max_steps <= 0 or max_tokens <= 0:
            raise ValueError("chat bounds must be positive")
        self._client = client
        self._max_steps = max_steps
        self._max_tokens = max_tokens

    @classmethod
    def for_environment(cls, *, max_steps: int, max_tokens: int) -> AnthropicChatClient:
        key = os.environ.get(API_KEY_ENV)
        if not key:
            raise MissingApiKey(_NO_KEY)
        return cls(
            client=anthropic.Anthropic(api_key=key),
            max_steps=max_steps,
            max_tokens=max_tokens,
        )

    def turn(self, *, model: str, transcript: Any, frame: str, tools: Any, call_tool: Any):
        messages = [
            {
                "role": "user",
                "content": json.dumps(
                    {"frame": frame, "transcript": transcript},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        ]
        for _step in range(self._max_steps):
            reply = self._client.messages.create(
                model=model,
                max_tokens=self._max_tokens,
                messages=messages,
                tools=list(tools),
            )
            assistant_content = []
            tool_results = []
            for block in getattr(reply, "content", ()):
                block_type = getattr(block, "type", None)
                if block_type == "text":
                    text = str(getattr(block, "text", ""))
                    assistant_content.append({"type": "text", "text": text})
                    yield {"kind": "text", "text": text}
                elif block_type == "tool_use":
                    name = str(getattr(block, "name", ""))
                    arguments = dict(getattr(block, "input", {}) or {})
                    call_id = str(getattr(block, "id", ""))
                    assistant_content.append(
                        {"type": "tool_use", "id": call_id, "name": name, "input": arguments}
                    )
                    yield {"kind": "tool_call", "name": name, "arguments": arguments}
                    result = call_tool(name, arguments)
                    if isinstance(result, dict) and result.get("executed") is False:
                        yield {"kind": "proposal", **result}
                    else:
                        chunk = {"kind": "tool_result", "name": name, "result": result}
                        if isinstance(result, dict) and isinstance(result.get("evidence"), list):
                            chunk["evidence"] = result["evidence"]
                        yield chunk
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": call_id,
                            "content": json.dumps(result, sort_keys=True),
                        }
                    )
            if not tool_results:
                yield {"kind": "done", "model": str(getattr(reply, "model", ""))}
                return
            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": tool_results})
        yield {
            "kind": "error",
            "code": "chat_tool_limit",
            "reason": "the bounded tool loop ended. Send another turn to continue.",
        }


def read_structured_reply(reply: Any, *, known: frozenset[str]) -> StructuredReply:
    """Read one non-intent tool call through the same adapter boundary."""
    if getattr(reply, "stop_reason", None) == "refusal":
        details = getattr(reply, "stop_details", None)
        raise ModelReplyError(
            f"the model declined to answer (category: {getattr(details, 'category', None)}). "
            "The current question is still unanswered; answer it again."
        )
    call = first_tool_call(getattr(reply, "content", []) or [], known=known)
    if call is None:
        raise ModelReplyError(
            "the reply did not call the schema it was offered. Tick reads this answer "
            "only from that schema; answer the current question again."
        )
    name, payload = call
    return StructuredReply(
        model=str(getattr(reply, "model", "") or ""),
        tool_name=name,
        payload=dict(payload),
    )


def read_model_reply(reply: Any) -> ModelReply:
    """Turn a Messages reply into a `ModelReply`, or refuse to read it.

    Public because the test fake replays recorded replies through it: how a
    provider reply becomes intents is part of the adapter, and a fake that
    re-implemented it would be testing itself.
    """
    if getattr(reply, "stop_reason", None) == "refusal":
        details = getattr(reply, "stop_details", None)
        raise ModelReplyError(
            f"the model declined to answer (category: {getattr(details, 'category', None)}). "
            f"The tick stopped and nothing was placed."
        )
    call = first_tool_call(getattr(reply, "content", []) or [], known=TOOL_NAMES)
    if call is None:
        raise ModelReplyError(
            f"the reply did not call {EMIT_TOOL_NAME}; stop_reason was "
            f"{getattr(reply, 'stop_reason', None)!r}. Tick reads a decision from the "
            f"tool schema it offered and never from prose."
        )
    _, payload = call
    return ModelReply(
        model=str(getattr(reply, "model", "") or ""),
        intents=intents_of(payload, source=EMIT_TOOL_NAME),
    )


#: A structural assertion for a reader: this adapter satisfies the port above.
_PORT_CONFORMANCE: type[ModelClient] = AnthropicModelClient
