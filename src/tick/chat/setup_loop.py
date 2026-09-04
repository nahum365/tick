"""Bounded setup turns that stop only for a person, completion, or the limit."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from tick.agents.errors import ModelReplyError, ProviderUnavailable

from .session import ChatError, compact_document_frame
from .setup import SETUP_FRAMES, SetupChatSession

__all__ = ["MAX_SETUP_MODEL_TURNS", "SetupLoopDecision", "run_setup_loop"]

MAX_SETUP_MODEL_TURNS = 6

_PROVIDER_KINDS = frozenset(
    {
        "text",
        "tool_call",
        "tool_result",
        "tool_error",
        "proposal",
        "document",
        "done",
        "error",
    }
)


@dataclass(frozen=True, slots=True)
class SetupLoopDecision:
    """One code-owned decision after a model turn and any deterministic proof."""

    status: Literal["retry", "waiting_for_person", "complete"]
    detail: str
    events: tuple[Mapping[str, Any], ...]
    completion_text: str | None


SetupAdapter = Callable[
    [tuple[Mapping[str, Any], ...], str],
    Iterable[Mapping[str, Any]],
]
SetupEvaluator = Callable[[], SetupLoopDecision]


def run_setup_loop(
    setup: SetupChatSession,
    *,
    now: Callable[[], datetime],
    adapter: SetupAdapter,
    evaluate: SetupEvaluator,
) -> Iterable[dict[str, Any]]:
    """Let the provider repair checked failures without waiting for a person.

    The limit applies to provider calls, not tool calls. Every emitted frame is
    written into the hash-linked transcript before it reaches the phone.
    """

    def append(chunk: Mapping[str, Any]) -> dict[str, Any]:
        value = dict(chunk)
        kind = str(value.pop("kind"))
        setup.chat.append(kind, value, at=now())
        if kind == "document":
            return compact_document_frame(value)
        return {"kind": kind, **value}

    def progress(step: str, detail: str) -> dict[str, Any]:
        return append({"kind": "progress", "step": step, "detail": detail})

    for _model_turn in range(MAX_SETUP_MODEL_TURNS):
        yield progress(
            "proposing",
            "The provider is preparing or repairing the complete setup document.",
        )
        terminal = False
        try:
            provider_chunks = adapter(
                setup.chat.turns_for_replay(),
                SETUP_FRAMES[setup.state.scope]
                + (
                    " This setup is for simulation only. Check the read capabilities needed for "
                    "account and market data. Do not require order preflight, live order "
                    "permissions, quantities or sides for order tests. Live trading is set up "
                    "separately. Explain progress in plain language: connection checks and the "
                    "person's agent plan, not documents, proofs, mappings or schema vocabulary. "
                    "Technical results remain in the activity record."
                    if setup.state.goal == "simulation"
                    and setup.state.scope.value == "broker_profile"
                    else (
                        " Explain the person’s agent plan in plain language. Start in simulation; "
                        "live trading is configured separately."
                    )
                    if setup.state.goal == "simulation"
                    else ""
                ),
            )
            for raw in provider_chunks:
                kind = raw.get("kind")
                if kind not in _PROVIDER_KINDS:
                    raise ChatError(
                        "CHAT_STREAM_INVALID",
                        "the provider adapter emitted an unknown setup chunk. The loop "
                        "stopped; retry it.",
                    )
                terminal = kind in {"done", "error"}
                yield append(raw)
        except (ModelReplyError, ProviderUnavailable) as exc:
            reason = str(exc)
            setup.chat.append("text", {"text": reason, "source": "provider"}, at=now())
            yield append(
                {
                    "kind": "refused",
                    "code": (
                        "provider_unavailable"
                        if isinstance(exc, ProviderUnavailable)
                        else "model_reply_error"
                    ),
                    "reason": reason,
                }
            )
            return
        if not terminal:
            yield append({"kind": "done"})

        yield progress("checking", "The box is checking the complete document.")
        decision = evaluate()
        for event in decision.events:
            yield append(event)
        state = setup.state
        yield append(
            {
                "kind": "document",
                "document": state.document,
                "valid": state.valid,
                "complete": state.complete,
                "verdict": state.verdict,
                "proof": state.proof,
            }
        )
        if decision.status == "retry":
            continue
        if decision.status == "waiting_for_person":
            yield progress("waiting_for_person", decision.detail)
            return
        yield progress("valid", decision.detail)
        if decision.completion_text is not None:
            yield append({"kind": "text", "text": decision.completion_text, "source": "box"})
        return

    state = setup.state
    reason = str(state.verdict.get("reason") or "The setup document is still incomplete.")
    stopped = (
        f"The setup loop stopped at its model-turn limit. {reason} "
        "You can edit the document, answer the requested values, or retry."
    )
    setup.save(
        document=state.document,
        valid=state.valid,
        complete=False,
        waiting_for=state.waiting_for,
        probe_values=state.probe_values,
        proof=state.proof,
        verdict={"code": "SETUP_LOOP_LIMIT", "reason": stopped},
        at=now(),
    )
    yield progress("stopped", stopped)
    yield append({"kind": "text", "text": stopped, "source": "box"})
