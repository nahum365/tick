"""`compile_text` — a person's own words in, a validated spec or a question out.

The whole flow is here and it is deliberately short:

    words ─► model (forced to call one of two tools)
              │
              ├─ ask_for_missing_details ──────────────► CompileRefusal
              │
              └─ emit_strategy_spec
                    │
                    ├─ fails the spec validators ─► ONE retry with the error
                    │      └─ fails again ────────────► CompileError (both)
                    │
                    └─ validates
                          │
                          ├─ a symbol or number is not in the words ─► CompileRefusal
                          │
                          └─ traceable ─────────────────► CompileResult + explanation

Three things this function does NOT do, each on purpose.

**It never executes anything.** It returns a document. The document is the same
`StrategySpec` a person could have written by hand, gated by the same
validators, run by the same engine under the same cage (invariant 3). There is
no path from a model's reply to a broker that does not go through them.

**It never invents.** The model is instructed not to, and then
`trace.questions_for_untraceable` checks — so a model that ignores the
instruction produces a refusal with a question in it, not a strategy nobody
chose. A refusal is a normal result, not an error.

**It retries once, and only for a malformed document.** A second attempt at a
validation failure is worth making: the readable error is usually enough for
the model to fix its own JSON. A third would be a machine arguing with itself
on the user's money, and a retry after a REFUSAL would be Tick pushing back on
the user's own "you didn't tell me" — so neither happens.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from tick.spec import SpecError, StrategySpec, parse_spec

from .client import ProposalRequest, QuestionsProposal, SpecProposal, SpecProposer
from .errors import CompileError, ModelReplyError
from .explain import RuleExplanation, explain
from .prompt import SYSTEM_PROMPT, user_messages
from .schema import tool_definitions
from .trace import questions_for_untraceable

__all__ = [
    "DEFAULT_MODEL",
    "MAX_OUTPUT_TOKENS",
    "CompileRefusal",
    "CompileResult",
    "compile_text",
]

#: The model used when a caller names none. It is not silent: every surface
#: that compiles a spec prints the model id it used, and `tick agent new`
#: takes `--model` to change it.
DEFAULT_MODEL = "claude-opus-5"

#: A spec is a small document; this is headroom, not a target.
MAX_OUTPUT_TOKENS = 16000

#: How many times the model may be asked. One call, plus one repair attempt.
MAX_ATTEMPTS = 2

_RETRY_PREAMBLE = (
    "That document is not a valid strategy spec. The runtime rejected it with "
    "the problems below. Emit a corrected spec that fixes every one of them, "
    "still using ONLY the securities and numbers in my words above — if fixing "
    "it would need a number I did not give you, call ask_for_missing_details "
    "instead.\n\n"
)


@dataclass(frozen=True, slots=True)
class CompileResult:
    """A spec the runtime will accept, with the explanation of what it does."""

    spec: StrategySpec
    explanation: tuple[RuleExplanation, ...]
    attempts: int
    model: str


@dataclass(frozen=True, slots=True)
class CompileRefusal:
    """No spec: something the user must answer first.

    `origin` says who refused — the model, because the words were missing an
    element; or Tick, because a symbol or number in the model's spec was not in
    the words. Both are the same thing from the user's side (here are the
    questions), and they are distinguished because only one of them is evidence
    that the model tried to author something.
    """

    questions: tuple[str, ...]
    origin: str
    attempts: int
    model: str


#: `origin` values for a refusal.
FROM_MODEL = "model"
FROM_TRACEABILITY = "traceability"


def compile_text(
    text: str,
    proposer: SpecProposer,
    *,
    model: str,
) -> CompileResult | CompileRefusal:
    """Compile one person's words into a spec, or refuse with questions.

    `proposer` is injected: the thing holding the user's API key is chosen by
    the caller, and this function neither builds one nor knows a URL. Raises
    `CompileError` when the model's output could not be validated twice, and
    `ModelReplyError` when the reply was not one of the two offered tools.
    """
    words = text.strip()
    if not words:
        raise CompileError(
            ["the text to compile is empty"],
            summary="there is nothing to compile.",
        )

    tools = tool_definitions()
    messages: list[dict[str, str]] = [dict(message) for message in user_messages(words)]
    problems: list[str] = []

    for attempt in range(1, MAX_ATTEMPTS + 1):
        request = ProposalRequest(
            model=model,
            system=SYSTEM_PROMPT,
            messages=tuple(messages),
            tools=tools,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        proposal = proposer.propose(request)

        if isinstance(proposal, QuestionsProposal):
            return CompileRefusal(
                questions=proposal.questions,
                origin=FROM_MODEL,
                attempts=attempt,
                model=model,
            )
        if not isinstance(proposal, SpecProposal):  # pragma: no cover - defensive
            raise ModelReplyError(
                [f"the proposer returned {type(proposal).__name__}"],
                summary="the model answered with something that is not a strategy spec.",
            )

        try:
            spec = parse_spec(proposal.payload, source="the compiled spec")
        except SpecError as exc:
            problems.append(f"attempt {attempt}: {exc}")
            if attempt == MAX_ATTEMPTS:
                raise CompileError(
                    problems,
                    summary=(
                        "the model could not produce a valid strategy spec in "
                        f"{MAX_ATTEMPTS} attempts. Nothing was written."
                    ),
                ) from exc
            messages.append({"role": "assistant", "content": _echo(proposal)})
            messages.append({"role": "user", "content": _RETRY_PREAMBLE + str(exc)})
            continue

        questions = questions_for_untraceable(spec, words)
        if questions:
            return CompileRefusal(
                questions=questions,
                origin=FROM_TRACEABILITY,
                attempts=attempt,
                model=model,
            )
        return CompileResult(
            spec=spec,
            explanation=explain(spec),
            attempts=attempt,
            model=model,
        )

    raise AssertionError("the attempt loop always returns or raises")  # pragma: no cover


def _echo(proposal: SpecProposal) -> str:
    """What the model said, put back in the conversation before the correction.

    The retry has to show the model its own previous answer or the correction
    has nothing to attach to. This is a rendering of the payload the model
    already produced — no account state joins the conversation on a retry.
    """
    return json.dumps(proposal.payload, sort_keys=True, default=str)
