"""Natural language → strategy spec. The compiler translates; it never authors.

    from tick.compile import AnthropicSpecProposer, compile_text

    proposer = AnthropicSpecProposer.for_environment()   # the USER's key
    outcome = compile_text(words, proposer, model="claude-opus-5")

What this package is for, and what keeps it honest:

- **It produces a document, never an action.** The output is a `StrategySpec`
  — the same closed grammar a person can write by hand — gated by the same
  validators and run under the same cage. No model output reaches a broker
  without passing them (CLAUDE.md invariant 3).
- **It may use only what the user said.** The prompt states the rule; `trace`
  ENFORCES it, checking every symbol and every number in the compiled spec
  against the user's own words and returning a `CompileRefusal` with specific
  questions for anything it cannot trace. Tick names no security and proposes
  no parameter, here or anywhere.
- **The model account is the user's.** The key comes from the user's
  environment or from the caller, is never written to disk, and reaches only
  the provider's own SDK: Tick operates no model endpoint. Account state never
  enters a request — the conversation is the user's text and nothing else.
- **Explanations are rendered from the validated spec**, not written by the
  model, so what a person reads before approving cannot describe a rule the
  document does not contain.
"""

from __future__ import annotations

from .anthropic_client import API_KEY_ENV, AnthropicSpecProposer, read_reply
from .client import (
    Proposal,
    ProposalRequest,
    QuestionsProposal,
    SpecProposal,
    SpecProposer,
)
from .compiler import (
    DEFAULT_MODEL,
    FROM_MODEL,
    FROM_TRACEABILITY,
    MAX_ATTEMPTS,
    MAX_OUTPUT_TOKENS,
    CompileRefusal,
    CompileResult,
    compile_text,
)
from .errors import CompileError, MissingApiKey, ModelReplyError
from .explain import RuleExplanation, describe_action, describe_condition, explain
from .prompt import SYSTEM_PROMPT, user_messages
from .schema import (
    ASK_TOOL_NAME,
    EMIT_TOOL_NAME,
    TOOL_NAMES,
    ask_tool,
    emit_tool,
    spec_input_schema,
    tool_definitions,
)
from .trace import numbers_in, questions_for_untraceable, symbol_is_in

__all__ = [
    "API_KEY_ENV",
    "ASK_TOOL_NAME",
    "DEFAULT_MODEL",
    "EMIT_TOOL_NAME",
    "FROM_MODEL",
    "FROM_TRACEABILITY",
    "MAX_ATTEMPTS",
    "MAX_OUTPUT_TOKENS",
    "SYSTEM_PROMPT",
    "TOOL_NAMES",
    "AnthropicSpecProposer",
    "CompileError",
    "CompileRefusal",
    "CompileResult",
    "MissingApiKey",
    "ModelReplyError",
    "Proposal",
    "ProposalRequest",
    "QuestionsProposal",
    "RuleExplanation",
    "SpecProposal",
    "SpecProposer",
    "ask_tool",
    "compile_text",
    "describe_action",
    "describe_condition",
    "emit_tool",
    "explain",
    "numbers_in",
    "questions_for_untraceable",
    "read_reply",
    "spec_input_schema",
    "symbol_is_in",
    "tool_definitions",
    "user_messages",
]
