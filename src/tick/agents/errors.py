"""What can go wrong between a model agent and a tick, as types callers act on.

Every one of these STOPS the tick that hit it. None of them is recovered from
by asking the model again: a second call on the user's money after a reply
nobody could read is a machine arguing with itself, and a retry after a
provider refusal is Tick pushing back on a decision that was not Tick's.

They are deliberately not `Refusal` values. A refusal is a fact about ONE
proposed order — the symbol is outside the universe, there is no quote, the
sell is larger than the position — and it is carried forward, recorded and
reported per order. These are facts about whether the agent may act at all.
"""

from __future__ import annotations

__all__ = [
    "InstructionsMissing",
    "MissingApiKey",
    "ModelAgentError",
    "ModelReplyError",
    "ProviderUnavailable",
]


class ModelAgentError(Exception):
    """Base for every failure of a model-driven agent."""


class ModelReplyError(ModelAgentError):
    """The model's reply was not the one tool call it was offered.

    Prose instead of a tool call, an unknown tool, a provider-side refusal, or
    a reply that does not say which model produced it. Nothing is parsed,
    nothing is placed, and the tick stops — a decision record that cannot name
    the model that made the decision is not a record.
    """


class ProviderUnavailable(ModelAgentError):
    """The provider the document pins cannot be reached from this machine.

    Model agents are bring-your-own: the key, the login and the binary are the
    user's. Tick stores none of them and operates no model endpoint, so there
    is nothing to fall back to — and no other provider is substituted, because
    the record names the one the document pinned.
    """


class MissingApiKey(ProviderUnavailable):
    """No model API key, and none in the environment."""


class InstructionsMissing(ModelAgentError):
    """The agent's own instructions file is absent or empty.

    Tick ships no default instructions — no starter strategy, no selection
    heuristic, no example. A model agent with no user-authored instructions
    has nothing to run, and inventing something for it to run is the one thing
    this product may not do.
    """
