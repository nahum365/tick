"""The notification grammar — the only sentences Tick is allowed to send.

Notifications are the one surface where the product speaks to a person about
their money, so it speaks in templates. Four of them, all mechanical past
tense, all naming the rule that fired and nothing else:

    Your rule 'dip-buyer' fired: bought 12 XYZ at $184.20 — simulated.
    Your rule 'dip-buyer' fired but the order was rejected: … — simulated.
    Your rule 'dip-buyer' fired but the notification text was withheld; … — simulated.
    Tick stopped agent 'a1b2c3': … — simulated.

**A simulated run always says so.** The mode tag is not optional and not a
default: every composer takes a `Mode` and appends `— simulated.` or
`— live.`, so there is no path that produces an untagged sentence (CLAUDE.md
invariant 6).

**Forbidden phrases are refused, not filtered.** `found`, `opportunity`,
`should`, `recommend` and `consider` are the vocabulary of a recommendation,
and we are not a registered adviser: a mined-and-pushed suggestion is a
recommendation under FINRA NTM 01-23 whatever it is called. The reason
fragments composed in here come from Tick's own engine and broker, so a
forbidden phrase in one is a bug — which is why `compose` raises
`NotificationRefused` rather than quietly rewriting the sentence. The runner
catches that, sends `withheld` instead, and records why: the evidence stays in
the record and the sentence that would have carried the phrase is never sent.

**A model agent's sentences name the model, and never a rule.** A
deterministic spec agent is a rule agent and its sentence says "your rule
fired"; a model-driven agent's sentence says "your model agent (claude-opus-5)"
and shows the model id, because the two are different things and describing one
as the other is the naming the audit rules out. The tense, the tag and the
forbidden vocabulary are identical — there is one `compose`, and everything
goes through it.

**Only fired rules, plus one exception that is stated.** Invariant 6 says
notifications carry fired rules. `run_stopped` is the one sentence that is not
about a rule, and it exists because invariant 8's fail-safe requires the
runtime to tell the user it stopped — a silent halt is the failure the user
most needs to hear about. It is still mechanical past tense and still tagged.
"""

from __future__ import annotations

import re

from .errors import NotificationRefused
from .modes import Mode

__all__ = [
    "FORBIDDEN_PHRASES",
    "compose",
    "fired_filled",
    "fired_not_placed",
    "model_filled",
    "model_not_placed",
    "model_withheld",
    "run_stopped",
    "withheld",
]

#: Words the product may not say about somebody's positions. Matched as whole
#: words with their obvious inflections, case-insensitively. The match
#: deliberately over-blocks (`considerate` trips `consider`): a notification
#: that is refused is a bug someone fixes, a notification that advises is not.
FORBIDDEN_PHRASES: tuple[str, ...] = ("found", "opportunity", "should", "recommend", "consider")

_FORBIDDEN = re.compile(
    r"\b(found|opportunit(?:y|ies)|should|shouldn't|recommend\w*|consider\w*)\b",
    re.IGNORECASE,
)

#: How long a reason may be before a notification truncates it. The whole
#: reason is always in the record; the notification is a push message.
MAX_REASON_LENGTH = 240


def compose(body: str, mode: Mode) -> str:
    """Tag `body` with the mode and refuse it if it says something we may not.

    Every sentence in this module goes through here, so there is exactly one
    place a notification acquires its `— simulated.` / `— live.` tag and
    exactly one place the vocabulary is enforced.
    """
    sentence = f"{body} — {Mode(mode).tag}."
    match = _FORBIDDEN.search(sentence)
    if match is not None:
        raise NotificationRefused(
            f"a notification would have said {match.group(0)!r}, which is the "
            f"vocabulary of a recommendation. Tick states what happened and names "
            f"the rule; it never advises. The sentence was not sent: {sentence!r}"
        )
    return sentence


def _reason(text: str) -> str:
    """One line, trimmed to a length a push notification can carry."""
    collapsed = " ".join(text.split())
    if not collapsed:
        raise ValueError("a notification that reports a refusal must carry the reason")
    if len(collapsed) > MAX_REASON_LENGTH:
        return collapsed[: MAX_REASON_LENGTH - 1].rstrip() + "…"
    return collapsed


def _rule(rule_id: str) -> str:
    if not rule_id.strip():
        raise ValueError("a notification names the rule that fired")
    return rule_id


def fired_filled(*, rule_id: str, description: str, mode: Mode) -> str:
    """A rule fired and the order executed.

    `description` is the broker's own past-tense phrase for the fill
    (`bought 12 XYZ at $184.20`), so the number in the sentence is the number
    that was traded — never the estimate the intent was sized at.
    """
    return compose(f"Your rule {_rule(rule_id)!r} fired: {_reason(description)}", mode)


def fired_not_placed(*, rule_id: str, reason: str, mode: Mode) -> str:
    """A rule fired and no order resulted — refused, caged, declined or rejected."""
    return compose(
        f"Your rule {_rule(rule_id)!r} fired but the order was rejected: {_reason(reason)}",
        mode,
    )


def withheld(*, rule_id: str, mode: Mode) -> str:
    """The fallback when a composed sentence carried a phrase we may not send.

    It says less, and it says that it says less. The full reason is in the
    record, which is the copy that matters.
    """
    return compose(
        f"Your rule {_rule(rule_id)!r} fired but the notification text was withheld; "
        f"the record carries what happened",
        mode,
    )


def _model(model_id: str) -> str:
    if not model_id.strip():
        raise ValueError("a model agent's notification names the model that decided")
    return model_id


def model_filled(*, model: str, description: str, mode: Mode) -> str:
    """A model agent's order executed. The model id is shown, always.

    `description` is the broker's own past-tense phrase for the fill, so the
    number in the sentence is the number that was traded.
    """
    return compose(f"Your model agent ({_model(model)}) {_reason(description)}", mode)


def model_not_placed(*, model: str, reason: str, mode: Mode) -> str:
    """A model agent proposed an order and no order resulted."""
    return compose(
        f"Your model agent ({_model(model)}) proposed an order that was rejected: "
        f"{_reason(reason)}",
        mode,
    )


def model_withheld(*, model: str, mode: Mode) -> str:
    """The model-agent fallback when a composed sentence carried a phrase we may not send."""
    return compose(
        f"Your model agent ({_model(model)}) acted but the notification text was "
        f"withheld; the record carries what happened",
        mode,
    )


def run_stopped(*, agent_id: str, reason: str, mode: Mode) -> str:
    """The runtime halted this run. Sent on every stop, including failures."""
    if not agent_id.strip():
        raise ValueError("a stop notification names the agent that stopped")
    return compose(f"Tick stopped agent {agent_id!r}: {_reason(reason)}", mode)
