"""The one tool a model agent is offered, and the strict shape of its answer.

The model does not write free-form text that Tick then interprets. It is
offered exactly one tool and forced to call it, so a reply is either a list of
intents in this schema or an unreadable reply that stops the tick.

    {"intents": [{"symbol": "XYZ", "side": "buy", "qty": 3, "reason": "..."}]}

**This file is the whole of Tick's contribution to what the model produces.**
The descriptions below are mechanical — what a field is, what shape it takes,
which values are legal. There is no guidance about *when* to buy, no notion of
a good trade, no ranking, no example strategy, and no worked case. Everything
about what to do lives in the user's own instructions file, which Tick neither
writes nor completes.

**An empty list is a first-class answer.** A model with no way to say "nothing
today" will find something to do, because a required field has to be filled.
`minItems` is deliberately absent and the description says so.

**`reason` is required and is evidence, not prose for the user.** It goes into
the decision record beside the intent, so that a fill six weeks later can be
read against what the model said it was doing. It is never shown as advice.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "EMIT_TOOL_NAME",
    "MAX_INTENTS",
    "MAX_REASON_LENGTH",
    "TOOL_NAMES",
    "emit_tool",
    "intents_schema",
    "tool_definitions",
]

EMIT_TOOL_NAME = "emit_order_intents"

#: Every tool name a reply may call. A call to anything else is an unreadable
#: reply, not a decision.
TOOL_NAMES: frozenset[str] = frozenset({EMIT_TOOL_NAME})

#: How many intents one tick may carry. A bound, not a target: a reply with a
#: hundred orders in it is a runaway, and the cage would reject most of them
#: one at a time while the broker saw every attempt.
MAX_INTENTS = 20

#: How long an intent's `reason` may be. It is a line in the record.
MAX_REASON_LENGTH = 500

_DESCRIPTION = (
    "Emit the order intents your instructions call for at this moment, given "
    "the snapshot. Emit an empty list to place nothing. Every intent is checked "
    "against the cage by the runtime after you answer; an intent that breaks a "
    "limit is rejected whole and recorded, never reduced to one that fits."
)


def intents_schema() -> dict[str, Any]:
    """The input schema of the one tool. Closed, and closed all the way down."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["intents"],
        "properties": {
            "intents": {
                "type": "array",
                "maxItems": MAX_INTENTS,
                "description": (
                    "The orders to place now. An empty list means place nothing, "
                    "and is a complete answer."
                ),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["symbol", "side", "qty", "reason"],
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": (
                                "One of the symbols in the snapshot's universe. Any "
                                "other symbol is refused by the runtime and recorded."
                            ),
                        },
                        "side": {
                            "type": "string",
                            "enum": ["buy", "sell"],
                            "description": (
                                "buy opens or adds to a long position; sell closes one "
                                "the account already holds. There is no short side: a "
                                "sell larger than the held quantity is refused whole."
                            ),
                        },
                        "qty": {
                            "type": "integer",
                            "minimum": 1,
                            "description": (
                                "A whole number of shares. Fractional shares do not "
                                "exist in this runtime."
                            ),
                        },
                        "reason": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": MAX_REASON_LENGTH,
                            "description": (
                                "Why you are proposing this order, in one line. It is "
                                "written into the append-only record beside the intent."
                            ),
                        },
                    },
                },
            }
        },
    }


def emit_tool() -> dict[str, Any]:
    """The one tool definition, built fresh so it cannot drift from the schema."""
    return {
        "name": EMIT_TOOL_NAME,
        "description": _DESCRIPTION,
        "input_schema": intents_schema(),
    }


def tool_definitions() -> tuple[dict[str, Any], ...]:
    """Every tool offered. There is one."""
    return (emit_tool(),)
