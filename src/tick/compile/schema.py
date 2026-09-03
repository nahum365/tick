"""The two tools the model is offered, both derived from the spec models.

The model never writes free-form JSON. It is offered exactly two tools and
forced to call one of them:

- `emit_strategy_spec`, whose input schema IS `StrategySpec.model_json_schema()`
  — the closed grammar, generated from the pydantic models so it cannot drift
  from what the validators accept; and
- `ask_for_missing_details`, whose input is a list of questions to put to the
  user.

The second tool is the whole translation rule made mechanical. A model with no
way to say "you did not tell me" will invent something, because a required
field has to be filled; giving it a refusal to reach for is how "translate,
never author" becomes a shape rather than an instruction.

One transformation is applied to the generated schema. Pydantic renders a
`Decimal` field as `number` OR `string`, and a JSON `number` is a binary float
by the time any SDK has parsed it — the exact thing `spec.base` refuses. The
`number` branch is removed here so the grammar the model is handed only allows
the exact form, rather than relying on a sentence in the prompt.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from tick.spec import StrategySpec

__all__ = [
    "ASK_TOOL_NAME",
    "EMIT_TOOL_NAME",
    "TOOL_NAMES",
    "ask_tool",
    "emit_tool",
    "spec_input_schema",
    "tool_definitions",
]

EMIT_TOOL_NAME = "emit_strategy_spec"
ASK_TOOL_NAME = "ask_for_missing_details"

#: Every tool name the compiler will accept in a reply. A call to anything
#: else is an unreadable reply, not a spec.
TOOL_NAMES: frozenset[str] = frozenset({EMIT_TOOL_NAME, ASK_TOOL_NAME})

_EMIT_DESCRIPTION = (
    "Emit the strategy spec that translates the user's words. Use only the "
    "securities the user named and only the numbers the user supplied. If any "
    "required field has no number in the user's words, do NOT fill it in: call "
    f"{ASK_TOOL_NAME} instead."
)

_ASK_DESCRIPTION = (
    "Refuse to compile, and list the specific questions the user must answer "
    "first. Use this whenever the user's words do not name the securities to "
    "trade, or do not supply a threshold, a size, or one of the cage limits. "
    "Ask about what is missing; never propose an instrument, a threshold, or a "
    "limit of your own."
)


def _is_decimal_union(node: dict[str, Any]) -> bool:
    """True for pydantic's rendering of a `Decimal` field: number OR string."""
    branches = node.get("anyOf")
    if not isinstance(branches, list) or len(branches) != 2:
        return False
    kinds = {branch.get("type") for branch in branches if isinstance(branch, dict)}
    return kinds == {"number", "string"}


def _string_branch(node: dict[str, Any]) -> dict[str, Any]:
    for branch in node["anyOf"]:
        if isinstance(branch, dict) and branch.get("type") == "string":
            return dict(branch)
    raise AssertionError("_is_decimal_union promised a string branch")  # pragma: no cover


def _exact_decimals(node: Any) -> Any:
    """Rewrite every `Decimal` union into its string form, in place, recursively."""
    if isinstance(node, dict):
        if _is_decimal_union(node):
            replacement = _string_branch(node)
            title = node.get("title")
            node.clear()
            node.update(replacement)
            if title is not None:
                node["title"] = title
            node["description"] = (
                'An exact decimal written as a JSON string, e.g. "12.50". Never a '
                "JSON number: a binary float is not the number the user wrote."
            )
            return node
        for value in node.values():
            _exact_decimals(value)
        return node
    if isinstance(node, list):
        for value in node:
            _exact_decimals(value)
    return node


def spec_input_schema() -> dict[str, Any]:
    """The `StrategySpec` JSON schema, as the model is allowed to see it.

    Generated from the pydantic models on every call — a schema cached in a
    literal would be a second copy of the grammar, free to drift from the one
    the validators enforce.
    """
    schema = deepcopy(StrategySpec.model_json_schema())
    return _exact_decimals(schema)


def emit_tool() -> dict[str, Any]:
    """The spec-emitting tool definition."""
    return {
        "name": EMIT_TOOL_NAME,
        "description": _EMIT_DESCRIPTION,
        "input_schema": spec_input_schema(),
    }


def ask_tool() -> dict[str, Any]:
    """The refusal tool definition — the compiler's way of asking, not guessing."""
    return {
        "name": ASK_TOOL_NAME,
        "description": _ASK_DESCRIPTION,
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "questions": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string"},
                    "description": (
                        "One specific question per missing element, in the user's "
                        "own terms. For example: 'Which symbols should this trade?' "
                        "or 'What position cap do you want?'"
                    ),
                }
            },
            "required": ["questions"],
        },
    }


def tool_definitions() -> tuple[dict[str, Any], ...]:
    """Both tools, in a stable order (the order is part of the cache prefix)."""
    return (emit_tool(), ask_tool())
