"""The grammar the model is handed, and the page of rules that comes with it.

Two claims live here. The schema the model must fill IS the spec's own pydantic
schema, so the shape it is allowed to emit cannot drift from the shape the
validators accept. And the system prompt is a constant that names no security
and proposes no strategy — Tick authors none, in a prompt least of all, because
a worked example in a prompt shapes every spec compiled afterwards.
"""

from __future__ import annotations

from tests.test_product_constraints import REAL_TICKERS
from tick.compile import (
    ASK_TOOL_NAME,
    EMIT_TOOL_NAME,
    SYSTEM_PROMPT,
    spec_input_schema,
    tool_definitions,
    user_messages,
)
from tick.spec import StrategySpec


def test_the_user_message_is_the_users_own_text_verbatim():
    words = "  buy 3 shares of XYZ when the price is below 10  "
    assert user_messages(words.strip()) == (
        {"role": "user", "content": "buy 3 shares of XYZ when the price is below 10"},
    )


def test_the_system_prompt_is_a_constant_and_names_no_security():
    assert REAL_TICKERS.findall(SYSTEM_PROMPT) == []
    assert "XYZ" in SYSTEM_PROMPT and "ABCD" in SYSTEM_PROMPT


def test_the_system_prompt_states_the_translation_rule_and_the_refusal_it_needs():
    assert "TRANSLATION RULE" in SYSTEM_PROMPT
    assert ASK_TOOL_NAME in SYSTEM_PROMPT
    assert "You may NOT" in SYSTEM_PROMPT


def test_the_system_prompt_states_that_the_runtime_is_long_only():
    assert "LONG ONLY" in SYSTEM_PROMPT
    assert "never open a short" in SYSTEM_PROMPT or "can never open a short" in SYSTEM_PROMPT


def test_the_emit_schema_is_generated_from_the_spec_models():
    schema = spec_input_schema()

    assert set(schema["properties"]) == set(StrategySpec.model_fields)
    assert schema["additionalProperties"] is False
    assert sorted(schema["required"]) == sorted(
        name for name, field in StrategySpec.model_fields.items() if field.is_required()
    )


def test_the_grammar_offered_to_the_model_has_no_short_side():
    """There is no short side in the spec, so there is none in the schema either."""
    schema = spec_input_schema()
    assert schema["$defs"]["Side"]["enum"] == ["buy", "sell"]

    def literals(node: object) -> list[str]:
        found: list[str] = []
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("enum", "const"):
                    found.extend(
                        str(item) for item in (value if isinstance(value, list) else [value])
                    )
                found.extend(literals(value))
        elif isinstance(node, list):
            for value in node:
                found.extend(literals(value))
        return found

    assert [value for value in literals(schema) if "short" in value.lower()] == []


def test_no_field_in_the_grammar_accepts_a_json_number():
    """A JSON number is a binary float by the time any SDK has parsed it."""
    schema = spec_input_schema()

    def numbers(node: object, path: str) -> list[str]:
        found: list[str] = []
        if isinstance(node, dict):
            if node.get("type") == "number":
                found.append(path)
            for key, value in node.items():
                found.extend(numbers(value, f"{path}.{key}"))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                found.extend(numbers(value, f"{path}[{index}]"))
        return found

    assert numbers(schema, "") == []


def test_both_tools_are_offered_every_time_so_refusing_is_always_reachable():
    """A model with no way to say "you did not tell me" has to invent something."""
    names = [tool["name"] for tool in tool_definitions()]
    assert names == [EMIT_TOOL_NAME, ASK_TOOL_NAME]
    ask = tool_definitions()[1]
    assert ask["input_schema"]["required"] == ["questions"]
    assert ask["input_schema"]["properties"]["questions"]["minItems"] == 1
