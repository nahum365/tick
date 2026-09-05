"""The shipped interview contains structure and no authored strategy content."""

from __future__ import annotations

import re
from dataclasses import MISSING, fields

from tick.interview import SLOTS, Slot

ADVICE = re.compile(
    r"\b(buy|sell|add|reduce|trim|exit|enter|hold|should|recommend|typically|most people)\b",
    re.IGNORECASE,
)
SYMBOL_SHAPED = re.compile(r"\b[A-Z]{1,5}(?:\.[A-Z]{1,2})?\b")


def test_every_slot_has_no_default_and_a_validator():
    for field in fields(Slot):
        assert field.default is MISSING
        assert field.default_factory is MISSING
    assert SLOTS
    assert all(callable(slot.validator) for slot in SLOTS)


def test_questions_have_no_number_symbol_or_advice_vocabulary():
    for slot in SLOTS:
        for text in (slot.question, slot.explains):
            assert not re.search(r"\d", text), slot.name
            assert not SYMBOL_SHAPED.search(text), slot.name
            assert not ADVICE.search(text), slot.name


def test_every_slot_explains_what_it_controls_without_a_value():
    """Guidance is structural: what the runtime does with the answer, never which answer."""
    for slot in SLOTS:
        assert len(slot.explains.split()) >= 8, slot.name
        assert slot.explains.endswith("."), slot.name
        assert "%" not in slot.explains and "$" not in slot.explains, slot.name


def test_every_slot_schema_carries_no_default():
    def walk(node):
        if isinstance(node, dict):
            assert "default" not in node
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    for slot in SLOTS:
        walk(slot.schema)
