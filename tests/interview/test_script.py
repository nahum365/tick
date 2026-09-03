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
        assert not re.search(r"\d", slot.question), slot.name
        assert not SYMBOL_SHAPED.search(slot.question), slot.name
        assert not ADVICE.search(slot.question), slot.name


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
