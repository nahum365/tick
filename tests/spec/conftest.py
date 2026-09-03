"""Shared fixture paths and helpers for the spec tests.

Nothing here touches the network, a broker, or `~/.tick`; the spec package is
pure data, and its tests read only the JSON documents in `tests/fixtures`.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "specs"
VALID_DIR = FIXTURES / "valid"
INVALID_DIR = FIXTURES / "invalid"


def valid_paths() -> list[Path]:
    return sorted(VALID_DIR.glob("*.json"))


def invalid_paths() -> list[Path]:
    return sorted(INVALID_DIR.glob("*.json"))


def read_document(path: Path) -> dict[str, Any]:
    """The raw JSON of a fixture, with numbers kept exact."""
    return json.loads(path.read_text(encoding="utf-8"), parse_float=Decimal)


@pytest.fixture
def simple_document() -> dict[str, Any]:
    """A minimal spec as a plain dict, for tests that mutate one field."""
    return {
        "name": "Minimal",
        "version": 1,
        "universe": ["AAPL"],
        "cadence": {"kind": "daily_close"},
        "rules": [
            {
                "id": "dip",
                "when": {
                    "kind": "compare",
                    "left": {"kind": "price"},
                    "op": "<",
                    "right": {"kind": "number", "value": "100.00"},
                },
                "then": {
                    "side": "buy",
                    "size": {"kind": "shares", "shares": 1},
                    "order_type": "market",
                },
            }
        ],
        "cage": {
            "max_position_pct": "10.00",
            "max_positions": 3,
            "max_order_notional": "500.00",
            "max_daily_drawdown_pct": "2.00",
            "allowed_session": "regular_hours",
        },
    }
