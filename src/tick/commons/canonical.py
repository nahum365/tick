"""Canonical JSON shared by the box signer and the hosted verifier.

The implementation lives on the box side so the service depends on the public
wire contract, never the reverse. Exact decimals stay strings, binary floats
are refused, and aware datetimes are rendered in one UTC spelling before a
signature or content identity is computed.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel

__all__ = ["canonical_bytes", "canonical_hash", "canonical_json", "normalize"]


def normalize(value: Any) -> Any:
    """Return JSON primitives without losing decimal or datetime meaning."""
    if isinstance(value, BaseModel):
        return normalize(value.model_dump(mode="python"))
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(
                "a commons datetime has no timezone; provide an aware datetime and retry"
            )
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, float):
        raise ValueError(
            "a commons value is a binary float; provide an exact decimal string and retry"
        )
    if isinstance(value, dict):
        return {str(key): normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"{type(value).__name__} is not a commons JSON value; use JSON values only")


def canonical_json(value: Any) -> str:
    """Encode a wire value with sorted keys and no insignificant whitespace."""
    return json.dumps(normalize(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_bytes(value: Any) -> bytes:
    """Return the exact bytes signed by contributors and checked by the gate."""
    return canonical_json(value).encode("utf-8")


def canonical_hash(value: Any) -> str:
    """Name an immutable commons value by its canonical SHA-256 digest."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()
