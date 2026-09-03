"""Canonical serialisation and the spec id — the immutability primitive.

`spec_id` is what makes an agent's record mean something. The record chains
against it, notifications quote it, and `tick run --live` refuses to run a spec
whose id is not the one the user approved. So the encoding has to be total and
stable: same spec, same id, on any machine, whatever order the author's JSON
happened to be written in.

Two properties do that work:

- **Sorted keys, no whitespace.** Reformatting a spec file cannot change its
  id, because the id is computed from the model, re-encoded canonically.
- **Decimals as strings, exactly as written.** `12.50` stays `"12.50"`. No
  binary float ever touches the encoding, so no rounding can enter it.

Note that `Decimal("1.5")` and `Decimal("1.50")` are equal numbers but
different spec ids. That is deliberate: the id names the document, and the
document is what a human approved.

The four spec-shaped helpers take a `SpecModel` rather than a `StrategySpec`.
A model agent's document (`tick.agents.spec`) is the same kind of frozen,
closed, exact-decimal document and gets its identity the same way, so widening
the annotation is what keeps ONE definition of "the id of the document a person
approved" rather than a second one that could drift.

`canonical_dumps`, `canonical_encode` and `sha256_hex` are the primitives
underneath, and they are shared rather than copied. `tick.records` hashes its
chain with exactly these functions, so "the encoding a spec id is computed
from" and "the encoding a record's hash is computed from" cannot drift apart
into two subtly different notions of canonical — the drift that would leave one
of the two hashes no longer meaning what its docstring says.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .base import SpecModel

__all__ = [
    "canonical_bytes",
    "canonical_dumps",
    "canonical_encode",
    "canonical_json",
    "sha256_hex",
    "short_spec_id",
    "spec_id",
]


def canonical_dumps(payload: Any) -> str:
    """The one true JSON encoding of an already-plain value.

    Sorted keys, no whitespace, no ASCII escaping. The input must already be
    JSON-primitive — `str`, `int`, `bool`, `None`, lists, and dicts with string
    keys — because this function's job is to encode, not to decide what a value
    means. Callers hand it the output of `model_dump(mode="json")` (specs) or
    of `tick.records.normalize_payload` (records); both turn `Decimal` into an
    exact string and refuse a binary float before anything reaches here.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_encode(payload: Any) -> bytes:
    """`canonical_dumps` as UTF-8 bytes — what actually gets hashed."""
    return canonical_dumps(payload).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    """SHA-256 of `data`, lowercase hex. The one hash function in the product."""
    return hashlib.sha256(data).hexdigest()


def canonical_json(spec: SpecModel) -> str:
    """The one true JSON encoding of a spec: sorted keys, compact, exact."""
    return canonical_dumps(spec.model_dump(mode="json"))


def canonical_bytes(spec: SpecModel) -> bytes:
    """`canonical_json` as UTF-8 bytes — what actually gets hashed."""
    return canonical_encode(spec.model_dump(mode="json"))


def spec_id(spec: SpecModel) -> str:
    """SHA-256 of the canonical encoding, lowercase hex."""
    return sha256_hex(canonical_bytes(spec))


def short_spec_id(spec: SpecModel) -> str:
    """First 12 hex characters of `spec_id`, for display only.

    Never use it as an identity check; it is a label for humans reading a
    notification, and the full id is what the runtime compares.
    """
    return spec_id(spec)[:12]
