"""Turning what happened into something that can be hashed exactly.

A record's payload is JSON, and the hash is computed over its canonical
encoding, so the payload has to be *plain* before it is written: no objects
with a repr, no binary floats, no naive datetimes, no dict keys that are not
strings. `normalize_payload` is the one door in, and it refuses rather than
coerces wherever a coercion would change a number.

The three rules that carry invariants:

- **`Decimal` becomes its exact string.** `Decimal("12.50")` is `"12.50"`, not
  `12.5` and not `12.500000000000000444089209850062616169452667236328125`. A
  reader gets the number back with `Decimal(value)`, unchanged.
- **A binary `float` is refused, never rounded.** It is the fabrication
  invariant (5) at the serialisation boundary: rounding one here would put a
  number in the permanent record that nobody computed.
- **A datetime must be timezone-aware.** A naive datetime is a time in an
  unstated zone, which in a record is an unanswerable question later.

Pydantic models are accepted and dumped in JSON mode, so a `Decision`, an
`OrderIntent` or a `Fill` can be recorded directly. The dump is then walked by
the same rules, which is why a model that happens to hold a float is refused
too rather than trusted because it came from a model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel

from .errors import PayloadError

__all__ = ["canonical_timestamp", "normalize_payload", "normalize_value"]

#: How deep a payload may nest before we refuse to walk it. A record is a
#: statement about one thing that happened; anything deeper than this is a
#: data structure that wandered in, and the recursion guard is cheaper than
#: the stack overflow.
MAX_PAYLOAD_DEPTH = 32


def canonical_timestamp(moment: datetime) -> str:
    """The one string form of a moment in the record: ISO 8601 with an offset.

    Timezone-aware only. The record's hash is computed over this string, so the
    form is part of the record: a file rewritten with `Z` in place of `+00:00`
    is the same instant and a different document, and it will not verify. That
    is what tamper-evidence means — the ledger notices any rewriting, including
    a well-meaning one.
    """
    if not isinstance(moment, datetime):
        raise PayloadError(f"a record timestamp is a datetime, not {type(moment).__name__}")
    if moment.tzinfo is None or moment.tzinfo.utcoffset(moment) is None:
        raise PayloadError(
            f"the timestamp {moment.isoformat()} has no timezone; a record of when "
            f"something happened cannot be written in an unstated zone"
        )
    return moment.isoformat()


def normalize_value(value: Any, *, where: str = "payload", _depth: int = 0) -> Any:
    """Convert one value into its exact JSON form, or refuse to.

    `where` names the position for the error message — `payload.intent.qty` —
    because a refusal a caller cannot locate is a refusal they cannot fix.
    """
    if _depth > MAX_PAYLOAD_DEPTH:
        raise PayloadError(
            f"{where}: a payload nests deeper than {MAX_PAYLOAD_DEPTH} levels; "
            f"a record is a statement about one thing that happened"
        )
    if isinstance(value, BaseModel):
        return normalize_value(value.model_dump(mode="json"), where=where, _depth=_depth)
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, float):
        raise PayloadError(
            f"{where}: {value!r} is a binary float. Record the exact number "
            f'instead (a Decimal, or a string like "12.50"); rounding it here '
            f"would put a number nobody computed in the permanent record"
        )
    if isinstance(value, int):
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise PayloadError(f"{where}: {value} is not a finite number")
        return str(value)
    if isinstance(value, datetime):
        return canonical_timestamp(value)
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise PayloadError(
                    f"{where}: the key {key!r} is a {type(key).__name__}; JSON object "
                    f"keys are strings, and converting it would rename the field"
                )
            normalized[key] = normalize_value(item, where=f"{where}.{key}", _depth=_depth + 1)
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [
            normalize_value(item, where=f"{where}[{index}]", _depth=_depth + 1)
            for index, item in enumerate(value)
        ]
    raise PayloadError(
        f"{where}: a {type(value).__name__} has no exact JSON form. Record the "
        f"fields you mean, not the object"
    )


def normalize_payload(payload: Mapping[str, Any], *, where: str = "payload") -> dict[str, Any]:
    """Normalize a whole payload. A payload is always a JSON object.

    Not a list and not a scalar: a record's payload names its parts, so that a
    reader two years from now can tell what `"184.20"` was the price *of*.
    """
    if not isinstance(payload, Mapping):
        raise PayloadError(
            f"{where}: a record payload is a JSON object with named fields, "
            f"not a {type(payload).__name__}"
        )
    return normalize_value(dict(payload), where=where)
