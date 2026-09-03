"""Reading and writing strategy specs, with validation errors a human can act on.

Two things happen here that are easy to get wrong and expensive to get wrong.

**JSON numbers are parsed as `Decimal`.** `json.loads` turns `12.10` into a
binary float, and a float that reaches a money field has already lost the
number the author wrote. `parse_float=Decimal` keeps it exact from the first
byte, and `base.ExactDecimal` refuses a float arriving any other way.

**Pydantic's error locations are translated into sentences.** A spec is a
document a person wrote (or reviewed after the compiler wrote it), so a
rejection reads `rule 'dip' references sma(0): n must be >= 1`.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .conditions import CONDITION_KINDS
from .errors import SpecFormatError, SpecValidationError
from .indicators import OPERAND_KINDS
from .strategy import CADENCE_KINDS, SIZE_KINDS, StrategySpec

#: Every discriminator VALUE that can appear in an error location. They are
#: structural, not fields the author wrote, so they are dropped from the
#: rendered path.
_TAG_SEGMENTS: frozenset[str] = OPERAND_KINDS | CONDITION_KINDS | SIZE_KINDS | CADENCE_KINDS

#: Messages authored in this package that already name their own location.
#: Prefixing them again would read "rules[0]: rule 'dip': ...".
_SELF_LOCATING = ("rule ", "rules ", "universe ", "cage.", "name ", "version(", "a spec ")

#: How deeply the RAW document may nest before we refuse to hand it to the
#: parser. Guards the recursive descent itself; the reviewable limit on
#: conditions is `MAX_CONDITION_DEPTH`, checked after parsing.
MAX_RAW_DEPTH = 64

_PYDANTIC_NOISE = ("Value error, ", "Assertion failed, ")


def _clean_message(message: str) -> str:
    for noise in _PYDANTIC_NOISE:
        if message.startswith(noise):
            return message[len(noise) :]
    return message


def _render_path(segments: tuple[Any, ...]) -> str:
    """Join a location into `then.size` / `of[1].left`, dropping union tags."""
    rendered = ""
    for segment in segments:
        if isinstance(segment, int):
            rendered += f"[{segment}]"
            continue
        if segment in _TAG_SEGMENTS:
            continue
        rendered = f"{rendered}.{segment}" if rendered else str(segment)
    return rendered


def _rule_name(raw: Any, index: int) -> str:
    """`rule 'dip'` when the raw document gives that rule an id we can quote."""
    if isinstance(raw, Mapping):
        rules = raw.get("rules")
        if isinstance(rules, list) and 0 <= index < len(rules):
            rule = rules[index]
            if isinstance(rule, Mapping):
                rule_id = rule.get("id")
                if isinstance(rule_id, str) and rule_id:
                    return f"rule {rule_id!r}"
    return f"rules[{index}]"


def _render_problem(error: Mapping[str, Any], raw: Any) -> str:
    message = _clean_message(str(error["msg"]))
    loc: tuple[Any, ...] = tuple(error["loc"])
    if not loc or message.startswith(_SELF_LOCATING):
        return message

    if loc[0] == "rules" and len(loc) >= 2 and isinstance(loc[1], int):
        name = _rule_name(raw, loc[1])
        rest = loc[2:]
        if rest and rest[-1] in OPERAND_KINDS:
            return f"{name} references {message}"
        path = _render_path(rest)
        return f"{name} {path}: {message}" if path else f"{name}: {message}"

    path = _render_path(loc)
    return f"{path}: {message}" if path else message


def _format_problems(error: ValidationError, raw: Any) -> list[str]:
    problems: list[str] = []
    for entry in error.errors():
        rendered = _render_problem(entry, raw)
        if rendered not in problems:
            problems.append(rendered)
    return problems


def _guard_raw_depth(document: Any) -> None:
    """Refuse absurdly nested input before the recursive parser sees it."""
    stack: list[tuple[Any, int]] = [(document, 1)]
    while stack:
        node, depth = stack.pop()
        if depth > MAX_RAW_DEPTH:
            raise SpecFormatError(
                f"spec document nests deeper than {MAX_RAW_DEPTH} levels; it is not a strategy spec"
            )
        if isinstance(node, Mapping):
            stack.extend((value, depth + 1) for value in node.values())
        elif isinstance(node, list):
            stack.extend((value, depth + 1) for value in node)


def parse_spec(document: Mapping[str, Any], *, source: str | None = None) -> StrategySpec:
    """Validate an already-decoded JSON object into a `StrategySpec`."""
    if not isinstance(document, Mapping):
        raise SpecFormatError(f"a strategy spec is a JSON object; got {type(document).__name__}")
    _guard_raw_depth(document)
    try:
        return StrategySpec.model_validate(dict(document))
    except ValidationError as exc:
        raise SpecValidationError(_format_problems(exc, document), source=source) from exc


def loads_spec(text: str, *, source: str | None = None) -> StrategySpec:
    """Parse JSON text into a `StrategySpec`. Numbers stay exact."""
    try:
        document = json.loads(text, parse_float=Decimal)
    except (json.JSONDecodeError, InvalidOperation) as exc:
        where = f" in {source}" if source else ""
        raise SpecFormatError(f"could not read the spec as JSON{where}: {exc}") from exc
    if not isinstance(document, Mapping):
        raise SpecFormatError(f"a strategy spec is a JSON object; got {type(document).__name__}")
    return parse_spec(document, source=source)


def load_spec_file(path: str | os.PathLike[str]) -> StrategySpec:
    """Read and validate a spec from a file. The path is used in error text."""
    resolved = Path(path)
    try:
        text = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        raise SpecFormatError(f"could not read spec file {resolved}: {exc}") from exc
    return loads_spec(text, source=str(resolved))


def load_spec(source: str | os.PathLike[str]) -> StrategySpec:
    """Load a spec from a path, or from a JSON document given as text.

    A `Path` is always a file. A `str` is JSON when it starts with `{` and a
    filesystem path otherwise — the two are never ambiguous because a spec is
    always a JSON object, never a bare scalar. Callers that want no sniffing
    at all should use `load_spec_file` or `loads_spec` directly.
    """
    if isinstance(source, str):
        if source.lstrip().startswith("{"):
            return loads_spec(source)
        return load_spec_file(source)
    if isinstance(source, os.PathLike):
        return load_spec_file(source)
    raise TypeError(f"load_spec takes a path or JSON text, not {type(source).__name__}")


def dump_spec(spec: StrategySpec, *, indent: int = 2) -> str:
    """Render a spec as readable JSON that loads back identically.

    Keys are sorted so a file rewritten by the runtime produces no spurious
    diff. This is the human-facing form; `canonical_json` is the one that gets
    hashed.
    """
    payload = spec.model_dump(mode="json")
    return json.dumps(payload, sort_keys=True, indent=indent, ensure_ascii=False) + "\n"
