"""The model agent's document, and the discriminator that tells the two apart.

A rule agent's document is a `StrategySpec`: a universe, a cadence, `when →
then` rules, and a cage. A model agent's document is this one: the same
universe, the same cadence, the same cage — and no rules, because the judgment
comes from a model instead. What it adds is the model id, pinned in the
document so that "which model decided this" is a property of the agent rather
than of whoever happened to run it.

**The two kinds are distinguished by a `kind` key, not by sniffing fields.**
`parse_agent_spec` reads `kind` and dispatches; a document with no `kind` is a
`StrategySpec`, exactly as every spec written before this file existed was.
That is why adopting model agents changed no existing document and no existing
`spec_id`: the discriminator is *absent* on the older kind, which is what
"absent when absence carries no fact" looks like on disk.

**The instructions are NOT in this document.** They live in a file the user
writes, `TICK_HOME/agents/<agent id>/instructions.md`, and the agent refuses to
run without one. Keeping them out of the spec is deliberate twice over: the
spec is the part Tick validates and hashes, and the instructions are the part
Tick has no opinion about at all — it ships none, proposes none, and completes
none.

**Everything the cage does to a rule agent it does to this one.** `Cage` is the
same required, all-fields-set model, `apply_cage` is the same function, and the
runtime applies it to a model's intents through code the model cannot call
(CLAUDE.md invariant 3).
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError, field_validator, model_validator

from tick.spec import (
    SYMBOL_PATTERN,
    Cadence,
    Cage,
    SpecFormatError,
    SpecModel,
    SpecValidationError,
    StrategySpec,
    parse_spec,
    spec_id,
)
from tick.spec.strategy import MAX_NAME_LENGTH

__all__ = [
    "MAX_MODEL_ID_LENGTH",
    "MODEL_AGENT_KIND",
    "AgentSpec",
    "ModelAgentSpec",
    "agent_spec_id",
    "dump_agent_spec",
    "is_model_agent",
    "load_agent_spec_file",
    "loads_agent_spec",
    "parse_agent_spec",
]

#: The value of `kind` that makes a document a model agent's. A document with
#: no `kind` at all is a `StrategySpec`, which is every spec written before
#: model agents existed.
MODEL_AGENT_KIND = "model_agent"

#: How long a model id may be. It is an identifier the user typed, recorded in
#: every decision this agent makes; a bound keeps a stray paragraph out of the
#: record.
MAX_MODEL_ID_LENGTH = 120


class ModelAgentSpec(SpecModel):
    """A caged model agent: a universe, a cadence, a cage, a provider and a model id.

    Every field is required, the cage most of all. The model has judgment and
    the cage has authority (CLAUDE.md invariant 3), so a cage carrying a value
    nobody chose would be the whole product's claim quietly weakened.
    """

    kind: Literal["model_agent"] = MODEL_AGENT_KIND
    name: str
    version: int
    universe: list[str]
    cadence: Cadence
    #: Which shipped adapter reaches the model: `anthropic` (the API, on your
    #: key) or `codex` (the Codex CLI, on your login). Pinned in the document so
    #: "how was this model reached" is a property of the agent, not of the shell
    #: it happened to run in.
    provider: Literal["anthropic", "codex"]
    #: The model that decides, e.g. `claude-opus-5`. Pinned in the document and
    #: written into every decision record, beside the id the provider reports
    #: having actually answered with.
    model: str
    cage: Cage

    @field_validator("name")
    @classmethod
    def _check_name(cls, value: str) -> str:
        if value != value.strip() or not value:
            raise ValueError("name must not be empty or padded with whitespace")
        if len(value) > MAX_NAME_LENGTH:
            raise ValueError(f"name must be at most {MAX_NAME_LENGTH} characters")
        if any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise ValueError("name must not contain control characters")
        return value

    @field_validator("model")
    @classmethod
    def _check_model(cls, value: str) -> str:
        if value != value.strip() or not value:
            raise ValueError(
                "model must name the model that decides; an agent whose record cannot "
                "say which model made a decision is not supervisable"
            )
        if len(value) > MAX_MODEL_ID_LENGTH:
            raise ValueError(f"model must be at most {MAX_MODEL_ID_LENGTH} characters")
        return value

    @field_validator("version")
    @classmethod
    def _check_version(cls, value: int) -> int:
        if value < 1:
            raise ValueError(f"version({value}): must be >= 1")
        return value

    @field_validator("universe")
    @classmethod
    def _check_universe(cls, value: list[str]) -> list[str]:
        """Same rule as a rule agent's universe: valid, unique, then sorted.

        The universe is the model's whole permitted world — an intent naming
        anything else is refused before it can be sized — so it is validated
        exactly as strictly here as it is there.
        """
        if not value:
            raise ValueError("universe must name at least one symbol")
        seen: set[str] = set()
        for symbol in value:
            if not SYMBOL_PATTERN.match(symbol):
                raise ValueError(
                    f"universe symbol {symbol!r}: must be 1-5 capital letters with an "
                    f"optional class suffix (e.g. 'ABCD', 'XYZ.B')"
                )
            if symbol in seen:
                raise ValueError(f"universe lists {symbol!r} more than once")
            seen.add(symbol)
        return sorted(value)

    @model_validator(mode="after")
    def _check(self) -> ModelAgentSpec:
        if self.kind != MODEL_AGENT_KIND:
            raise ValueError(f"kind must be {MODEL_AGENT_KIND!r}")
        return self


#: What an agent's document may be. Both run under the same cage, on the same
#: cadence floor, through the same broker port.
AgentSpec = StrategySpec | ModelAgentSpec


def is_model_agent(spec: AgentSpec) -> bool:
    """True when this agent's decisions come from a model rather than from rules."""
    return isinstance(spec, ModelAgentSpec)


def agent_spec_id(spec: AgentSpec) -> str:
    """The document's identity — the same hash, over either kind.

    One function so that an agent id means the same thing for both kinds: the
    sha256 of the canonical encoding of the document the user approved.
    """
    return spec_id(spec)


def parse_agent_spec(document: Mapping[str, Any], *, source: str | None = None) -> AgentSpec:
    """Validate a decoded JSON object as whichever kind of agent it declares."""
    if not isinstance(document, Mapping):
        raise SpecFormatError(f"an agent spec is a JSON object; got {type(document).__name__}")
    kind = document.get("kind")
    if kind is None:
        # A rule agent's document takes the loader it has had since slice 01,
        # error rendering included, rather than a second path that words its
        # complaints slightly differently.
        return parse_spec(document, source=source)
    if kind != MODEL_AGENT_KIND:
        raise SpecFormatError(
            f"{kind!r} is not an agent kind. A document with no 'kind' is a strategy "
            f"spec (a rule agent); {MODEL_AGENT_KIND!r} is a model-driven agent."
        )
    try:
        return ModelAgentSpec.model_validate(dict(document))
    except ValidationError as exc:
        problems: list[str] = []
        for error in exc.errors():
            where = ".".join(str(part) for part in error["loc"]) or "spec"
            message = str(error["msg"]).removeprefix("Value error, ")
            rendered = f"{where}: {message}"
            if rendered not in problems:
                problems.append(rendered)
        raise SpecValidationError(problems, source=source) from exc


def loads_agent_spec(text: str, *, source: str | None = None) -> AgentSpec:
    """Parse JSON text as an agent document of either kind. Numbers stay exact."""
    try:
        document = json.loads(text, parse_float=Decimal)
    except (json.JSONDecodeError, ValueError) as exc:
        where = f" in {source}" if source else ""
        raise SpecFormatError(f"could not read the spec as JSON{where}: {exc}") from exc
    return parse_agent_spec(document, source=source)


def load_agent_spec_file(path: str | os.PathLike[str]) -> AgentSpec:
    """Read and validate an agent document of either kind from a file."""
    resolved = Path(path)
    try:
        text = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        raise SpecFormatError(f"could not read agent spec file {resolved}: {exc}") from exc
    return loads_agent_spec(text, source=str(resolved))


def dump_agent_spec(spec: AgentSpec, *, indent: int = 2) -> str:
    """Render either kind as readable JSON that loads back identically."""
    payload = spec.model_dump(mode="json")
    return json.dumps(payload, sort_keys=True, indent=indent, ensure_ascii=False) + "\n"
