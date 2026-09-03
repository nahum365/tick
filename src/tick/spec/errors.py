"""Errors raised when a strategy spec cannot be read or cannot be trusted.

The spec is the deterministic contract between a human (or the compiler in
slice 05) and the runtime, so a rejection has to say what is wrong in words
the author can act on: `rule 'dip' references sma(0): n must be >= 1`, not a
stack of pydantic locations.
"""

from __future__ import annotations

from collections.abc import Iterable


class SpecError(Exception):
    """Base class for every spec failure."""


class SpecFormatError(SpecError):
    """The bytes handed to the loader are not a JSON object at all."""


class SpecValidationError(SpecError):
    """A JSON object was read but it is not a valid strategy spec.

    Carries every problem found, already rendered for a human. `problems` is
    the machine-readable form; `str(exc)` is the report.
    """

    def __init__(self, problems: Iterable[str], *, source: str | None = None) -> None:
        self.problems: tuple[str, ...] = tuple(problems)
        self.source = source
        if not self.problems:  # pragma: no cover - defensive
            raise ValueError("SpecValidationError needs at least one problem")
        where = f" in {source}" if source else ""
        count = len(self.problems)
        noun = "problem" if count == 1 else "problems"
        body = "\n".join(f"  - {problem}" for problem in self.problems)
        super().__init__(f"invalid strategy spec{where} ({count} {noun}):\n{body}")
