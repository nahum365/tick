"""What can go wrong between a person's words and a validated spec.

A refusal is NOT an error: the compiler refusing to invent a threshold is the
compiler working, and it comes back as a `CompileRefusal` result carrying the
questions to ask. The exceptions here are for the cases where no answer of any
kind could be produced.
"""

from __future__ import annotations

from collections.abc import Iterable

__all__ = ["CompileError", "MissingApiKey", "ModelReplyError"]


class CompileError(Exception):
    """The compiler could not produce a spec, and no question would help.

    Carries every problem it hit, already rendered for a human. The double
    validation failure raises this with BOTH attempts' problems, because the
    second error alone rarely explains what the model kept doing wrong.
    """

    def __init__(self, problems: Iterable[str], *, summary: str) -> None:
        self.problems: tuple[str, ...] = tuple(problems)
        if not self.problems:  # pragma: no cover - defensive
            raise ValueError("CompileError needs at least one problem")
        self.summary = summary
        body = "\n".join(f"  - {problem}" for problem in self.problems)
        super().__init__(f"{summary}\n{body}")


class ModelReplyError(CompileError):
    """The model's reply was not one of the two tool calls it was offered.

    Text instead of a tool call, an unknown tool name, or a safety refusal from
    the provider. Nothing is parsed and nothing is written.
    """


class MissingApiKey(CompileError):
    """No API key was supplied, and none is in the environment.

    Tick never stores a key: the compiler reads one from the environment or
    takes one the caller passes in for the length of the call. There is no
    Tick-operated endpoint to fall back to.
    """
