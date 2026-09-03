"""Readable refusals from authoring by interview."""

from __future__ import annotations

__all__ = ["InterviewError"]


class InterviewError(Exception):
    """A refusal with a stable code and a sentence that leaves a next step."""

    def __init__(self, code: str, reason: str) -> None:
        self.code = code
        self.reason = reason
        super().__init__(f"{code}: {reason}")
