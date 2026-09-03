"""The candidate document and the provenance gate before adoption."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from tick.agents import AgentSpec, ModelAgentSpec
from tick.runtime import ApprovalMode
from tick.spec import StrategySpec

from .errors import InterviewError

__all__ = ["Draft", "meaning_bearing_fields"]

_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")


def meaning_bearing_fields(spec: AgentSpec) -> frozenset[str]:
    """Every interview slot whose value changes what is run or approved."""
    common = {
        "universe",
        "cadence",
        "kind",
        "cage.max_position_pct",
        "cage.max_positions",
        "cage.max_order_notional",
        "cage.max_daily_drawdown_pct",
        "cage.allowed_session",
        "approval",
    }
    if isinstance(spec, ModelAgentSpec):
        common.update(("provider", "model", "instructions"))
    else:
        common.add("rules")
    return frozenset(common)


class Draft(BaseModel):
    """A complete candidate that still has no authority to create an agent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    draft_id: str
    spec: StrategySpec | ModelAgentSpec
    instructions: str | None
    approval: ApprovalMode
    provenance: dict[str, str]
    transcript_sha256: str
    provider: str
    model_reported: str

    @field_validator("transcript_sha256")
    @classmethod
    def _check_hash(cls, value: str) -> str:
        if not _SHA256.match(value):
            raise ValueError("transcript_sha256 must be a lowercase sha256 digest")
        return value

    def to_agent(self) -> AgentSpec:
        """Return the candidate only when every meaning-bearing slot has provenance."""
        required = meaning_bearing_fields(self.spec)
        missing = sorted(required - self.provenance.keys())
        invalid = sorted(
            key
            for key in required & self.provenance.keys()
            if self.provenance[key] != "user"
            and not (
                self.provenance[key].startswith("model:")
                and self.provenance[key].removeprefix("model:").strip()
            )
        )
        if missing or invalid:
            details: list[str] = []
            if missing:
                details.append("missing " + ", ".join(missing))
            if invalid:
                details.append("invalid " + ", ".join(invalid))
            raise InterviewError(
                "DRAFT_PROVENANCE_INCOMPLETE",
                "the draft cannot be adopted because its provenance is "
                + "; ".join(details)
                + ". Answer those interview questions before adopting it.",
            )
        return self.spec

    def payload(self) -> dict[str, Any]:
        """The private JSON representation persisted inside session state."""
        return self.model_dump(mode="json")
