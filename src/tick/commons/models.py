"""Closed outward models for claims, sources, releases, and subject passes.

These types are the privacy boundary as a shape. Unknown fields are rejected
instead of silently discarded, and the vocabulary contains public subjects
and evidence only; there is nowhere to put private brokerage state.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator


class CommonsModel(BaseModel):
    """Frozen, closed base for every commons request and response model."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class EvidenceClass(StrEnum):
    CHECKED = "checked"
    CORROBORATED = "corroborated"


class LicenseClass(StrEnum):
    DISPLAY_ONLY = "display_only"
    DERIVED_OK = "derived_ok"
    REDISTRIBUTABLE = "redistributable"


class CompensationProvenance(StrEnum):
    NONE = "none"
    EMPLOYER = "employer"
    ISSUER = "issuer"
    PAID_RESEARCH = "paid_research"
    OTHER_DISCLOSED = "other_disclosed"


class SubjectAlias(CommonsModel):
    ticker: str
    valid_from: AwareDatetime
    valid_to: AwareDatetime | None


class Subject(CommonsModel):
    subject_id: str
    kind: Literal["issuer", "concept"]
    cik: str | None
    figi: str | None
    aliases: tuple[SubjectAlias, ...]
    name: str


class SubjectsResponse(CommonsModel):
    subjects: tuple[Subject, ...]


class XbrlRecipe(CommonsModel):
    kind: Literal["xbrl_fact"]
    concept: tuple[str, ...]
    taxonomy: Literal["us-gaap"]


class FilingEventRecipe(CommonsModel):
    kind: Literal["sec_filing_event"]
    forms: tuple[str, ...]


class SicRecipe(CommonsModel):
    kind: Literal["sec_sic"]
    field: Literal["sic"]


Recipe = Annotated[XbrlRecipe | FilingEventRecipe | SicRecipe, Field(discriminator="kind")]


class Predicate(CommonsModel):
    id: str
    type: Literal["decimal", "integer", "text", "event"]
    unit: str
    period_kind: Literal["duration", "instant", "event"]
    recipe: Recipe
    position_shaped: Literal[False]


class Period(CommonsModel):
    start_at: AwareDatetime | None
    end_at: AwareDatetime
    fiscal_year: int | None
    fiscal_period: str | None


class SourceLocator(CommonsModel):
    url: str
    accession: str
    fragment: str


class Source(CommonsModel):
    source_id: str
    locator: SourceLocator
    fetched_at: AwareDatetime
    license_class: LicenseClass
    publisher: str


class Claim(CommonsModel):
    subject_id: str
    predicate_id: str
    value: str | int
    period: Period | None
    as_of: AwareDatetime
    observed_at: AwareDatetime
    source_id: str
    evidence_class: EvidenceClass
    supersedes: str | None
    resolves_dispute: str | None = None

    @field_validator("value", mode="before")
    @classmethod
    def value_is_exact(cls, value: object) -> object:
        """A binary float is fabricated precision, never a public fact."""
        if isinstance(value, float):
            raise ValueError("value is a binary float; provide an exact decimal string and retry")
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ValueError("value must be an exact decimal string, integer, or registered text")
        return value


class Attestation(CommonsModel):
    adapter: str
    model: str | None
    sanction: Literal["official", "community"]
    compensation_provenance: CompensationProvenance


class ClaimBody(CommonsModel):
    claim: Claim
    source: Source
    attestation: Attestation


class SignedClaimRequest(CommonsModel):
    body: ClaimBody
    contributor_id: str
    signature: str


class Verification(CommonsModel):
    method: Literal["recipe", "unsampled", "contributor_recheck"]
    result: Literal["verified", "unsampled"]
    evidence: str
    verified_at: AwareDatetime
    verifier_key: str


class ClaimAccepted(CommonsModel):
    claim_id: str
    verification: Verification


class PassClaim(CommonsModel):
    claim_id: str
    claim: Claim
    source: Source
    contributor_id: str
    verification: Verification


class PassResponse(CommonsModel):
    release_id: str
    subject: Subject
    claims: tuple[PassClaim, ...]


class ScreenOperator(StrEnum):
    LT = "lt"
    LTE = "lte"
    GT = "gt"
    GTE = "gte"
    EQ = "eq"
    BETWEEN = "between"
    EXISTS = "exists"


class ScreenCriterion(CommonsModel):
    predicate_id: str
    op: ScreenOperator
    value: str | int | None = None
    values: tuple[str | int, ...] | None = None
    period: str | None = None

    @model_validator(mode="after")
    def operands_match_operator(self) -> ScreenCriterion:
        """Require the operands whose meaning the selected comparison declares."""
        if self.op is ScreenOperator.EXISTS:
            if self.value is not None or self.values is not None:
                raise ValueError("exists takes no value; remove its value and retry")
            return self
        if self.op is ScreenOperator.BETWEEN:
            if self.value is not None or self.values is None or len(self.values) != 2:
                raise ValueError("between needs exactly two values; provide both bounds and retry")
            return self
        if self.value is None or self.values is not None:
            raise ValueError(f"{self.op} needs one value; provide that value and retry")
        return self

    @field_validator("value", mode="before")
    @classmethod
    def scalar_is_exact(cls, value: object) -> object:
        if isinstance(value, float):
            raise ValueError("screen values cannot be binary floats; use exact strings and retry")
        return value

    @field_validator("values", mode="before")
    @classmethod
    def bounds_are_exact(cls, value: object) -> object:
        if isinstance(value, (list, tuple)) and any(isinstance(item, float) for item in value):
            raise ValueError("screen values cannot be binary floats; use exact strings and retry")
        return value


class ScreenRequest(CommonsModel):
    criteria: tuple[ScreenCriterion, ...]
    observed_before: AwareDatetime | None = None


class ScreenMatchedClaim(CommonsModel):
    claim_id: str
    predicate_id: str
    value: str | int
    period: Period | None


class ScreenMatch(CommonsModel):
    subject: Subject
    claims: tuple[ScreenMatchedClaim, ...]


class ScreenResponse(CommonsModel):
    release_id: str
    matches: tuple[ScreenMatch, ...]
    next_after: str | None


class GraphEdge(CommonsModel):
    claim_id: str
    from_subject: str
    to_subject: str
    predicate_id: str


class GraphResponse(CommonsModel):
    release_id: str
    subject: Subject
    claims: tuple[PassClaim, ...]
    sources: tuple[Source, ...]
    edges: tuple[GraphEdge, ...]
    neighbors: tuple[Subject, ...]


class ConceptMember(CommonsModel):
    subject: Subject
    claim: PassClaim


class ConceptMembersResponse(CommonsModel):
    release_id: str
    concept: Subject
    members: tuple[ConceptMember, ...]
    next_after: str | None


class ReverifyRequest(CommonsModel):
    contributor_id: str
    signature: str


class ReverifyResponse(CommonsModel):
    claim_id: str
    verification: Verification
    credit_earned: int


class DisputeRequest(CommonsModel):
    source_id: str
    reason_code: str


class Dispute(CommonsModel):
    dispute_id: str
    claim_id: str
    source_id: str
    reason_code: str
    at: AwareDatetime


class DisputeAccepted(CommonsModel):
    dispute: Dispute


class ClaimDetailResponse(CommonsModel):
    release_id: str
    claim: PassClaim
    open_disputes: tuple[Dispute, ...]


class CreditEntryKind(StrEnum):
    EARNED_NOVEL_WRITE = "earned_novel_write"
    EARNED_REVERIFICATION = "earned_reverification"
    CONVERTED_STANDING = "converted_standing"


class CreditEntry(CommonsModel):
    entry_id: str
    key: str
    kind: CreditEntryKind
    claim_id: str | None
    amount: int
    at: AwareDatetime
    release_id: str


class StandingSlot(CommonsModel):
    key: str
    granted_at: AwareDatetime
    release_id: str


class CreditsResponse(CommonsModel):
    # The wire spelling is required by PRD 6.8. Internally it is named for
    # credits so the public-claim private-state vocabulary scan stays exact.
    credit_total: int = Field(validation_alias="balance", serialization_alias="balance")
    entries: tuple[CreditEntry, ...]
    standing_slots: tuple[StandingSlot, ...]
    next_after: str | None


class KeyAlarm(CommonsModel):
    alarm_id: str
    key: str
    at: AwareDatetime
    reason: str


class AlarmsResponse(CommonsModel):
    release_id: str
    alarms: tuple[KeyAlarm, ...]
    next_after: str | None


class ReleaseRequest(CommonsModel):
    note: str


class ReleaseResponse(CommonsModel):
    release_id: str
    cursor: int
    published_at: AwareDatetime


class HealthResponse(CommonsModel):
    status: Literal["ok"]


class ErrorResponse(CommonsModel):
    code: str
    reason: str


OUTWARD_MODELS = (
    SubjectAlias,
    Subject,
    SubjectsResponse,
    Period,
    SourceLocator,
    Source,
    Claim,
    Attestation,
    ClaimBody,
    SignedClaimRequest,
    Verification,
    ClaimAccepted,
    PassClaim,
    PassResponse,
    ScreenCriterion,
    ScreenRequest,
    ScreenMatchedClaim,
    ScreenMatch,
    ScreenResponse,
    GraphEdge,
    GraphResponse,
    ConceptMember,
    ConceptMembersResponse,
    ReverifyRequest,
    ReverifyResponse,
    DisputeRequest,
    Dispute,
    DisputeAccepted,
    ClaimDetailResponse,
    CreditEntry,
    StandingSlot,
    CreditsResponse,
    KeyAlarm,
    AlarmsResponse,
    ReleaseRequest,
    ReleaseResponse,
    HealthResponse,
    ErrorResponse,
)


def utc_now() -> datetime:
    """Return an aware wall-clock value for CLI-only local marker records."""
    from datetime import UTC

    return datetime.now(UTC)
