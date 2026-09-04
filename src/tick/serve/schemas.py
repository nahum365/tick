"""Wire shapes for the box API mirrored by the phone client.

Ledger rows intentionally expose ``at`` even though the persisted chain calls its
record timestamp ``ts``.  The mapping happens at the transport boundary so the
on-disk hash format remains unchanged while the app receives its established key.
"""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict


class ErrorResponse(TypedDict):
    code: str
    reason: str


class BrowserViewport(TypedDict):
    width: int
    height: int


class BrowserSessionResponse(TypedDict):
    session_id: str
    origin: str


class BrowserFrame(TypedDict):
    done: bool
    t: NotRequired[int]
    jpeg: NotRequired[str]
    origin: NotRequired[str]
    reason: NotRequired[str]


class ProviderAvailability(TypedDict):
    available: bool


class ProviderStatus(TypedDict):
    codex: ProviderAvailability
    anthropic: ProviderAvailability


class BrokerStatus(TypedDict):
    profile_state: Literal["confirmed", "drifted", "none"]
    profile_hash: str | None
    server_host: str | None
    sanction: str | None


class AgentStatus(TypedDict):
    id: str
    name: str
    kind: str
    mode: str
    approval: str
    last_tick: str | None
    stopped: bool
    drifted: bool
    run_state: Literal["running", "stopped", "unknown"]
    current_mode: Literal["paper", "live"] | None
    previous_run_mode: Literal["paper", "live"] | None
    transition: str | None
    attention_required: bool
    last_contact: str | None


class StatusResponse(TypedDict):
    version: str
    box_time: str
    agents: list[AgentStatus]
    provider: ProviderStatus
    broker: BrokerStatus
    ledger_ok: bool


class ApprovalDecisionBody(TypedDict):
    decision: Literal["approve", "decline"]


class LaunchBody(TypedDict):
    live: bool
    standing_ok: bool
    idempotency_key: NotRequired[str]


class InterviewStartBody(TypedDict):
    provider: Literal["codex", "anthropic"]
    kind: Literal["rule", "model"]
    model: NotRequired[str]


class InterviewAnswerBody(TypedDict):
    answer: str


class InterviewAdoptBody(TypedDict):
    name: str
    max_cancels: int
    approval: NotRequired[Literal["each", "standing"]]


class BrokerConfirmBody(TypedDict):
    tool: str
    confirm: Literal[True]


class CommonsGraphQuery(TypedDict):
    ticker: str
    depth: Literal[1, 2]
    observed_before: NotRequired[str]


class CommonsScreenQuery(TypedDict):
    criterion: list[str]
    observed_before: NotRequired[str]


class CommonsCreditsQuery(TypedDict):
    observed_before: NotRequired[str]


class ChatProposalEvent(TypedDict):
    proposal_id: str
    action: str
    arguments: dict[str, object]
    transcript_hash: str
    executed: Literal[False]


class SetupChatStartBody(TypedDict):
    scope: Literal["broker_profile", "agent_draft"]
    provider: Literal["codex", "anthropic"]
    resume: bool
    model: NotRequired[str]


class SetupChatTurnBody(TypedDict):
    text: str


class SetupChatResponse(TypedDict):
    chat: dict[str, object]
    transcript: list[dict[str, object]]
    document: dict[str, object] | None
    valid: bool
    verdict: dict[str, object]
