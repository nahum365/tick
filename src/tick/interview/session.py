"""A private, persisted interview that translates answers one slot at a time.

The model sees only the user's transcript and the current slot schema. Tick's
questions never enter the request. A direct extraction is accepted only when
the provider returns an exact span of the current answer; a suggestion takes a
second request and then a separate user `accept` act before it gains provenance.
"""

from __future__ import annotations

import json
import os
import secrets
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from tick.agents import (
    ModelAgentError,
    ModelAgentSpec,
    ModelReply,
    ModelRequest,
    Provider,
    StructuredReply,
    agent_spec_id,
    client_for,
)
from tick.engine import CadenceRefused, check_cadence
from tick.records import (
    DataSource,
    RecordKind,
    ensure_private_dir,
    normalize_payload,
    tick_home,
    utc_clock,
    write_private_file,
)
from tick.runtime import AgentRun, ApprovalMode, agent_id_for
from tick.spec import Cage, StrategySpec, sha256_hex

from .draft import Draft
from .errors import InterviewError
from .script import SLOTS, AgentKind, Slot, slot_by_name

__all__ = [
    "EXTRACT_TOOL_NAME",
    "INTERVIEW_MODEL_ENV",
    "SUGGESTION_DISCOURAGEMENT",
    "InterviewSession",
    "accept",
    "adopt",
    "answer",
    "next_question",
    "start",
]

EXTRACT_TOOL_NAME = "extract_slot"
MAX_OUTPUT_TOKENS = 2000
INTERVIEW_MODEL_ENV = "TICK_INTERVIEW_MODEL"
STATE_FILE = "state.json"
TRANSCRIPT_FILE = "transcript.jsonl"
TRANSCRIPT_MODE = 0o600

SUGGESTION_DISCOURAGEMENT = (
    "Your own values make the draft traceable. Answer with your value, or ask again "
    "to hear the provider's suggestion."
)


class PendingSuggestion(BaseModel):
    """A provider value that has no authority until a separate acceptance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    slot: str
    value: Any
    model: str


class InterviewState(BaseModel):
    """The complete restartable state stored under one private draft directory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    draft_id: str
    provider: Provider
    answers: dict[str, Any]
    provenance: dict[str, str]
    suggestion_discouraged: bool
    pending_suggestion: PendingSuggestion | None
    model_reported: str | None
    draft: dict[str, Any] | None


class InterviewSession:
    """One bounded interview loaded from `TICK_HOME/drafts/<id>`."""

    def __init__(self, home: str | os.PathLike[str], draft_id: str) -> None:
        if not draft_id or draft_id != Path(draft_id).name or draft_id in {".", ".."}:
            raise InterviewError(
                "DRAFT_ID_INVALID",
                f"{draft_id!r} is not a draft id. Run `tick agent interview` to start one.",
            )
        self.home = Path(home)
        self.draft_id = draft_id
        self.directory = self.home / "drafts" / draft_id
        self.state_path = self.directory / STATE_FILE
        self.transcript_path = self.directory / TRANSCRIPT_FILE

    @classmethod
    def create(
        cls,
        home: str | os.PathLike[str],
        *,
        provider: Provider,
        kind: AgentKind | None,
    ) -> InterviewSession:
        """Create empty private state; a supplied kind is an explicit user value."""
        provider = Provider(provider)
        kind = AgentKind(kind) if kind is not None else None
        draft_id = secrets.token_hex(6)
        session = cls(home, draft_id)
        ensure_private_dir(session.directory)
        answers: dict[str, Any] = {}
        provenance: dict[str, str] = {}
        if kind is not None:
            answers["kind"] = kind.value
            provenance["kind"] = "user"
        state = InterviewState(
            draft_id=draft_id,
            provider=provider,
            answers=answers,
            provenance=provenance,
            suggestion_discouraged=False,
            pending_suggestion=None,
            model_reported=None,
            draft=None,
        )
        session._save(state)
        session._ensure_transcript()
        return session

    @classmethod
    def from_chat_document(
        cls,
        home: str | os.PathLike[str],
        *,
        draft_id: str,
        provider: Provider,
        document: Mapping[str, Any],
        transcript: bytes,
    ) -> InterviewSession:
        """Materialize a valid chat proposal behind the existing adoption gate."""
        required = {"spec", "instructions", "approval", "provenance", "model_reported"}
        if set(document) != required:
            missing = sorted(required - set(document))
            extra = sorted(set(document) - required)
            details = []
            if missing:
                details.append("missing " + ", ".join(missing))
            if extra:
                details.append("off-schema " + ", ".join(extra))
            raise InterviewError(
                "DRAFT_DOCUMENT_INVALID",
                "the proposed agent document is "
                + "; ".join(details)
                + ". Ask the provider to emit the complete document again.",
            )
        session = cls(home, draft_id)
        ensure_private_dir(session.directory)
        try:
            transcript_text = transcript.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise InterviewError(
                "TRANSCRIPT_INVALID",
                "the setup transcript is not UTF-8. Delete the setup chat and start again.",
            ) from exc
        write_private_file(session.transcript_path, transcript_text)
        try:
            draft = Draft.model_validate(
                {
                    "draft_id": draft_id,
                    "provider": Provider(provider).value,
                    "transcript_sha256": sha256_hex(transcript),
                    **dict(document),
                }
            )
        except ValueError as exc:
            raise InterviewError(
                "DRAFT_DOCUMENT_INVALID",
                f"the proposed agent document does not match the agent schema ({exc}). "
                "Ask the provider to correct it and emit the complete document again.",
            ) from exc
        session._save(
            InterviewState(
                draft_id=draft_id,
                provider=Provider(provider),
                answers={},
                provenance=draft.provenance,
                suggestion_discouraged=False,
                pending_suggestion=None,
                model_reported=draft.model_reported,
                draft=draft.payload(),
            )
        )
        draft.to_agent()
        return session

    @property
    def state(self) -> InterviewState:
        try:
            text = self.state_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise InterviewError(
                "DRAFT_NOT_FOUND",
                f"draft {self.draft_id} cannot be read ({exc}). Start a new interview.",
            ) from exc
        try:
            payload = json.loads(text)
            return InterviewState.model_validate(payload)
        except (json.JSONDecodeError, ValueError) as exc:
            raise InterviewError(
                "DRAFT_STATE_INVALID",
                f"draft {self.draft_id} has invalid state ({exc}). Start a new interview; "
                "nothing has been adopted from this draft.",
            ) from exc

    @property
    def completed_draft(self) -> Draft:
        payload = self.state.draft
        if payload is None:
            raise InterviewError(
                "DRAFT_INCOMPLETE",
                f"draft {self.draft_id} still has unanswered fields. Answer the current "
                "question before adopting it.",
            )
        draft = Draft.model_validate(payload)
        actual_hash = self._transcript_hash()
        if actual_hash != draft.transcript_sha256:
            raise InterviewError(
                "TRANSCRIPT_HASH_MISMATCH",
                f"draft {self.draft_id}'s transcript now hashes to {actual_hash}, not the "
                f"recorded {draft.transcript_sha256}. Start a new interview; this draft "
                "cannot be adopted.",
            )
        return draft

    def ask(self) -> str | None:
        """The next unanswered required slot, or `None` when the draft is complete."""
        state = self.state
        if state.draft is not None:
            return None
        slot = self._next_slot(state)
        return None if slot is None else slot.question

    def answer(self, text: str) -> str:
        """Extract one answer with the user's provider, then ask the next question."""
        state = self.state
        if state.draft is not None:
            raise InterviewError(
                "DRAFT_ALREADY_COMPLETE",
                f"draft {self.draft_id} is complete. Show or adopt it instead of adding "
                "another answer.",
            )
        if state.pending_suggestion is not None:
            if text.strip().lower() == "accept":
                return self.accept()
            state = state.model_copy(update={"pending_suggestion": None})
            self._save(state)
        if not text.strip():
            question = self.ask()
            raise InterviewError(
                "ANSWER_EMPTY",
                f"the answer is empty. Answer the current question: {question}",
            )
        slot = self._next_slot(state)
        if slot is None:
            raise InterviewError(
                "DRAFT_STATE_INVALID",
                "the draft has no next slot but is not complete. Start a new interview; "
                "nothing has been adopted from this draft.",
            )
        self._append_transcript({"role": "user", "content": text})
        request = self._request(state, slot)
        try:
            reply = client_for(state.provider).propose(request)
        except ModelAgentError as exc:
            raise InterviewError(
                "INTERVIEW_PROVIDER_FAILED",
                f"{exc} The current field remains unanswered; answer its question again "
                "after the provider is available.",
            ) from exc
        if isinstance(reply, ModelReply) or not isinstance(reply, StructuredReply):
            return self._repeat(
                state,
                slot,
                "EXTRACTION_SHAPE_INVALID",
                "the provider answered with a different schema",
            )
        self._append_transcript(
            {
                "role": "model",
                "model": reply.model,
                "tool": reply.tool_name,
                "payload": dict(reply.payload),
            }
        )
        if reply.tool_name != EXTRACT_TOOL_NAME:
            return self._repeat(
                state,
                slot,
                "EXTRACTION_TOOL_INVALID",
                f"the provider called {reply.tool_name!r} instead of {EXTRACT_TOOL_NAME!r}",
            )
        payload = dict(reply.payload)
        asked = payload.get("asked_for_suggestion")
        if not isinstance(asked, bool):
            return self._repeat(
                state,
                slot,
                "EXTRACTION_SHAPE_INVALID",
                "the provider did not report whether the user asked for a suggestion",
            )
        if asked:
            return self._suggestion(state, slot, payload, model=reply.model)

        quoted = payload.get("quoted_span")
        if not isinstance(quoted, str) or not quoted or quoted not in text:
            return self._repeat(
                state,
                slot,
                "EXTRACTION_UNTRACED",
                "the provider's quoted span is not an exact substring of your answer",
            )
        try:
            value = slot.validator(payload.get("value"))
        except (CadenceRefused, TypeError, ValueError) as exc:
            return self._repeat(
                state,
                slot,
                "EXTRACTION_VALUE_INVALID",
                f"the extracted value does not fit {slot.type}: {exc}",
            )
        return self._record(state, slot, value, provenance="user", model=reply.model)

    def accept(self) -> str:
        """Give a pending provider suggestion provenance in a separate user act."""
        state = self.state
        pending = state.pending_suggestion
        if pending is None:
            raise InterviewError(
                "NO_SUGGESTION_PENDING",
                "there is no pending suggestion to accept. Answer the current question.",
            )
        slot = slot_by_name(pending.slot)
        self._append_transcript({"role": "user", "content": "accept", "act": "accept"})
        return self._record(
            state,
            slot,
            pending.value,
            provenance=f"model:{pending.model}",
            model=pending.model,
        )

    def _suggestion(
        self,
        state: InterviewState,
        slot: Slot,
        payload: dict[str, Any],
        *,
        model: str,
    ) -> str:
        if not state.suggestion_discouraged:
            updated = state.model_copy(
                update={"suggestion_discouraged": True, "model_reported": model}
            )
            self._save(updated)
            return f"{SUGGESTION_DISCOURAGEMENT}\n{slot.question}"
        try:
            value = slot.validator(payload.get("suggestion"))
        except (CadenceRefused, TypeError, ValueError) as exc:
            return self._repeat(
                state,
                slot,
                "SUGGESTION_VALUE_INVALID",
                f"the provider's suggestion does not fit {slot.type}: {exc}",
            )
        updated = state.model_copy(
            update={
                "pending_suggestion": PendingSuggestion(slot=slot.name, value=value, model=model),
                "model_reported": model,
            }
        )
        self._save(updated)
        shown = json.dumps(value, ensure_ascii=False, sort_keys=True)
        return (
            f"Provider suggestion for {slot.name}: {shown}\n"
            "Type accept in a separate turn to use it."
        )

    def _record(
        self,
        state: InterviewState,
        slot: Slot,
        value: Any,
        *,
        provenance: str,
        model: str,
    ) -> str:
        answers = dict(state.answers)
        answers[slot.name] = value
        provenance_map = dict(state.provenance)
        provenance_map[slot.name] = provenance
        updated = InterviewState(
            draft_id=state.draft_id,
            provider=state.provider,
            answers=answers,
            provenance=provenance_map,
            suggestion_discouraged=state.suggestion_discouraged,
            pending_suggestion=None,
            model_reported=model,
            draft=None,
        )
        next_slot = self._next_slot(updated)
        if next_slot is None:
            try:
                completed = self._build_draft(updated)
            except (TypeError, ValueError) as exc:
                raise InterviewError(
                    "DRAFT_DOCUMENT_INVALID",
                    f"the collected fields do not form a valid agent document ({exc}). "
                    "Start a new interview with consistent rules and limits; nothing "
                    "was created.",
                ) from exc
            updated = updated.model_copy(update={"draft": completed.payload()})
            self._save(updated)
            return (
                f"Draft {self.draft_id} is complete. Review it with `tick agent draft show "
                f"{self.draft_id}`; nothing exists as an agent until you adopt it."
            )
        self._save(updated)
        return next_slot.question

    def _repeat(self, state: InterviewState, slot: Slot, code: str, reason: str) -> str:
        self._save(state)
        return f"{code}: {reason}. Answer the question again.\n{slot.question}"

    def _next_slot(self, state: InterviewState) -> Slot | None:
        kind_value = state.answers.get("kind")
        kind = AgentKind(kind_value) if kind_value is not None else None
        for slot in SLOTS:
            if slot.name in state.answers:
                continue
            if slot.name in {"rules", "provider", "model", "instructions"}:
                if kind is None or kind not in slot.kinds:
                    continue
            return slot
        return None

    def _request(self, state: InterviewState, slot: Slot) -> ModelRequest:
        messages = tuple(
            {"role": "user", "content": event["content"]}
            for event in self._transcript()
            if event.get("role") == "user"
            and isinstance(event.get("content"), str)
            and event.get("act") != "accept"
        )
        return ModelRequest(
            model=str(state.answers.get("model") or os.environ.get(INTERVIEW_MODEL_ENV, "")),
            messages=messages,
            tools=(_extract_tool(slot),),
            max_tokens=MAX_OUTPUT_TOKENS,
        )

    def _build_draft(self, state: InterviewState) -> Draft:
        answers = state.answers
        kind = AgentKind(answers["kind"])
        cage = Cage.model_validate(
            {
                "max_position_pct": answers["cage.max_position_pct"],
                "max_positions": answers["cage.max_positions"],
                "max_order_notional": answers["cage.max_order_notional"],
                "max_daily_drawdown_pct": answers["cage.max_daily_drawdown_pct"],
                "allowed_session": answers["cage.allowed_session"],
            }
        )
        shared = {
            "name": "Interview draft",
            "version": 1,
            "universe": answers["universe"],
            "cadence": answers["cadence"],
            "cage": cage,
        }
        instructions: str | None
        if kind is AgentKind.RULE:
            spec = StrategySpec.model_validate(shared | {"rules": answers["rules"]})
            instructions = None
        else:
            spec = ModelAgentSpec.model_validate(
                shared
                | {
                    "kind": "model_agent",
                    "provider": answers["provider"],
                    "model": answers["model"],
                }
            )
            instructions = str(answers["instructions"])
        if state.model_reported is None:
            raise InterviewError(
                "MODEL_PROVENANCE_MISSING",
                "the provider did not report a model id. Answer the current question again "
                "after the provider can report it.",
            )
        return Draft(
            draft_id=state.draft_id,
            spec=spec,
            instructions=instructions,
            approval=ApprovalMode(answers["approval"]),
            provenance=dict(state.provenance),
            transcript_sha256=self._transcript_hash(),
            provider=state.provider.value,
            model_reported=state.model_reported,
        )

    def _save(self, state: InterviewState) -> None:
        payload = normalize_payload(state.model_dump(mode="python"))
        write_private_file(
            self.state_path,
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        )

    def _ensure_transcript(self) -> None:
        descriptor = os.open(
            self.transcript_path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            TRANSCRIPT_MODE,
        )
        try:
            os.fchmod(descriptor, TRANSCRIPT_MODE)
        finally:
            os.close(descriptor)

    def _append_transcript(self, event: dict[str, Any]) -> None:
        normalized = normalize_payload(event, where="transcript")
        line = json.dumps(normalized, sort_keys=True, separators=(",", ":")) + "\n"
        descriptor = os.open(
            self.transcript_path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            TRANSCRIPT_MODE,
        )
        try:
            os.fchmod(descriptor, TRANSCRIPT_MODE)
            os.write(descriptor, line.encode("utf-8"))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _transcript(self) -> tuple[dict[str, Any], ...]:
        try:
            text = self.transcript_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise InterviewError(
                "TRANSCRIPT_UNAVAILABLE",
                f"draft {self.draft_id}'s transcript cannot be read ({exc}). Start a new "
                "interview; nothing has been adopted from this draft.",
            ) from exc
        events: list[dict[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InterviewError(
                    "TRANSCRIPT_INVALID",
                    f"draft {self.draft_id}'s transcript line {line_number} is invalid "
                    f"({exc}). Start a new interview; nothing has been adopted.",
                ) from exc
            if not isinstance(payload, dict):
                raise InterviewError(
                    "TRANSCRIPT_INVALID",
                    f"draft {self.draft_id}'s transcript line {line_number} is not an "
                    "object. Start a new interview; nothing has been adopted.",
                )
            events.append(payload)
        return tuple(events)

    def _transcript_hash(self) -> str:
        try:
            content = self.transcript_path.read_bytes()
        except OSError as exc:
            raise InterviewError(
                "TRANSCRIPT_UNAVAILABLE",
                f"draft {self.draft_id}'s transcript cannot be hashed ({exc}). Start a "
                "new interview; nothing has been adopted.",
            ) from exc
        return sha256_hex(content)


def _extract_tool(slot: Slot) -> dict[str, Any]:
    def nullable(schema: Any) -> dict[str, Any]:
        return {"anyOf": [dict(schema), {"type": "null"}]}

    return {
        "name": EXTRACT_TOOL_NAME,
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["value", "quoted_span", "asked_for_suggestion", "suggestion"],
            "properties": {
                "value": nullable(slot.schema),
                "quoted_span": {"type": ["string", "null"]},
                "asked_for_suggestion": {"type": "boolean"},
                "suggestion": nullable(slot.schema),
            },
        },
    }


def start(provider: Provider | str, kind: AgentKind | str | None) -> str:
    """Start the box-API-shaped interview and return its private draft id."""
    selected_provider = Provider(provider)
    selected_kind = AgentKind(kind) if kind is not None else None
    return InterviewSession.create(
        tick_home(os.environ), provider=selected_provider, kind=selected_kind
    ).draft_id


def next_question(draft_id: str) -> str | None:
    """Return the next question for a persisted interview."""
    return InterviewSession(tick_home(os.environ), draft_id).ask()


def answer(draft_id: str, text: str) -> str:
    """Record and extract one answer for a persisted interview."""
    return InterviewSession(tick_home(os.environ), draft_id).answer(text)


def accept(draft_id: str) -> str:
    """Accept the pending suggestion as a separate user act."""
    return InterviewSession(tick_home(os.environ), draft_id).accept()


def adopt(
    draft_id: str,
    *,
    max_cancels: int,
    approval: ApprovalMode | str | None,
) -> AgentRun:
    """Create the agent and make adoption its first append-only record."""
    home = tick_home(os.environ)
    draft = InterviewSession(home, draft_id).completed_draft
    spec = draft.to_agent()
    try:
        check_cadence(spec.cadence)
    except CadenceRefused as exc:
        raise InterviewError(
            "DRAFT_CADENCE_REFUSED",
            f"{exc} Change the cadence in a new interview before adopting this draft.",
        ) from exc
    selected_approval = draft.approval if approval is None else ApprovalMode(approval)
    prospective = AgentRun(home, agent_id_for(spec))
    if prospective.exists:
        raise InterviewError(
            "AGENT_ALREADY_EXISTS",
            f"agent {prospective.agent_id} already exists. Use that agent or author a "
            "different draft; nothing was overwritten and no adoption note was added.",
        )
    run = AgentRun.create(
        home,
        spec,
        max_cancels_per_session=max_cancels,
        approval=selected_approval,
        created_at=datetime.now(UTC),
        instructions=draft.instructions,
    )
    run.ledger(clock=utc_clock).append(
        RecordKind.NOTE,
        {
            "event": "adopted",
            "draft_id": draft.draft_id,
            "spec_id": agent_spec_id(spec),
            "provenance": draft.provenance,
            "transcript_sha256": draft.transcript_sha256,
            "provider": draft.provider,
            "model_reported": draft.model_reported,
        },
        source=DataSource.RUNTIME,
    )
    return run
