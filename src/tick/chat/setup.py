"""Scoped setup conversations whose only completion signal comes from code."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict

from tick.agents import Provider
from tick.records import write_private_file

from .session import ChatError, ChatSession

__all__ = ["SETUP_FRAMES", "SetupChatSession", "SetupScope", "SetupState"]


class SetupScope(StrEnum):
    """The two authoring tasks allowed to replace forms with conversation."""

    BROKER_PROFILE = "broker_profile"
    AGENT_DRAFT = "agent_draft"


SETUP_FRAMES = {
    SetupScope.BROKER_PROFILE: (
        "Help the person produce one complete broker profile document from the advertised "
        "contracts. Use only this setup session's broker tools. Read the compact inventory, "
        "request a full contract only when needed, then propose the whole document. Warnings "
        "are advisory; the denial registry is final. After a checked failure, fix the complete "
        "document and propose it again without waiting for the person. When proof needs a "
        "person-supplied symbol, count, quantity, or side, ask once in one message that lists "
        "exactly the missing values and why proof needs them; suggest no value. Extract the "
        "answer into prove_broker_draft. A proposal or proof grants no authority. The person "
        "finalizes mapped read tools together and every order tool individually."
    ),
    SetupScope.AGENT_DRAFT: (
        "Interview the person until one complete agent draft exists. Use only this setup "
        "session's agent-draft tools: call interview_script and agent_draft before your first "
        "question; the script is the complete list of fields, and no other field exists. "
        "Ask exactly one field per message, in script order. Each message names the field in "
        "plain words, repeats the script's explanation of what it controls, and offers one "
        "concrete suggestion drawn from what the person has told you, marked as your "
        "suggestion; the person may accept it, change it, or give their own value. A value "
        "the person states carries user provenance; a suggestion the person accepts carries "
        "model:<reported-id> provenance; never record a value the person has not confirmed. "
        "Fields already settled by the conversation are not asked again: the agent kind from "
        "the opening message, and for a model-driven agent the provider and model of this very "
        "session. For approval, propose confirming each order and say the person chooses this "
        "again at adoption. Supply structure, never strategy content. When every field is "
        "settled, call propose_agent_draft with the whole document; after a checked structural "
        "failure, fix and re-propose without waiting. A proposal creates no agent; only the "
        "person may adopt it."
    ),
}


class SetupState(BaseModel):
    """Restartable document verdict; validity is always a deterministic box result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scope: SetupScope
    goal: Literal["full", "simulation"] = "full"
    document: dict[str, Any] | None
    valid: bool
    complete: bool
    waiting_for: tuple[str, ...]
    probe_values: dict[str, Any]
    proof: dict[str, Any]
    verdict: dict[str, Any]
    updated_at: AwareDatetime


class SetupChatSession:
    """A ``ChatSession`` plus the latest checked setup document."""

    def __init__(self, home: Path, session_id: str) -> None:
        self.chat = ChatSession(home, session_id)
        self.state_path = self.chat.directory / "setup.json"

    @classmethod
    def create(
        cls,
        home: Path,
        *,
        scope: SetupScope,
        provider: Provider,
        model: str | None,
        codex_cli_version: str | None,
        at: datetime,
        goal: Literal["full", "simulation"] = "full",
    ) -> SetupChatSession:
        selected = SetupScope(scope)
        chat = ChatSession.create_setup(
            home,
            provider=provider,
            model=model,
            codex_cli_version=codex_cli_version,
            at=at,
            scope=selected.value,
        )
        session = cls(home, chat.session_id)
        session.save(
            document=None,
            valid=False,
            complete=False,
            waiting_for=(),
            probe_values={},
            proof={},
            verdict={
                "code": "SETUP_DOCUMENT_MISSING",
                "reason": "Keep chatting until the box accepts a complete document.",
            },
            at=at,
            goal=goal,
        )
        return session

    @property
    def state(self) -> SetupState:
        try:
            metadata = self.chat.metadata
            state = SetupState.model_validate_json(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ChatError(
                "SETUP_CHAT_NOT_FOUND",
                f"setup chat {self.chat.session_id} cannot be read ({exc}). Start it again.",
            ) from exc
        if metadata.get("scope") != state.scope.value:
            raise ChatError(
                "SETUP_SCOPE_MISMATCH",
                "the setup chat scope does not match its document state. "
                "Delete it and start again.",
            )
        return state

    def save(
        self,
        *,
        document: dict[str, Any] | None,
        valid: bool,
        complete: bool,
        waiting_for: tuple[str, ...],
        probe_values: dict[str, Any],
        proof: dict[str, Any],
        verdict: dict[str, Any],
        at: datetime,
        goal: Literal["full", "simulation"] | None = None,
    ) -> SetupState:
        state = SetupState(
            scope=SetupScope(self.chat.metadata["scope"]),
            goal=goal
            if goal is not None
            else (self.state.goal if self.state_path.exists() else "full"),
            document=document,
            valid=valid,
            complete=complete,
            waiting_for=waiting_for,
            probe_values=probe_values,
            proof=proof,
            verdict=verdict,
            updated_at=at,
        )
        write_private_file(self.state_path, state.model_dump_json(indent=2) + "\n")
        return state

    def response(self) -> dict[str, Any]:
        state = self.state
        return {
            "chat": self.chat.metadata,
            "goal": state.goal,
            "transcript": [turn.model_dump(mode="json") for turn in self.chat.turns()],
            "document": state.document,
            "valid": state.valid,
            "complete": state.complete,
            "waiting_for": list(state.waiting_for),
            "proof": state.proof,
            "verdict": state.verdict,
        }

    def delete(self) -> None:
        """Delete setup-only state before the ordinary chat removes its directory."""
        state = self.state  # Refuse a general chat ID before removing any private files.
        try:
            self.state_path.unlink()
        except FileNotFoundError:
            pass
        if state.scope is SetupScope.AGENT_DRAFT:
            draft = self.chat.home / "drafts" / self.chat.session_id
            for name in ("state.json", "transcript.jsonl"):
                try:
                    (draft / name).unlink()
                except FileNotFoundError:
                    pass
            try:
                draft.rmdir()
            except FileNotFoundError:
                pass
            except OSError:
                pass
        self.chat.delete()
