"""Hash-linked chat transcripts with prose and evidence kept visibly distinct."""

from __future__ import annotations

import json
import secrets
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, model_validator

from tick.agents import Provider
from tick.agents.errors import ModelReplyError, ProviderUnavailable
from tick.records import ensure_private_dir, write_private_file
from tick.spec import canonical_encode, sha256_hex

__all__ = ["CHAT_FRAME", "ChatError", "ChatSession", "ChatTurn", "stream_turn"]

CHAT_FRAME = (
    "This box contains user-created agents, append-only records, approvals, broker "
    "profile state, and a kill switch. Read tools return box state. Proposal tools "
    "record a proposed action for separate confirmation by the user."
)


class ChatError(Exception):
    def __init__(self, code: str, reason: str) -> None:
        self.code = code
        self.reason = reason
        super().__init__(f"{code}: {reason}")


class ChatTurn(BaseModel):
    """One transcript line; its kind preserves prose versus sourced tool results."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seq: int
    at: AwareDatetime
    kind: Literal[
        "user",
        "text",
        "tool_call",
        "tool_result",
        "tool_error",
        "proposal",
        "document",
        "done",
        "error",
    ]
    payload: Mapping[str, Any]
    prev_hash: str
    hash: str

    @model_validator(mode="after")
    def _hash(self) -> ChatTurn:
        body = self.model_dump(mode="json", exclude={"hash"})
        if sha256_hex(canonical_encode(body)) != self.hash:
            raise ValueError("chat turn hash does not match its contents")
        return self


class ChatSession:
    """One provider-pinned transcript below ``TICK_HOME/chat``."""

    def __init__(self, home: Path, session_id: str) -> None:
        if not session_id or session_id != Path(session_id).name:
            raise ChatError("CHAT_ID_INVALID", "choose a chat id returned by GET /v1/chat.")
        self.home = home
        self.session_id = session_id
        self.directory = home / "chat" / session_id
        self.metadata_path = self.directory / "session.json"
        self.transcript_path = self.directory / "transcript.jsonl"

    @classmethod
    def create(
        cls,
        home: Path,
        *,
        provider: Provider,
        model: str | None,
        codex_cli_version: str | None,
        at: datetime,
    ) -> ChatSession:
        return cls._create(
            home,
            provider=provider,
            model=model,
            codex_cli_version=codex_cli_version,
            at=at,
            scope=None,
        )

    @classmethod
    def create_setup(
        cls,
        home: Path,
        *,
        provider: Provider,
        model: str | None,
        codex_cli_version: str | None,
        at: datetime,
        scope: str,
    ) -> ChatSession:
        """Create a scoped conversation without changing ordinary chat tools."""
        if not scope.strip():
            raise ChatError(
                "CHAT_SCOPE_INVALID",
                "scope must name one setup conversation. Choose a supported setup scope.",
            )
        return cls._create(
            home,
            provider=provider,
            model=model,
            codex_cli_version=codex_cli_version,
            at=at,
            scope=scope,
        )

    @classmethod
    def _create(
        cls,
        home: Path,
        *,
        provider: Provider,
        model: str | None,
        codex_cli_version: str | None,
        at: datetime,
        scope: str | None,
    ) -> ChatSession:
        provider = Provider(provider)
        if provider is Provider.ANTHROPIC and (model is None or not model.strip()):
            raise ChatError(
                "CHAT_MODEL_REQUIRED",
                "anthropic requires the model you chose. Name it and create the chat again.",
            )
        if provider is Provider.CODEX and (model is None or not model.strip()):
            raise ChatError(
                "CHAT_MODEL_REQUIRED",
                "codex needs a probed or person-chosen model. Resolve it and create the chat "
                "again.",
            )
        if provider is Provider.CODEX and (
            codex_cli_version is None or not codex_cli_version.strip()
        ):
            raise ChatError(
                "CHAT_CODEX_VERSION_REQUIRED",
                "codex needs its installed CLI version recorded. Check the CLI and create the "
                "chat again.",
            )
        if provider is Provider.ANTHROPIC and codex_cli_version is not None:
            raise ChatError(
                "CHAT_CODEX_VERSION_FORBIDDEN",
                "an anthropic chat cannot carry a Codex CLI version. Remove it and create the "
                "chat again.",
            )
        session = cls(home, secrets.token_hex(8))
        ensure_private_dir(session.directory)
        metadata = {
            "id": session.session_id,
            "provider": provider.value,
            "model": model,
            "created_at": at.isoformat(),
            "via": "api",
        }
        if codex_cli_version is not None:
            metadata["codex_cli_version"] = codex_cli_version
        if scope is not None:
            metadata["scope"] = scope
        write_private_file(
            session.metadata_path,
            json.dumps(metadata, sort_keys=True) + "\n",
        )
        write_private_file(session.transcript_path, "")
        return session

    @staticmethod
    def list(home: Path) -> list[dict[str, Any]]:
        root = home / "chat"
        if not root.is_dir():
            return []
        values: list[dict[str, Any]] = []
        for path in sorted(root.glob("*/session.json")):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if "scope" not in value:
                    values.append(value)
            except (OSError, json.JSONDecodeError):
                continue
        return values

    @property
    def metadata(self) -> dict[str, Any]:
        try:
            return json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ChatError(
                "CHAT_NOT_FOUND",
                f"chat {self.session_id} cannot be read ({exc}). Refresh the chat list.",
            ) from exc

    def turns(self) -> tuple[ChatTurn, ...]:
        try:
            lines = self.transcript_path.read_text(encoding="utf-8").splitlines()
            return tuple(ChatTurn.model_validate_json(line) for line in lines if line.strip())
        except (OSError, ValueError) as exc:
            raise ChatError(
                "CHAT_TRANSCRIPT_INVALID",
                f"chat {self.session_id} transcript is unreadable ({exc}). "
                "Delete this chat or inspect it on the box.",
            ) from exc

    def append(self, kind: str, payload: Mapping[str, Any], *, at: datetime) -> ChatTurn:
        turns = self.turns()
        previous = turns[-1].hash if turns else sha256_hex(b"tick.chat.v1.genesis")
        body = {
            "seq": len(turns) + 1,
            "at": at.isoformat().replace("+00:00", "Z"),
            "kind": kind,
            "payload": dict(payload),
            "prev_hash": previous,
        }
        turn = ChatTurn.model_validate(body | {"hash": sha256_hex(canonical_encode(body))})
        current = self.transcript_path.read_text(encoding="utf-8")
        write_private_file(
            self.transcript_path,
            current + json.dumps(turn.model_dump(mode="json"), sort_keys=True) + "\n",
        )
        return turn

    def transcript_hash(self) -> str:
        """Bind a proposed document to the exact private transcript bytes it followed."""
        try:
            return sha256_hex(self.transcript_path.read_bytes())
        except OSError as exc:
            raise ChatError(
                "CHAT_TRANSCRIPT_INVALID",
                f"chat {self.session_id} transcript cannot be hashed ({exc}). "
                "Delete this chat or inspect it on the box.",
            ) from exc

    def delete(self) -> None:
        """Delete only this user-owned conversation, never an agent record."""
        for path in (self.transcript_path, self.metadata_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        try:
            self.directory.rmdir()
        except OSError as exc:
            raise ChatError(
                "CHAT_DELETE_FAILED",
                f"chat {self.session_id} could not be deleted ({exc}). Inspect its directory.",
            ) from exc


TurnAdapter = Callable[[tuple[ChatTurn, ...], str], Iterable[Mapping[str, Any]]]


def stream_turn(
    session: ChatSession,
    text: str,
    *,
    at: datetime,
    adapter: TurnAdapter,
) -> Iterable[dict[str, Any]]:
    """Persist and return JSON-line chunks; conversation text is not filtered."""
    if not text.strip():
        raise ChatError("CHAT_TURN_EMPTY", "write a message, then send the turn again.")
    session.append("user", {"text": text}, at=at)
    provider_chunks = adapter(session.turns(), CHAT_FRAME)

    def generate() -> Iterable[dict[str, Any]]:
        terminal = False
        try:
            for raw in provider_chunks:
                kind = raw.get("kind")
                if kind not in {
                    "text",
                    "tool_call",
                    "tool_result",
                    "tool_error",
                    "proposal",
                    "document",
                    "done",
                    "error",
                }:
                    raise ChatError(
                        "CHAT_STREAM_INVALID",
                        "the provider adapter emitted an unknown chunk. The turn stopped; "
                        "retry it.",
                    )
                chunk = dict(raw)
                session.append(
                    str(kind),
                    {key: value for key, value in chunk.items() if key != "kind"},
                    at=at,
                )
                terminal = kind in {"done", "error"}
                yield chunk
        except (ModelReplyError, ProviderUnavailable) as exc:
            reason = str(exc)
            session.append("text", {"text": reason, "source": "provider"}, at=at)
            yield {
                "kind": "refused",
                "code": (
                    "provider_unavailable"
                    if isinstance(exc, ProviderUnavailable)
                    else "model_reply_error"
                ),
                "reason": reason,
            }
            return
        if not terminal:
            session.append("done", {}, at=at)
            yield {"kind": "done"}

    return generate()
