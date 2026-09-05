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

__all__ = [
    "CHAT_FRAME",
    "MAX_REPLAY_CHARACTERS",
    "ChatError",
    "ChatSession",
    "ChatTurn",
    "compact_document_frame",
    "stream_turn",
]

MAX_REPLAY_CHARACTERS = 200_000
_MAX_REPLAY_STRING_CHARACTERS = 2_000
_ELISION_TEXT = "Earlier tool results were elided; call the tool again if you need them."

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
        "progress",
        "refused",
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

    def turns_for_replay(self) -> tuple[Mapping[str, Any], ...]:
        """Bound provider context while the private transcript keeps exact bodies.

        Tool evidence is recoverable through the named box tool, so replay retains
        hashes and decision-bearing summaries instead of repeatedly sending contracts.
        Person and model prose is never discarded; the latest document summary is also
        permanent because it tells the provider what the box currently holds.
        """
        replay = [_compact_turn(turn) for turn in self.turns()]
        latest_document = next(
            (
                index
                for index in range(len(replay) - 1, -1, -1)
                if replay[index]["kind"] == "document"
            ),
            None,
        )
        removable = [
            index
            for index, turn in enumerate(replay)
            if turn["kind"] == "tool_result"
            or (turn["kind"] == "document" and index != latest_document)
        ]
        removed: set[int] = set()
        while _replay_characters(_with_elision(replay, removed)) > MAX_REPLAY_CHARACTERS:
            if not removable:
                raise ChatError(
                    "CHAT_REPLAY_TOO_LARGE",
                    "the person's and model's text alone exceeds the provider replay limit. "
                    "Delete this chat and start a new one; the full transcript remains on "
                    "the box until you do.",
                )
            removed.add(removable.pop(0))
        return tuple(_with_elision(replay, removed))

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


TurnAdapter = Callable[[tuple[Mapping[str, Any], ...], str], Iterable[Mapping[str, Any]]]


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
    provider_chunks = adapter(session.turns_for_replay(), CHAT_FRAME)

    def generate() -> Iterable[dict[str, Any]]:
        terminal = False
        try:
            for raw in provider_chunks:
                kind = raw.get("kind")
                if kind == "text_delta":
                    # Streamed prose fragments reach the phone as they arrive; the
                    # completed message is the one the transcript records.
                    yield dict(raw)
                    continue
                if kind not in {
                    "text",
                    "tool_call",
                    "tool_result",
                    "tool_error",
                    "proposal",
                    "document",
                    "progress",
                    "refused",
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


def compact_document_frame(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Expose document state without replaying a broker contract body."""
    document = payload.get("document")
    value: dict[str, Any] = {
        "kind": "document",
        "summary": _document_summary(document, payload),
    }
    if document is not None:
        value["document_hash"] = sha256_hex(canonical_encode(document))
    for key in ("valid", "complete"):
        if key in payload:
            value[key] = payload[key]
    return value


def _compact_turn(turn: ChatTurn) -> dict[str, Any]:
    if turn.kind == "document":
        return compact_document_frame(turn.payload)
    value = {"kind": turn.kind, **turn.payload}
    if turn.kind == "tool_result":
        tool = str(turn.payload.get("name") or "the same tool")
        value["result"] = _compact_tool_result(turn.payload.get("result"), tool=tool)
    return value


def _document_summary(document: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
    source = document if isinstance(document, Mapping) else {}
    proof = payload.get("proof") if isinstance(payload.get("proof"), Mapping) else {}
    raw_tools = source.get("tools") if isinstance(source.get("tools"), Mapping) else {}
    tools: dict[str, Any] = {}
    for name in sorted(str(value) for value in raw_tools):
        raw = raw_tools.get(name)
        if not isinstance(raw, Mapping):
            continue
        contract = raw.get("contract") if isinstance(raw.get("contract"), Mapping) else {}
        raw_proof = proof.get(name) if isinstance(proof.get(name), Mapping) else raw.get("proof")
        tool = {
            "category": raw.get("category"),
            "arguments": dict(raw.get("arguments"))
            if isinstance(raw.get("arguments"), Mapping)
            else {},
            "result": dict(raw.get("result")) if isinstance(raw.get("result"), Mapping) else {},
            "warnings": list(raw.get("warnings"))
            if isinstance(raw.get("warnings"), (list, tuple))
            else [],
            "finalized": bool(raw.get("finalized") or raw.get("confirmed_at") is not None),
            "proved": bool(isinstance(raw_proof, Mapping) and raw_proof.get("success") is True),
        }
        contract_hash = contract.get("contract_hash") or raw.get("contract_hash")
        if isinstance(contract_hash, str):
            tool["contract_hash"] = contract_hash
        tools[name] = _compact_tool_result(tool, tool="broker_draft")
    return {
        "server": source.get("server") or source.get("server_url"),
        "inventory_hash": source.get("inventory_hash"),
        "tools": tools,
    }


def _compact_tool_result(value: Any, *, tool: str) -> Any:
    if isinstance(value, str):
        if len(value) <= _MAX_REPLAY_STRING_CHARACTERS:
            return value
        marker = f"… [truncated; call {tool} again for the full value.]"
        kept = max(0, _MAX_REPLAY_STRING_CHARACTERS - len(marker))
        return value[:kept] + marker
    if isinstance(value, Mapping):
        compact: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            if name == "contract":
                compact[name] = _contract_reference(item)
            elif name == "contracts":
                compact[name] = (
                    [_contract_reference(contract) for contract in item]
                    if isinstance(item, (list, tuple))
                    else []
                )
            elif name == "document":
                compact[name] = _document_reference(item)
            else:
                compact[name] = _compact_tool_result(item, tool=tool)
        return compact
    if isinstance(value, (list, tuple)):
        return [_compact_tool_result(item, tool=tool) for item in value]
    return value


def _contract_reference(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    contract_hash = value.get("contract_hash")
    return {"contract_hash": contract_hash} if isinstance(contract_hash, str) else {}


def _document_reference(value: Any) -> dict[str, Any]:
    if value is None:
        return {"summary": _document_summary(None, {})}
    return {
        "document_hash": sha256_hex(canonical_encode(value)),
        "summary": _document_summary(value, {}),
    }


def _with_elision(replay: list[dict[str, Any]], removed: set[int]) -> list[Mapping[str, Any]]:
    if not removed:
        return list(replay)
    first = min(removed)
    values: list[Mapping[str, Any]] = []
    for index, turn in enumerate(replay):
        if index == first:
            values.append({"kind": "text", "text": _ELISION_TEXT, "source": "box"})
        if index not in removed:
            values.append(turn)
    return values


def _replay_characters(turns: list[Mapping[str, Any]]) -> int:
    return len(json.dumps(turns, ensure_ascii=False, sort_keys=True))
