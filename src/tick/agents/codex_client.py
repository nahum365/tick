"""A model agent on the user's own Codex login, one `codex exec` per tick.

This is the CLI shape of a model adapter. The user installed `codex` and
logged in with it; Tick runs it as a subprocess and reads its answer. Nothing
about the user's OpenAI credential is read, stored or seen by Tick — the
process inherits the user's own login the way any command they type would.

What one call looks like, and why each flag is there:

- `--output-schema <file>` — the intents schema, the same one the Anthropic
  adapter offers as a tool. The final message must conform to it, so a reply
  is a list of intents or nothing readable.
- `-o <file>` — the final message is read from a file, not scraped from the
  transcript.
- `--ignore-user-config` and `--ignore-rules` — the user's `config.toml` may
  register MCP servers (a brokerage, for instance) and hooks. A model agent's
  only route to a broker is the cage; a Codex session that could reach a
  broker's tools directly would be a second route with no cage on it. Auth is
  unaffected by this flag, per Codex's own help text.
- `-s read-only`, `--ephemeral`, `--skip-git-repo-check`, `-C <empty dir>` —
  no writes, no session files, no repository, nothing on the machine to read.
- `-m <model>` — the model the user's agent document or chat session pins.
  Chat creation observes the effective model once with a non-JSON header
  probe when the person did not name one, then every JSON turn passes it.

**The reply must say which model answered.** Codex prints a header on stderr
that names it (`model: …`); that line, not the id we asked for, is what goes
into the record. A run whose header names no model raises rather than being
recorded against the id in the document.

JSON chat events do not name the model in Codex 0.149. The session metadata is
therefore the source for chat turns; the model-agent structured reply still
uses its non-JSON stderr header as described above.

The subprocess runner is injected (`run=`) so every path here is exercised
against a fake that writes the files a real run would write, and no test
starts a process. `for_environment()` is the one function that constructs
the real one, and it refuses when the binary is not installed.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .client import ModelClient, ModelReply, ModelRequest, StructuredReply, intents_of
from .errors import ModelReplyError, ProviderUnavailable

__all__ = [
    "CODEX_BINARY",
    "CODEX_TIMEOUT_SECONDS",
    "CodexChatClient",
    "CodexChatIdentity",
    "CodexModelClient",
    "CompletedRun",
]

#: The command the user installed and logged in with. Never a path of ours.
CODEX_BINARY = "codex"

#: One agent turn, generously bounded. A tick that outlives this stops.
CODEX_TIMEOUT_SECONDS = 600.0

_MODEL_LINE = re.compile(r"^model:\s*(\S.*?)\s*$", re.MULTILINE)
_VERSION_LINE = re.compile(
    r"(?:^OpenAI Codex v|^codex-cli\s+)(\S+)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True, slots=True)
class CodexChatIdentity:
    """The model and CLI release observed once before a private chat starts."""

    model: str
    cli_version: str


class CompletedRun(Protocol):
    """The part of `subprocess.CompletedProcess` this adapter reads."""

    returncode: int
    stdout: str
    stderr: str


#: What a runner is: argv, the prompt on stdin, a timeout, back to a result.
Runner = Callable[[Sequence[str], str, float], CompletedRun]


def _real_runner(argv: Sequence[str], prompt: str, timeout: float) -> CompletedRun:
    return subprocess.run(  # noqa: S603 - argv is built here, from constants and the document
        list(argv),
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


class CodexModelClient:
    """Turns a `ModelRequest` into one `codex exec` run and reads its answer."""

    def __init__(
        self,
        *,
        run: Runner,
        binary: str = CODEX_BINARY,
        timeout_seconds: float = CODEX_TIMEOUT_SECONDS,
    ) -> None:
        self._run = run
        self._binary = binary
        self._timeout = timeout_seconds

    @classmethod
    def for_environment(cls) -> CodexModelClient:
        """Build a client on the `codex` this machine has. Refuse if it has none."""
        resolved = shutil.which(CODEX_BINARY)
        if resolved is None:
            raise ProviderUnavailable(
                f"no `{CODEX_BINARY}` command on this machine. A Codex-backed agent runs on "
                f"YOUR Codex login: install the Codex CLI and log in with it, then run the "
                f"agent again. Tick stores no credential and will not tick without it."
            )
        return cls(run=_real_runner, binary=resolved)

    def argv(self, request: ModelRequest, *, workdir: Path, schema: Path, last: Path) -> list[str]:
        """The exact command line, exposed so a test can read it apart."""
        argv = [
            self._binary,
            "exec",
            "--ignore-user-config",
            "--ignore-rules",
            "--ephemeral",
            "--skip-git-repo-check",
            "-s",
            "read-only",
            "-C",
            str(workdir),
            "--output-schema",
            str(schema),
            "-o",
            str(last),
        ]
        if request.model:
            argv.extend(("-m", request.model))
        argv.append("-")
        return argv

    def propose(self, request: ModelRequest) -> ModelReply | StructuredReply:
        tool_name, schema = _tool_of(request)
        prompt = request.composed_text()
        with tempfile.TemporaryDirectory(prefix="tick-codex-") as tmp:
            root = Path(tmp)
            workdir = root / "cwd"
            workdir.mkdir()
            schema_path = root / "schema.json"
            schema_path.write_text(json.dumps(schema, sort_keys=True), encoding="utf-8")
            last_path = root / "last.json"
            argv = self.argv(request, workdir=workdir, schema=schema_path, last=last_path)
            try:
                result = self._run(argv, prompt, self._timeout)
            except subprocess.TimeoutExpired as exc:
                raise ModelReplyError(
                    f"codex did not answer within {self._timeout:.0f}s. The tick stopped; "
                    f"nothing was placed and nothing was asked again."
                ) from exc
            if result.returncode != 0:
                raise ModelReplyError(
                    f"codex exited with status {result.returncode}: "
                    f"{_tail(result.stderr) or _tail(result.stdout) or 'no output'}. The tick "
                    f"stopped; nothing was placed."
                )
            model = _model_named_in(result.stderr)
            if model is None:
                raise ModelReplyError(
                    "codex's run header did not say which model answered. Tick will not "
                    "record a decision against a model id it was told rather than shown; "
                    "nothing was placed."
                )
            if not last_path.is_file():
                raise ModelReplyError(
                    "codex finished without writing its final message. Tick reads a decision "
                    "from the schema-checked answer file and never from the transcript; "
                    "nothing was placed."
                )
            text = last_path.read_text(encoding="utf-8")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ModelReplyError(
                f"codex's final message is not JSON ({exc.msg}). The schema was offered and "
                f"not honoured; nothing was placed."
            ) from exc
        if not isinstance(payload, dict):
            raise ModelReplyError(
                f"codex's final message is a {type(payload).__name__} where the schema "
                f"declares an object with an 'intents' list. Nothing was placed."
            )
        if tool_name == "emit_order_intents":
            return ModelReply(
                model=model,
                intents=intents_of(payload, source="codex's final message"),
            )
        return StructuredReply(model=model, tool_name=tool_name, payload=payload)


class CodexChatClient:
    """Run one isolated Codex process per chat turn with only Tick's box MCP."""

    def __init__(
        self,
        *,
        run: Runner,
        binary: str,
        tick_command: str,
        timeout_seconds: float,
    ) -> None:
        self._run = run
        self._binary = binary
        self._tick_command = tick_command
        self._timeout = timeout_seconds

    @classmethod
    def for_environment(cls, *, tick_command: str) -> CodexChatClient:
        resolved = shutil.which(CODEX_BINARY)
        if resolved is None:
            raise ProviderUnavailable(
                "no `codex` command is installed. Install it and run `codex login`; "
                "the chat remains on this box."
            )
        if shutil.which("codex-code-mode-host") is None:
            raise ProviderUnavailable(
                "no `codex-code-mode-host` command is installed. Run `tick provider install "
                "codex`, then create the chat again."
            )
        return cls(
            run=_real_runner,
            binary=resolved,
            tick_command=tick_command,
            timeout_seconds=CODEX_TIMEOUT_SECONDS,
        )

    def identify(self, requested_model: str | None) -> CodexChatIdentity:
        """Resolve the chat model once; a person's explicit choice needs no probe."""
        chosen = requested_model.strip() if requested_model is not None else None
        if chosen:
            result = self._checked_run(
                [self._binary, "--version"],
                "",
                purpose="report its installed version",
            )
            version = _codex_version_named_in(result.stdout, result.stderr)
            if version is None:
                raise ProviderUnavailable(
                    "codex did not report its CLI version. Reinstall or update the Codex CLI, "
                    "then create the chat again."
                )
            return CodexChatIdentity(model=chosen, cli_version=version)

        with tempfile.TemporaryDirectory(prefix="tick-chat-probe-") as tmp:
            workdir = Path(tmp) / "cwd"
            workdir.mkdir(mode=0o700)
            result = self._checked_run(
                self.probe_argv(workdir=workdir),
                "Reply with ready.",
                purpose="name the effective chat model",
            )
        model = _model_named_in(result.stderr)
        version = _codex_version_named_in(result.stdout, result.stderr)
        if model is None:
            raise ProviderUnavailable(
                "codex's probe header did not name the effective model. Reinstall or update "
                "the Codex CLI, or name a model when creating the chat."
            )
        if version is None:
            raise ProviderUnavailable(
                "codex's probe header did not name its CLI version. Reinstall or update the "
                "Codex CLI, then create the chat again."
            )
        return CodexChatIdentity(model=model, cli_version=version)

    def probe_argv(self, *, workdir: Path) -> list[str]:
        """Run without JSON once so the CLI's own header can name its effective model."""
        return [
            self._binary,
            "exec",
            "--ignore-user-config",
            "--ignore-rules",
            "--ephemeral",
            "--skip-git-repo-check",
            "-s",
            "read-only",
            "-C",
            str(workdir),
            "-",
        ]

    def argv(
        self,
        *,
        workdir: Path,
        setup_session_id: str | None,
        model: str,
    ) -> list[str]:
        """Expose the complete isolation boundary for a direct invariant test."""
        mcp_args = ["mcp"]
        if setup_session_id is not None:
            mcp_args.extend(("--setup-session", setup_session_id))
        return [
            self._binary,
            "exec",
            "--json",
            "--ignore-user-config",
            "--ignore-rules",
            "--ephemeral",
            "--skip-git-repo-check",
            "-s",
            "read-only",
            "-c",
            'mcp_servers.tick.default_tools_approval_mode="approve"',
            "-C",
            str(workdir),
            "-m",
            model,
            "-c",
            f"mcp_servers.tick.command={json.dumps(self._tick_command)}",
            "-c",
            f"mcp_servers.tick.args={json.dumps(mcp_args, separators=(',', ':'))}",
            "-",
        ]

    def turn(
        self,
        transcript: Sequence[Mapping[str, Any]],
        frame: str,
        *,
        setup_session_id: str | None,
        model: str,
    ) -> Iterable[dict[str, Any]]:
        """Replay the private transcript and preserve prose versus tool evidence."""
        prompt = json.dumps(
            {"frame": frame, "transcript": list(transcript)},
            ensure_ascii=False,
            sort_keys=True,
        )
        with tempfile.TemporaryDirectory(prefix="tick-chat-") as tmp:
            workdir = Path(tmp) / "cwd"
            workdir.mkdir(mode=0o700)
            try:
                result = self._run(
                    self.argv(
                        workdir=workdir,
                        setup_session_id=setup_session_id,
                        model=model,
                    ),
                    prompt,
                    self._timeout,
                )
            except subprocess.TimeoutExpired as exc:
                raise ModelReplyError(
                    f"codex did not answer within {self._timeout:.0f}s. The chat turn stopped; "
                    "send it again if you still want an answer."
                ) from exc
        if result.returncode != 0:
            raise ModelReplyError(
                f"codex exited with status {result.returncode}: "
                f"{_tail(result.stderr) or _tail(result.stdout) or 'no output'}. "
                "The chat turn stopped; retry after correcting the provider login."
            )
        completed: dict[str, Any] | None = None
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ModelReplyError(
                    f"codex emitted unreadable JSONL ({exc.msg}). The chat turn stopped; retry it."
                ) from exc
            chunk = _chat_chunk(event)
            if chunk is not None:
                if chunk.get("kind") == "done":
                    completed = chunk
                else:
                    yield chunk
        yield {"kind": "done", "model": model, **(completed or {})}

    def _checked_run(
        self,
        argv: Sequence[str],
        prompt: str,
        *,
        purpose: str,
    ) -> CompletedRun:
        try:
            result = self._run(argv, prompt, self._timeout)
        except subprocess.TimeoutExpired as exc:
            raise ProviderUnavailable(
                f"codex did not {purpose} within {self._timeout:.0f}s. Check the CLI login "
                "and create the chat again."
            ) from exc
        if result.returncode != 0:
            detail = _tail(result.stderr) or _tail(result.stdout) or "no output"
            raise ProviderUnavailable(
                f"codex could not {purpose} (status {result.returncode}: {detail}). Correct "
                "the CLI login or installation, then create the chat again."
            )
        return result


def _chat_chunk(event: Any) -> dict[str, Any] | None:
    if not isinstance(event, dict):
        return None
    event_type = str(event.get("type", ""))
    if event_type == "turn.completed":
        usage = event.get("usage")
        return {"kind": "done", "usage": usage if isinstance(usage, dict) else {}}
    item = event.get("item") if isinstance(event.get("item"), dict) else event
    item_type = str(item.get("type", event_type))

    if event_type == "item.completed" and item_type in {"agent_message", "message", "text"}:
        text = item.get("text") or item.get("message")
        if not isinstance(text, str) or not text:
            return None
        return {"kind": "text", "text": text}

    if event_type == "item.completed" and item_type == "error":
        message = _error_message(item.get("error") or item.get("message") or item.get("text"))
        return {
            "kind": "text",
            "text": message or "the provider reported an error without a message.",
            "source": "provider",
        }

    if item_type == "mcp_tool_call":
        name = item.get("tool") or item.get("name") or item.get("tool_name")
        arguments = item.get("arguments") or item.get("input") or {}
        server = item.get("server")
        if event_type == "item.started":
            return {
                "kind": "tool_call",
                "name": name,
                "server": server,
                "arguments": arguments,
            }
        if event_type != "item.completed":
            return None
        if item.get("status") != "completed":
            return {
                "kind": "tool_error",
                "name": name,
                "message": _error_message(item.get("error"))
                or (
                    "the provider reported that this tool call failed. Ask again or inspect "
                    "the box."
                ),
            }
        result = _decode_mcp_result(item.get("result"))
        if isinstance(result, dict) and result.get("executed") is False:
            return {"kind": "proposal", **result}
        chunk = {"kind": "tool_result", "name": name, "result": result}
        if isinstance(result, dict) and isinstance(result.get("evidence"), list):
            chunk["evidence"] = result["evidence"]
        return chunk

    # Compatibility for the pre-0.149 fixture vocabulary retained by older transcripts.
    text = item.get("text") or item.get("message")
    if isinstance(text, str) and text and ("message" in item_type or "text" in item_type):
        return {"kind": "text", "text": text}
    if "tool" in item_type:
        name = item.get("name") or item.get("tool_name")
        arguments = item.get("arguments") or item.get("input") or {}
        result = item.get("result") or item.get("output")
        if result is None:
            return {"kind": "tool_call", "name": name, "arguments": arguments}
        if isinstance(result, dict) and result.get("executed") is False:
            return {"kind": "proposal", **result}
        chunk = {"kind": "tool_result", "name": name, "result": result}
        if isinstance(result, dict) and isinstance(result.get("evidence"), list):
            chunk["evidence"] = result["evidence"]
        return chunk
    return None


def _decode_mcp_result(result: Any) -> Any:
    """Unwrap Codex Code Mode's MCP text envelopes without inventing a result."""
    value = result
    for _layer in range(2):
        if not isinstance(value, dict):
            return value
        structured = value.get("structured_content")
        if isinstance(structured, dict):
            return structured
        content = value.get("content")
        if not isinstance(content, list):
            return value
        texts = [
            item.get("text")
            for item in content
            if isinstance(item, dict)
            and item.get("type") == "text"
            and isinstance(item.get("text"), str)
        ]
        if len(texts) != 1:
            return value
        try:
            value = json.loads(texts[0])
        except json.JSONDecodeError:
            return value
    return value


def _error_message(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict):
        message = value.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
    return None


def _tool_of(request: ModelRequest) -> tuple[str, dict[str, Any]]:
    """The one named schema in the request, for intents or another bounded task."""
    found: list[tuple[str, dict[str, Any]]] = []
    for tool in request.tools:
        schema = tool.get("input_schema")
        name = tool.get("name")
        if isinstance(name, str) and name and isinstance(schema, dict):
            found.append((name, schema))
    if len(found) == 1:
        return found[0]
    raise ModelReplyError(
        "the request carries no intents schema or other single named output schema, so "
        "codex cannot be told what shape to answer in. Nothing was asked; correct the "
        "request and try again."
    )


def _model_named_in(stderr: str) -> str | None:
    match = _MODEL_LINE.search(stderr or "")
    return match.group(1) if match else None


def _codex_version_named_in(*streams: str) -> str | None:
    for stream in streams:
        match = _VERSION_LINE.search(stream or "")
        if match:
            return match.group(1)
    return None


def _tail(text: str, lines: int = 3) -> str:
    kept = [line for line in (text or "").splitlines() if line.strip()][-lines:]
    return " | ".join(kept)


#: A structural assertion for a reader: this adapter satisfies the port.
_PORT_CONFORMANCE: type[ModelClient] = CodexModelClient
