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
- `-m <model>` — the model the user's document pins. During an interview it
  is omitted until the user names one, so the user's Codex installation
  resolves the model and reports that id in its header; Tick chooses none.

**The reply must say which model answered.** Codex prints a header on stderr
that names it (`model: …`); that line, not the id we asked for, is what goes
into the record. A run whose header names no model raises rather than being
recorded against the id in the document.

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
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from .client import ModelClient, ModelReply, ModelRequest, StructuredReply, intents_of
from .errors import ModelReplyError, ProviderUnavailable

__all__ = [
    "CODEX_BINARY",
    "CODEX_TIMEOUT_SECONDS",
    "CodexChatClient",
    "CodexModelClient",
    "CompletedRun",
]

#: The command the user installed and logged in with. Never a path of ours.
CODEX_BINARY = "codex"

#: One agent turn, generously bounded. A tick that outlives this stops.
CODEX_TIMEOUT_SECONDS = 600.0

_MODEL_LINE = re.compile(r"^model:\s*(\S.*?)\s*$", re.MULTILINE)


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
        return cls(
            run=_real_runner,
            binary=resolved,
            tick_command=tick_command,
            timeout_seconds=CODEX_TIMEOUT_SECONDS,
        )

    def argv(self, *, workdir: Path, setup_session_id: str | None) -> list[str]:
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
            "-C",
            str(workdir),
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
    ) -> tuple[dict[str, Any], ...]:
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
                    self.argv(workdir=workdir, setup_session_id=setup_session_id),
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
        chunks: list[dict[str, Any]] = []
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
                chunks.append(chunk)
        model = _model_named_in(result.stderr)
        if model is None:
            raise ModelReplyError(
                "codex's run header did not say which model answered. The turn is not "
                "complete; inspect the installed Codex CLI and retry."
            )
        chunks.append({"kind": "done", "model": model})
        return tuple(chunks)


def _chat_chunk(event: Any) -> dict[str, Any] | None:
    if not isinstance(event, dict):
        return None
    event_type = str(event.get("type", ""))
    item = event.get("item") if isinstance(event.get("item"), dict) else event
    item_type = str(item.get("type", event_type))
    text = item.get("text") or item.get("message")
    if isinstance(text, str) and text and ("message" in item_type or "text" in item_type):
        return {"kind": "text", "text": text}
    if "tool" in item_type:
        name = item.get("name") or item.get("tool_name")
        arguments = item.get("arguments") or item.get("input") or {}
        result = item.get("result") or item.get("output")
        if result is not None:
            if isinstance(result, dict) and result.get("executed") is False:
                return {"kind": "proposal", **result}
            chunk = {"kind": "tool_result", "name": name, "result": result}
            if isinstance(result, dict) and isinstance(result.get("evidence"), list):
                chunk["evidence"] = result["evidence"]
            return chunk
        return {"kind": "tool_call", "name": name, "arguments": arguments}
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


def _tail(text: str, lines: int = 3) -> str:
    kept = [line for line in (text or "").splitlines() if line.strip()][-lines:]
    return " | ".join(kept)


#: A structural assertion for a reader: this adapter satisfies the port.
_PORT_CONFORMANCE: type[ModelClient] = CodexModelClient
