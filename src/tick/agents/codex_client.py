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
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from tick import __version__

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


class AppServerTransport(Protocol):
    """One JSON-RPC connection to a `codex app-server` process over stdio."""

    def send(self, message: Mapping[str, Any]) -> None: ...

    def receive(self, timeout: float) -> dict[str, Any] | None:
        """The next message, or None once the process has closed its output."""
        ...

    def close(self) -> None: ...


class _StdioAppServer:
    """A real app-server child; a reader thread keeps stdout draining."""

    def __init__(self, argv: Sequence[str]) -> None:
        self._process = subprocess.Popen(  # noqa: S603 - argv is built here from constants
            list(argv),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        self._lines: queue.Queue[str | None] = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        assert self._process.stdout is not None
        for line in self._process.stdout:
            self._lines.put(line)
        self._lines.put(None)

    def send(self, message: Mapping[str, Any]) -> None:
        assert self._process.stdin is not None
        self._process.stdin.write(json.dumps(message) + "\n")
        self._process.stdin.flush()

    def receive(self, timeout: float) -> dict[str, Any] | None:
        try:
            line = self._lines.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError from exc
        if line is None:
            return None
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ModelReplyError(
                f"codex app-server emitted an unreadable line ({exc.msg}). The chat turn "
                "stopped; retry it."
            ) from exc
        if not isinstance(value, dict):
            raise ModelReplyError(
                "codex app-server emitted a non-object message. The chat turn stopped; retry it."
            )
        return value

    def stderr_tail(self) -> str:
        if self._process.stderr is None:
            return ""
        try:
            return _tail(self._process.stderr.read())
        except (OSError, ValueError):
            return ""

    def close(self) -> None:
        self._process.terminate()
        try:
            self._process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        for stream in (self._process.stdin, self._process.stdout, self._process.stderr):
            if stream is not None:
                stream.close()


TransportFactory = Callable[[Sequence[str]], AppServerTransport]

#: Server-to-client requests Tick answers with a refusal: approvals, elicitations,
#: dynamic tool calls, token refreshes. A Tick chat grants nothing interactively.
_REFUSAL = {"code": -32601, "message": "Tick refuses interactive requests"}


class CodexAppServer:
    """Drive one app-server conversation: initialize, one thread, streamed turns.

    The connection lives for one Tick turn. Server requests are refused, every
    notification is handed to the caller in arrival order, and the response to
    each request is matched by id.
    """

    def __init__(self, transport: AppServerTransport, *, timeout_seconds: float) -> None:
        self._transport = transport
        self._deadline = time.monotonic() + timeout_seconds
        self._sequence = 0
        self.notifications: list[dict[str, Any]] = []

    def _remaining(self) -> float:
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError
        return remaining

    def request(self, method: str, params: Mapping[str, Any]) -> dict[str, Any]:
        """Send one request and return its result; notifications are collected meanwhile."""
        self._sequence += 1
        expected = self._sequence
        self._transport.send({"id": expected, "method": method, "params": dict(params)})
        while True:
            message = self._transport.receive(self._remaining())
            if message is None:
                raise ModelReplyError(
                    f"codex app-server closed before answering {method}. The chat turn "
                    "stopped; retry after checking the provider login."
                )
            if message.get("id") == expected and "method" not in message:
                if "error" in message:
                    detail = _error_message(message.get("error")) or "no detail"
                    raise ModelReplyError(
                        f"codex app-server refused {method}: {detail}. The chat turn stopped; "
                        "retry after correcting the provider login or model."
                    )
                result = message.get("result")
                return result if isinstance(result, dict) else {}
            self._handle(message)

    def notify(self, method: str) -> None:
        self._transport.send({"method": method})

    def events(self) -> Iterable[dict[str, Any]]:
        """Yield notifications until the turn completes; refuse server requests."""
        while True:
            message = self._transport.receive(self._remaining())
            if message is None:
                raise ModelReplyError(
                    "codex app-server closed before the turn completed. The chat turn stopped; "
                    "retry it."
                )
            if self._handle(message):
                yield message
                if message.get("method") == "turn/completed":
                    return

    def _handle(self, message: dict[str, Any]) -> bool:
        method = message.get("method")
        if method is None:
            return False
        if "id" in message:
            self._transport.send({"id": message["id"], "error": dict(_REFUSAL)})
            return False
        self.notifications.append(message)
        return True

    def close(self) -> None:
        self._transport.close()


class CodexChatClient:
    """Stream one Codex app-server turn per chat message, with only Tick's box MCP."""

    def __init__(
        self,
        *,
        run: Runner,
        binary: str,
        tick_command: str,
        timeout_seconds: float,
        transport: TransportFactory | None = None,
    ) -> None:
        self._run = run
        self._binary = binary
        self._tick_command = tick_command
        self._timeout = timeout_seconds
        self._transport = transport or _StdioAppServer

    @classmethod
    def for_environment(cls, *, tick_command: str) -> CodexChatClient:
        resolved = shutil.which(CODEX_BINARY)
        if resolved is None:
            raise ProviderUnavailable(
                "no `codex` command is installed. Install it and connect Codex from the app; "
                "the chat remains on this box."
            )
        return cls(
            run=_real_runner,
            binary=resolved,
            tick_command=tick_command,
            timeout_seconds=CODEX_TIMEOUT_SECONDS,
        )

    def app_server_argv(self) -> list[str]:
        """The one process this client starts; `CODEX_HOME` comes from Tick's environment."""
        return [self._binary, "app-server", "--listen", "stdio://"]

    def thread_params(
        self,
        *,
        workdir: Path,
        setup_session_id: str | None,
        model: str | None,
        frame: str | None,
    ) -> dict[str, Any]:
        """The complete isolation boundary of a Tick thread, exposed for a direct test."""
        mcp_args = ["mcp"]
        if setup_session_id is not None:
            mcp_args.extend(("--setup-session", setup_session_id))
        params: dict[str, Any] = {
            "cwd": str(workdir),
            "ephemeral": True,
            "sandbox": "read-only",
            "approvalPolicy": "never",
            "config": {
                "mcp_servers": {
                    "tick": {
                        "command": self._tick_command,
                        "args": mcp_args,
                        "default_tools_approval_mode": "approve",
                    }
                }
            },
        }
        if model is not None:
            params["model"] = model
        if frame is not None:
            params["developerInstructions"] = frame
        return params

    def _open(
        self, *, workdir: Path, setup_session_id: str | None, model: str | None, frame: str | None
    ) -> tuple[CodexAppServer, str, str]:
        server = CodexAppServer(
            self._transport(self.app_server_argv()), timeout_seconds=self._timeout
        )
        try:
            server.request(
                "initialize",
                {"clientInfo": {"name": "tick", "title": "Tick", "version": __version__}},
            )
            server.notify("initialized")
            started = server.request(
                "thread/start",
                self.thread_params(
                    workdir=workdir, setup_session_id=setup_session_id, model=model, frame=frame
                ),
            )
        except BaseException:
            server.close()
            raise
        thread = started.get("thread") if isinstance(started.get("thread"), dict) else {}
        thread_id = thread.get("id")
        effective = started.get("model")
        if not isinstance(thread_id, str) or not thread_id:
            server.close()
            raise ModelReplyError(
                "codex app-server started a thread without an id. The chat turn stopped; retry it."
            )
        if not isinstance(effective, str) or not effective.strip():
            server.close()
            raise ProviderUnavailable(
                "codex app-server did not name the thread's model. Choose a model in model "
                "settings, then create the chat again."
            )
        return server, thread_id, effective

    def identify(self, requested_model: str | None) -> CodexChatIdentity:
        """Resolve the chat model once from the server itself, never from a header."""
        result = self._checked_run(
            [self._binary, "--version"], "", purpose="report its installed version"
        )
        version = _codex_version_named_in(result.stdout, result.stderr)
        if version is None:
            raise ProviderUnavailable(
                "codex did not report its CLI version. Reinstall or update the Codex CLI, "
                "then create the chat again."
            )
        chosen = requested_model.strip() if requested_model is not None else None
        if chosen:
            return CodexChatIdentity(model=chosen, cli_version=version)
        with tempfile.TemporaryDirectory(prefix="tick-chat-probe-") as tmp:
            workdir = Path(tmp) / "cwd"
            workdir.mkdir(mode=0o700)
            try:
                server, _thread, effective = self._open(
                    workdir=workdir, setup_session_id=None, model=None, frame=None
                )
            except TimeoutError as exc:
                raise ProviderUnavailable(
                    f"codex did not name the effective chat model within {self._timeout:.0f}s. "
                    "Check the CLI login and create the chat again."
                ) from exc
            except ModelReplyError as exc:
                raise ProviderUnavailable(str(exc)) from exc
            server.close()
        return CodexChatIdentity(model=effective, cli_version=version)

    def turn(
        self,
        transcript: Sequence[Mapping[str, Any]],
        frame: str,
        *,
        setup_session_id: str | None,
        model: str,
    ) -> Iterable[dict[str, Any]]:
        """Replay the private transcript and stream prose, tool evidence and completion."""
        prompt = json.dumps(
            {"frame": frame, "transcript": list(transcript)},
            ensure_ascii=False,
            sort_keys=True,
        )
        with tempfile.TemporaryDirectory(prefix="tick-chat-") as tmp:
            workdir = Path(tmp) / "cwd"
            workdir.mkdir(mode=0o700)
            try:
                server, thread_id, effective = self._open(
                    workdir=workdir, setup_session_id=setup_session_id, model=model, frame=frame
                )
            except TimeoutError as exc:
                raise ModelReplyError(
                    f"codex app-server did not start a thread within {self._timeout:.0f}s. "
                    "The chat turn stopped; send it again if you still want an answer."
                ) from exc
            try:
                server.request(
                    "turn/start",
                    {"threadId": thread_id, "input": [{"type": "text", "text": prompt}]},
                )
                usage: dict[str, Any] = {}
                for event in server.events():
                    method = event.get("method")
                    params = event.get("params") if isinstance(event.get("params"), dict) else {}
                    if method == "thread/tokenUsage/updated":
                        reported = params.get("tokenUsage")
                        usage = reported if isinstance(reported, dict) else usage
                        continue
                    if method == "turn/completed":
                        turn = params.get("turn") if isinstance(params.get("turn"), dict) else {}
                        if turn.get("status") != "completed":
                            detail = _error_message(turn.get("error")) or "no detail"
                            raise ModelReplyError(
                                f"codex turn {turn.get('status') or 'failed'}: {detail}. The "
                                "chat turn stopped; retry after correcting the provider login."
                            )
                        break
                    chunk = _app_server_chunk(method, params)
                    if chunk is not None:
                        yield chunk
            except TimeoutError as exc:
                raise ModelReplyError(
                    f"codex did not answer within {self._timeout:.0f}s. The chat turn stopped; "
                    "send it again if you still want an answer."
                ) from exc
            finally:
                server.close()
        yield {"kind": "done", "model": effective, "usage": usage}

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


def _app_server_chunk(method: str | None, params: Mapping[str, Any]) -> dict[str, Any] | None:
    """Map one app-server notification onto Tick's chat chunk vocabulary."""
    if method == "item/agentMessage/delta":
        delta = params.get("delta")
        return {"kind": "text_delta", "text": delta} if isinstance(delta, str) and delta else None
    if method == "error":
        message = _error_message(params.get("error")) or _error_message(params.get("message"))
        if message is None:
            return None
        return {"kind": "text", "text": message, "source": "provider"}
    if method not in {"item/started", "item/completed"}:
        return None
    item = params.get("item") if isinstance(params.get("item"), dict) else {}
    item_type = item.get("type")
    if method == "item/completed" and item_type == "agentMessage":
        text = item.get("text")
        return {"kind": "text", "text": text} if isinstance(text, str) and text else None
    if item_type == "mcpToolCall":
        name = item.get("tool")
        arguments = item.get("arguments") if item.get("arguments") is not None else {}
        if method == "item/started":
            return {
                "kind": "tool_call",
                "name": name,
                "server": item.get("server"),
                "arguments": arguments,
            }
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
        chunk: dict[str, Any] = {"kind": "tool_result", "name": name, "result": result}
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


def codex_models(env: Mapping[str, str], *, timeout: float = 25) -> list[dict[str, str]]:
    process = subprocess.Popen(  # noqa: S603
        ["codex", "app-server", "--listen", "stdio://"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        env=dict(env),
        start_new_session=True,
    )
    lines: queue.Queue[str | None] = queue.Queue()

    def read() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            lines.put(line)
        lines.put(None)

    threading.Thread(target=read, daemon=True).start()
    deadline = time.monotonic() + timeout
    sequence = 0

    def send(value: dict[str, Any]) -> None:
        assert process.stdin is not None
        process.stdin.write(json.dumps(value) + "\n")
        process.stdin.flush()

    def rpc(method: str, params: dict[str, Any]) -> dict[str, Any]:
        nonlocal sequence
        sequence += 1
        send({"id": sequence, "method": method, "params": params})
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ValueError("Codex model discovery timed out. Retry the connection.")
            try:
                line = lines.get(timeout=remaining)
            except queue.Empty as exc:
                raise ValueError("Codex model discovery timed out. Retry the connection.") from exc
            if line is None:
                raise ValueError("Codex closed model discovery. Reconnect Codex.")
            message = json.loads(line)
            if not isinstance(message, dict):
                raise ValueError("Codex returned an unreadable response.")
            if message.get("id") == sequence and "method" not in message:
                if "error" in message:
                    raise ValueError("Codex could not list your models. Reconnect or update Codex.")
                result = message.get("result")
                if not isinstance(result, dict):
                    raise ValueError("Codex returned an unreadable model catalog.")
                return result
            if "id" in message and "method" in message:
                # Discovery never approves tools, supplies secrets, or starts work.
                send(
                    {
                        "id": message["id"],
                        "error": {"code": -32601, "message": "Metadata-only client"},
                    }
                )

    try:
        rpc("initialize", {"clientInfo": {"name": "tick", "title": "Tick", "version": "0.1.0"}})
        send({"method": "initialized"})
        account = rpc("account/read", {"refreshToken": False})
        if account.get("account") is None:
            raise ValueError("Sign in to Codex on your server to see your models.")
        cursor: str | None = None
        models: list[dict[str, str]] = []
        seen: set[str] = set()
        for _ in range(20):
            page = rpc("model/list", {"limit": 100, "includeHidden": False, "cursor": cursor})
            rows = page.get("data")
            if not isinstance(rows, list):
                raise ValueError("Codex returned an unreadable model catalog.")
            for row in rows:
                if not isinstance(row, dict) or row.get("hidden") is True:
                    continue
                model = row.get("model")
                if isinstance(model, str) and model.strip():
                    models.append(
                        {
                            "provider": "codex",
                            "model": model,
                            "display_name": row.get("displayName")
                            if isinstance(row.get("displayName"), str)
                            else model,
                        }
                    )
            cursor = page.get("nextCursor")
            if not cursor:
                return list({row["model"]: row for row in models}.values())
            if not isinstance(cursor, str) or cursor in seen:
                break
            seen.add(cursor)
        raise ValueError("Codex model pagination did not finish. Retry discovery.")
    finally:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        if process.stdin:
            process.stdin.close()
        if process.stdout:
            process.stdout.close()
