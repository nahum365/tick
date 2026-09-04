"""Small authenticated stdlib HTTP server for one user-owned Tick box."""

from __future__ import annotations

import hmac
import json
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from tick.records import normalize_payload

from . import handlers
from .handlers import APIError, ServeContext
from .pairing import PairingError, load_secret

__all__ = ["BoxServer", "FailureLimiter", "make_server", "serve"]

MAX_BODY_BYTES = 64 * 1024


@dataclass(frozen=True, slots=True)
class JSONLines:
    """A completed turn rendered with the streaming wire grammar."""

    chunks: tuple[dict[str, Any], ...]


class FailureLimiter:
    """Per-IP authentication throttling with an injected clock and wait function."""

    def __init__(
        self,
        *,
        monotonic: Callable[[], float],
        sleeper: Callable[[float], None],
    ) -> None:
        self._monotonic = monotonic
        self._sleeper = sleeper
        self._lock = threading.Lock()
        self._failures: dict[str, int] = {}
        self._next: dict[str, float] = {}

    def before_attempt(self, address: str) -> None:
        with self._lock:
            if self._failures.get(address, 0) < 5:
                return
            now = self._monotonic()
            permitted = max(now, self._next.get(address, now))
            self._next[address] = permitted + 1.0
            delay = permitted - now
        if delay > 0:
            self._sleeper(delay)

    def failed(self, address: str) -> None:
        with self._lock:
            self._failures[address] = self._failures.get(address, 0) + 1

    def succeeded(self, address: str) -> None:
        with self._lock:
            self._failures.pop(address, None)
            self._next.pop(address, None)


class BoxServer(ThreadingHTTPServer):
    """HTTP server carrying explicit dependencies rather than module globals."""

    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        *,
        context: ServeContext,
        limiter: FailureLimiter,
    ) -> None:
        self.context = context
        self.limiter = limiter
        super().__init__(address, BoxRequestHandler)


class BoxRequestHandler(BaseHTTPRequestHandler):
    """Route HTTP without logging headers, bodies, or pairing credentials."""

    server: BoxServer
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802 - stdlib callback name
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802 - stdlib callback name
        self._dispatch("POST")

    def do_PATCH(self) -> None:  # noqa: N802 - stdlib callback name
        self._dispatch("PATCH")

    def do_DELETE(self) -> None:  # noqa: N802 - stdlib callback name
        self._dispatch("DELETE")

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress access logs: authorization material never reaches a log line."""
        del format, args

    def _dispatch(self, method: str) -> None:
        parsed = urlsplit(self.path)
        path = unquote(parsed.path)
        try:
            if method == "GET" and path == "/v1/health":
                self._write(200, handlers.health(self.server.context))
                return
            if method == "POST" and path == "/v1/pair/recover":
                status, payload = handlers.pair_recover(self.server.context, self._body())
                self._write(status, payload)
                return
            self._authenticate()
            status, payload = self._route(method, path, parse_qs(parsed.query))
            if isinstance(payload, JSONLines):
                self._write_json_lines(status, payload)
            else:
                self._write(status, payload)
        except APIError as exc:
            self._write(exc.status, {"code": exc.code, "reason": exc.reason})
        except PairingError as exc:
            self._write(503, {"code": exc.code, "reason": exc.reason})
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception:  # noqa: BLE001 - transport converts unknown failures safely
            self._write(
                500,
                {
                    "code": "internal_error",
                    "reason": (
                        "the box could not complete this request. No authority was inferred; "
                        "inspect the box locally and try again."
                    ),
                },
            )

    def _authenticate(self) -> None:
        address = str(self.client_address[0])
        self.server.limiter.before_attempt(address)
        supplied = self.headers.get("Authorization", "")
        secret = load_secret(self.server.context.home)
        expected = f"Bearer {secret}"
        if not hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8")):
            self.server.limiter.failed(address)
            raise APIError(
                401,
                "unauthorized",
                "the pairing credential is missing or incorrect. Pair again on the box and retry.",
            )
        self.server.limiter.succeeded(address)

    def _route(
        self, method: str, path: str, query: Mapping[str, list[str]]
    ) -> tuple[int, dict[str, Any] | JSONLines]:
        context = self.server.context
        if method == "GET" and path == "/v1/status":
            return 200, handlers.status(context)
        if method == "GET" and path == "/v1/tunnel":
            return 200, handlers.tunnel(context)
        if method == "GET" and path == "/v1/doctor":
            return 200, handlers.doctor(context)
        if method == "POST" and path == "/v1/chat":
            return handlers.chat_create(context, self._body())
        if method == "GET" and path == "/v1/chat":
            return 200, handlers.chat_list(context)
        if method == "POST" and path == "/v1/provider/codex/login":
            self._require_empty_body()
            return handlers.provider_login_start(context)
        if method == "POST" and path == "/v1/provider/codex/install":
            self._require_empty_body()
            return handlers.provider_codex_install(context)
        if method == "POST" and path == "/v1/broker/connect":
            return handlers.broker_connect_start(context, self._body())
        if method == "GET" and path == "/v1/agents":
            return 200, handlers.agents(context)
        if method == "GET" and path == "/v1/drafts":
            return 200, handlers.drafts(context)
        if method == "GET" and path == "/v1/commons/status":
            return 200, handlers.commons_status(context)
        if method == "POST" and path == "/v1/commons/keygen":
            self._require_empty_body()
            return handlers.commons_keygen(context)
        if method == "POST" and path == "/v1/commons/opt-in":
            return handlers.commons_opt_in(context, self._body())
        if method == "GET" and path == "/v1/commons/pass":
            raw = query.get("ticker", [])
            if len(raw) != 1:
                raise APIError(400, "ticker_required", "ticker must appear once. Correct it.")
            return 200, handlers.commons_pass(context, raw[0])
        if method == "GET" and path == "/v1/commons/graph":
            ticker = query.get("ticker", [])
            depth = query.get("depth", [])
            observed = query.get("observed_before", [])
            if len(ticker) != 1:
                raise APIError(400, "ticker_required", "ticker must appear once. Correct it.")
            if len(depth) != 1:
                raise APIError(
                    400, "depth_required", "depth must appear once as 1 or 2. Correct it."
                )
            if len(observed) > 1:
                raise APIError(
                    400,
                    "observed_before_invalid",
                    "observed_before may appear once. Correct it and retry.",
                )
            try:
                parsed_depth = int(depth[0])
            except ValueError as exc:
                raise APIError(
                    400, "depth_invalid", "depth must be 1 or 2. Choose one and retry."
                ) from exc
            return 200, handlers.commons_graph(
                context,
                ticker[0],
                parsed_depth,
                observed[0] if observed else None,
            )
        if method == "GET" and path == "/v1/commons/screen":
            observed = query.get("observed_before", [])
            if len(observed) > 1:
                raise APIError(
                    400,
                    "observed_before_invalid",
                    "observed_before may appear once. Correct it and retry.",
                )
            return 200, handlers.commons_screen(
                context,
                query.get("criterion", []),
                observed[0] if observed else None,
            )
        if method == "GET" and path == "/v1/commons/credits":
            observed = query.get("observed_before", [])
            if len(observed) > 1:
                raise APIError(
                    400,
                    "observed_before_invalid",
                    "observed_before may appear once. Correct it and retry.",
                )
            return 200, handlers.commons_credits(context, observed[0] if observed else None)
        if method == "POST" and path == "/v1/purge":
            return handlers.purge(context, self._body())
        if method == "GET" and path == "/v1/notifications":
            raw = query.get("after", ["0"])
            try:
                after = int(raw[0]) if len(raw) == 1 else -1
            except ValueError as exc:
                raise APIError(
                    400, "after_invalid", "after must be one integer. Retry it."
                ) from exc
            return 200, handlers.notifications(context, after)
        if method == "GET" and path == "/v1/approvals":
            return 200, handlers.approvals(context)
        if method == "GET" and path == "/v1/broker/profile":
            return 200, handlers.broker_profile(context)
        if method == "GET" and path == "/v1/broker/profile/diff":
            return 200, handlers.broker_profile_diff(context)
        if method == "POST" and path == "/v1/broker/propose":
            return handlers.broker_propose(context, self._body())
        if method == "POST" and path == "/v1/broker/prove":
            return handlers.broker_prove(context, self._body())
        if method == "POST" and path == "/v1/broker/disconnect":
            self._require_empty_body()
            return handlers.broker_disconnect(context)
        if method == "POST" and path == "/v1/pair/rotate":
            return handlers.pair_rotate(context)
        if method == "POST" and path == "/v1/interview":
            return handlers.interview_start(context, self._body())
        if method == "POST" and path == "/v1/broker/confirm":
            return handlers.broker_confirm(context, self._body())
        if method == "POST" and path == "/v1/doctor/ack-demotion":
            return handlers.doctor_ack_demotion(context, self._body())

        pieces = [piece for piece in path.split("/") if piece]
        if (
            len(pieces) == 5
            and pieces[:4] == ["v1", "provider", "codex", "login"]
            and method == "GET"
        ):
            return 200, handlers.provider_login_status(context, pieces[4])
        if len(pieces) == 4 and pieces[:3] == ["v1", "broker", "connect"]:
            if method == "GET":
                return 200, handlers.broker_connect_status(context, pieces[3])
        if (
            len(pieces) == 5
            and pieces[:3] == ["v1", "broker", "connect"]
            and pieces[4] == "complete"
            and method == "POST"
        ):
            return handlers.broker_connect_complete(context, pieces[3], self._body())
        if len(pieces) == 3 and pieces[:2] == ["v1", "chat"]:
            if method == "GET":
                return 200, handlers.chat_get(context, pieces[2])
            if method == "DELETE":
                self._require_empty_body()
                return handlers.chat_delete(context, pieces[2])
        if len(pieces) == 3 and pieces[:2] == ["v1", "drafts"] and method == "GET":
            return 200, handlers.draft_get(context, pieces[2])
        if len(pieces) == 3 and pieces[:2] == ["v1", "agents"] and method == "GET":
            return 200, handlers.agent_document(context, pieces[2])
        if (
            len(pieces) == 4
            and pieces[:2] == ["v1", "chat"]
            and pieces[3] == "turns"
            and method == "POST"
        ):
            return 200, JSONLines(handlers.chat_turn(context, pieces[2], self._body()))
        if len(pieces) == 4 and pieces[:2] == ["v1", "approvals"] and method == "POST":
            raise APIError(404, "route_not_found", "this box route does not exist. Check the URL.")
        if len(pieces) == 3 and pieces[:2] == ["v1", "approvals"] and method == "POST":
            return handlers.approval_decide(context, pieces[2], self._body())
        if len(pieces) == 4 and pieces[:2] == ["v1", "agents"]:
            agent_id, action = pieces[2], pieces[3]
            if method == "PATCH" and action == "instructions":
                return handlers.agent_instructions(context, agent_id, self._body())
            if method == "POST" and action == "approval-mode":
                return handlers.agent_approval_mode(context, agent_id, self._body())
            if method == "GET" and action == "ledger":
                raw = query.get("after", ["0"])
                if len(raw) != 1:
                    raise APIError(400, "after_invalid", "after must appear once as an integer.")
                try:
                    after = int(raw[0])
                except ValueError as exc:
                    raise APIError(
                        400, "after_invalid", "after must be an integer sequence number."
                    ) from exc
                return 200, handlers.ledger(context, agent_id, after)
            if method == "POST" and action == "stop":
                return handlers.stop(context, agent_id, self._optional_body())
            if method == "POST" and action == "launch":
                return handlers.launch(context, agent_id, self._body())
        if (
            len(pieces) == 5
            and pieces[:2] == ["v1", "agents"]
            and pieces[3:] == ["ledger", "export"]
            and method == "GET"
        ):
            raw = query.get("for-evidence", [])
            if raw != ["true"]:
                raise APIError(
                    400,
                    "evidence_flag_required",
                    "set for-evidence=true; an unredacted export is not available.",
                )
            return 200, handlers.ledger_export(context, pieces[2])
        if (
            len(pieces) == 4
            and pieces[:2] == ["v1", "ledger"]
            and pieces[3] == "new"
            and method == "POST"
        ):
            return handlers.ledger_new(context, pieces[2], self._body())
        if len(pieces) == 4 and pieces[:2] == ["v1", "interview"] and method == "POST":
            draft_id, action = pieces[2], pieces[3]
            if action == "answer":
                return handlers.interview_answer(context, draft_id, self._body())
            if action == "accept":
                return handlers.interview_accept(context, draft_id, self._optional_body())
            if action == "adopt":
                return handlers.interview_adopt(context, draft_id, self._body())
        raise APIError(404, "route_not_found", "this box route does not exist. Check the URL.")

    def _body(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise APIError(400, "body_required", "this route requires a JSON object body.")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise APIError(
                400, "body_invalid", "Content-Length must be a non-negative integer."
            ) from exc
        if length < 0 or length > MAX_BODY_BYTES:
            raise APIError(
                413, "body_too_large", "the JSON body is too large. Send one route request."
            )
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise APIError(
                400, "body_invalid", f"the request body is not valid JSON ({exc}). Correct it."
            ) from exc
        if not isinstance(payload, dict):
            raise APIError(400, "body_invalid", "the request body must be a JSON object.")
        return payload

    def _optional_body(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None or raw_length == "0":
            return {}
        return self._body()

    def _require_empty_body(self) -> None:
        raw_length = self.headers.get("Content-Length")
        if raw_length is not None and int(raw_length) > 0:
            raise APIError(400, "body_forbidden", "this route takes no body. Remove it and retry.")

    def _write(self, status: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(
            normalize_payload(dict(payload), where="box response"),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _write_json_lines(self, status: int, payload: JSONLines) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        for chunk in payload.chunks:
            line = (
                json.dumps(
                    normalize_payload(chunk, where="chat stream"),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                + b"\n"
            )
            self.wfile.write(f"{len(line):X}\r\n".encode("ascii") + line + b"\r\n")
            self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


def make_server(
    bind: str,
    port: int,
    *,
    context: ServeContext,
    monotonic: Callable[[], float],
    sleeper: Callable[[float], None],
) -> BoxServer:
    """Construct without serving; tests bind ephemeral loopback ports in-process."""
    limiter = FailureLimiter(monotonic=monotonic, sleeper=sleeper)
    return BoxServer((bind, port), context=context, limiter=limiter)


def serve(
    bind: str,
    port: int,
    *,
    context: ServeContext,
) -> None:
    """Serve until interrupted; bind posture is enforced by the CLI."""
    server = make_server(
        bind,
        port,
        context=context,
        monotonic=time.monotonic,
        sleeper=time.sleep,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
