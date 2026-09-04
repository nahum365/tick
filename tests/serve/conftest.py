from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from http.client import HTTPConnection
from pathlib import Path

import pytest

from tests.runtime.conftest import build_spec
from tick.runtime import AgentRun, ApprovalMode
from tick.serve.handlers import ServeContext
from tick.serve.pairing import create_secret
from tick.serve.server import make_server


@pytest.fixture
def box_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "tick-home"
    monkeypatch.setenv("TICK_HOME", str(home))
    return home


@pytest.fixture
def box_agent(box_home: Path) -> AgentRun:
    return AgentRun.create(
        box_home,
        build_spec(),
        max_cancels_per_session=2,
        approval=ApprovalMode.EACH,
        created_at=datetime(2026, 9, 3, 12, 0, tzinfo=UTC),
        instructions=None,
    )


@pytest.fixture
def server_box(box_home: Path):
    _, secret = create_secret(box_home)
    started: list[list[str]] = []
    signalled: list[int] = []
    running: set[int] = set()

    class WireResult:
        def __init__(self, payload):
            self.payload = payload

        def model_dump(self, **_kwargs):
            return self.payload

    class FakeCommonsClient:
        def __init__(self):
            self.calls = []

        def pass_for(self, ticker):
            self.calls.append(("pass", ticker))
            return WireResult({"ticker": ticker, "claims": []})

        def graph_for(self, ticker, depth, observed_before):
            self.calls.append(("graph", ticker, depth, observed_before))
            return WireResult(
                {
                    "release_id": "release-1",
                    "subject": {"subject_id": "issuer:xyz", "name": "XYZ"},
                    "claims": [],
                    "sources": [],
                    "edges": [],
                    "neighbors": [],
                }
            )

        def screen(self, criteria, observed_before):
            self.calls.append(("screen", criteria, observed_before))
            return WireResult({"release_id": "release-1", "matches": [], "next_after": None})

        def credits(self, observed_before):
            self.calls.append(("credits", observed_before))
            return WireResult(
                {"balance": 2, "entries": [], "standing_slots": [], "next_after": None}
            )

    commons_client = FakeCommonsClient()

    class FakeBrowserBridge:
        def __init__(self):
            self.closed = {}
            self.inputs = []

        def open(self, url, viewport, purpose):
            self.opened = (url, viewport, purpose)
            return {"session_id": "browser-1", "origin": "https://login.example.invalid"}

        def knows(self, session_id):
            return session_id == "browser-1"

        def frames(self, session_id):
            assert session_id == "browser-1"
            yield 1234, b"fixture-jpeg", "https://login.example.invalid"

        def close_reason(self, session_id):
            return self.closed.get(session_id, "callback_received")

        def input(self, session_id, events):
            self.inputs.append((session_id, events))

        def close(self, session_id, reason):
            self.closed[session_id] = reason

    browser_bridge = FakeBrowserBridge()

    def start(argv):
        started.append(list(argv))
        running.add(7001)
        return 7001

    context = ServeContext(
        home=box_home,
        env={},
        now=lambda: datetime(2026, 9, 3, 12, 0, tzinfo=UTC),
        pid_alive=lambda pid: pid in running,
        start_process=start,
        signal_process=lambda pid: signalled.append(pid),
        provider_status=lambda: (True, "fixture provider is available."),
        loopback_status=lambda: (True, "fixture loopback is reachable."),
        tunnel_status=lambda: (True, "fixture tunnel is direct."),
        unit_fragments=lambda: (True, "fixture units are paper-only.", ("--market broker",)),
        chat_adapter=lambda _provider, _model, _transcript, _frame: (
            {"kind": "text", "text": "fixture reply"},
            {"kind": "done", "model": "fixture-model"},
        ),
        provider_login_start=lambda: {
            "login_id": "login-1",
            "url": "https://login.example.invalid/device",
            "code": "ABCD-WXYZ",
            "expires_at": "2026-09-03T12:15:00Z",
        },
        provider_browser_login_start=lambda viewport: {
            "login_id": "login-browser-1",
            "session_id": "browser-1",
            "origin": "https://login.example.invalid",
        },
        provider_login_status=lambda login_id: {"login_id": login_id, "state": "pending"},
        codex_install=lambda: {
            "code": "CODEX_INSTALLED",
            "path": "/fixture/bin/codex",
            "release": "rust-v0.0.0",
            "sha256": "f" * 64,
            "reason": "fixture installed",
        },
        broker_connect_start=lambda server_url, _scheme: {
            "authorization_url": "https://login.example.invalid/authorize?state=test",
            "connect_id": "connect-1",
            "redirect_uri": "http://127.0.0.1:48123/tick/callback",
            "disclosure": "fixture disclosure",
        },
        broker_connect_complete=lambda connect_id, _url: {
            "connect_id": connect_id,
            "state": "pending",
        },
        broker_connect_status=lambda connect_id: {
            "connect_id": connect_id,
            "state": "pending",
        },
        browser_ceremony_url=lambda _purpose: "https://login.example.invalid/authorize?state=test",
        browser_bridge=browser_bridge,
        broker_profile_operation=lambda action, body: {"action": action, "body": dict(body)},
        commons_client=lambda: commons_client,
        metadata=type("FixtureMetadata", (), {"tags": lambda self: frozenset()})(),
    )
    sleeps: list[float] = []
    server = make_server(
        "127.0.0.1",
        0,
        context=context,
        monotonic=lambda: 100.0,
        sleeper=sleeps.append,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, secret, started, signalled, running, sleeps
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def request(server, method: str, path: str, *, secret: str | None, body=None):
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
    headers = {}
    if secret is not None:
        headers["Authorization"] = f"Bearer {secret}"
    encoded = None
    if body is not None:
        encoded = json.dumps(body)
        headers["Content-Type"] = "application/json"
    connection.request(method, path, body=encoded, headers=headers)
    response = connection.getresponse()
    payload = json.loads(response.read())
    connection.close()
    return response.status, payload
