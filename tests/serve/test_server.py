from __future__ import annotations

import json
from http.client import HTTPConnection
from urllib.parse import quote

from tick.records import read
from tick.runtime import (
    ApprovalMode,
    ApprovalQueue,
    ApprovalWindow,
    Mode,
    RunLease,
    save_run_lease,
    state_summary,
)
from tick.serve.pairing import load_secret
from tick.serve.server import FailureLimiter, make_server

from .conftest import request


def test_health_is_the_only_unauthenticated_shape(server_box):
    server, secret, *_ = server_box
    status, payload = request(server, "GET", "/v1/health", secret=None)
    assert status == 200
    assert payload == {"tick": "0.0.1"}

    status, payload = request(server, "GET", "/v1/status", secret=None)
    assert status == 401
    assert set(payload) == {"code", "reason"}
    assert payload["code"] == "unauthorized"
    assert "Pair again" in payload["reason"]

    status, payload = request(server, "GET", "/v1/status", secret=secret + "x")
    assert status == 401
    assert payload["code"] == "unauthorized"


def test_failure_limiter_slows_after_five_failures():
    moments = iter([10.0, 10.25])
    sleeps: list[float] = []
    limiter = FailureLimiter(monotonic=lambda: next(moments), sleeper=sleeps.append)
    for _ in range(5):
        limiter.failed("127.0.0.1")
    limiter.before_attempt("127.0.0.1")
    limiter.before_attempt("127.0.0.1")
    assert sleeps == [0.75]


def test_status_has_the_app_join_shape(server_box, box_agent):
    server, secret, *_ = server_box
    status, payload = request(server, "GET", "/v1/status", secret=secret)
    assert status == 200
    assert set(payload) == {"version", "box_time", "agents", "provider", "broker", "ledger_ok"}
    agent = payload["agents"][0]
    assert agent["id"] == box_agent.agent_id
    assert agent["run_state"] == "unknown"
    assert agent["current_mode"] is None
    assert payload["broker"]["profile_state"] == "none"


def test_new_paper_run_exposes_reboot_demotion(server_box, box_agent, monkeypatch):
    monkeypatch.setattr("tick.serve.handlers.boot_id", lambda: "boot-two")
    save_run_lease(
        box_agent.home,
        RunLease(
            agent_id=box_agent.agent_id,
            run_id="paper-run",
            boot_id="boot-two",
            pid=7001,
            mode=Mode.PAPER,
            approval=ApprovalMode.EACH,
            launch_source="supervisor",
            started_at=box_agent.state.created_at,
            previous_run_id="live-run",
            previous_run_mode=Mode.LIVE,
            previous_run_boot_id="boot-one",
        ),
    )
    server, secret, _, _, running, _ = server_box
    running.add(7001)
    status, payload = request(server, "GET", "/v1/status", secret=secret)
    assert status == 200
    agent = payload["agents"][0]
    assert agent["current_mode"] == "paper"
    assert agent["previous_run_mode"] == "live"
    assert agent["transition"] == "reboot_demoted_live_to_paper"
    assert agent["attention_required"] is True


def test_ledger_rows_emit_at_not_ts(server_box, box_agent):
    box_agent.ledger(clock=lambda: box_agent.state.created_at).append(
        "note", {"text": "local record"}, source="runtime"
    )
    server, secret, *_ = server_box
    status, payload = request(
        server, "GET", f"/v1/agents/{box_agent.agent_id}/ledger?after=0", secret=secret
    )
    assert status == 200
    assert "at" in payload["records"][0]
    assert "ts" not in payload["records"][0]


def test_approval_route_commits_api_resolution(server_box, box_agent):
    queue = ApprovalQueue.system(box_agent.home, box_agent.agent_id)
    pending = queue.create(
        run_id="run-one",
        tick_id="tick-one",
        window=ApprovalWindow(seconds=300),
        symbol="XYZ",
        side="buy",
        qty=1,
        est_price=None,
        price_source="unavailable",
        data_class="local_fixture",
        est_notional=None,
        cage_checks=("session",),
        proposed_by="rule:one",
        intent={"symbol": "XYZ", "qty": 1},
        evidence={"price": None},
    )
    server, secret, *_ = server_box
    status, payload = request(
        server,
        "POST",
        f"/v1/approvals/{pending.approval_id}",
        secret=secret,
        body={"decision": "decline"},
    )
    assert status == 200
    assert payload["resolution"]["outcome"] == "declined"
    assert payload["resolution"]["decided_via"] == "api"

    status, payload = request(
        server,
        "POST",
        f"/v1/approvals/{pending.approval_id}",
        secret=secret,
        body={"decision": "approve"},
    )
    assert status == 409
    assert payload["code"] == "already_resolved"
    assert "already declined" in payload["reason"]
    assert set(payload) == {"code", "reason"}


def test_chat_confirmation_requires_and_records_the_transcript_hash(server_box, box_agent):
    queue = ApprovalQueue.system(box_agent.home, box_agent.agent_id)
    pending = queue.create(
        run_id="run-chat",
        tick_id="tick-chat",
        window=ApprovalWindow(seconds=300),
        symbol="XYZ",
        side="buy",
        qty=1,
        est_price=None,
        price_source="unavailable",
        data_class="local_fixture",
        est_notional=None,
        cage_checks=("session",),
        proposed_by="rule:one",
        intent={"symbol": "XYZ", "qty": 1},
        evidence={"price": None},
    )
    server, secret, *_ = server_box
    path = f"/v1/approvals/{pending.approval_id}"
    status, refused = request(
        server,
        "POST",
        path,
        secret=secret,
        body={"decision": "decline", "via": "chat"},
    )
    assert status == 400
    assert "transcript_hash" in refused["reason"]

    transcript_hash = "a" * 64
    status, payload = request(
        server,
        "POST",
        path,
        secret=secret,
        body={
            "decision": "decline",
            "via": "chat",
            "transcript_hash": transcript_hash,
        },
    )
    assert status == 200
    assert payload["resolution"]["decided_via"] == "chat"
    note = list(read(box_agent.ledger_path))[-1]
    assert note.payload["via"] == "chat"
    assert note.payload["transcript_hash"] == transcript_hash


def test_launch_is_per_process_and_idempotent(server_box, box_agent):
    server, secret, started, _, _, _ = server_box
    before = box_agent.state_path.read_bytes()
    body = {"live": False, "standing_ok": False, "idempotency_key": "one"}
    status, first = request(
        server, "POST", f"/v1/agents/{box_agent.agent_id}/launch", secret=secret, body=body
    )
    again_status, second = request(
        server, "POST", f"/v1/agents/{box_agent.agent_id}/launch", secret=secret, body=body
    )
    assert status == 202
    assert again_status == 200
    assert first == second
    assert len(started) == 1
    assert box_agent.state_path.read_bytes() == before
    assert "--live" not in started[0]


def test_stop_is_idempotent_and_keeps_first_reason(server_box, box_agent):
    server, secret, *_ = server_box
    path = f"/v1/agents/{box_agent.agent_id}/stop"
    first_status, first = request(server, "POST", path, secret=secret)
    original = box_agent.stop_path.read_bytes()
    second_status, second = request(server, "POST", path, secret=secret)
    assert first_status == second_status == 200
    assert first["reason"] == second["reason"] == "stopped from the box API"
    assert box_agent.stop_path.read_bytes() == original


def test_rotation_rejects_old_secret_on_next_request(server_box):
    server, secret, *_ = server_box
    status, payload = request(server, "POST", "/v1/pair/rotate", secret=secret)
    assert status == 200
    replacement = payload["secret"]
    assert replacement == load_secret(server.context.home)

    old_status, _ = request(server, "GET", "/v1/status", secret=secret)
    new_status, _ = request(server, "GET", "/v1/status", secret=replacement)
    assert old_status == 401
    assert new_status == 200
    assert secret not in json.dumps(payload)


def test_paired_control_routes_are_always_available(box_agent, server_box):
    running_server, secret, _, _, _, _ = server_box
    context = running_server.context
    server = make_server(
        "127.0.0.1",
        0,
        context=context,
        monotonic=lambda: 1.0,
        sleeper=lambda seconds: None,
    )
    import threading

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, payload = request(server, "GET", "/v1/agents", secret=secret)
        assert status == 200
        assert payload["agents"][0]["id"] == box_agent.agent_id
        assert request(server, "GET", "/v1/status", secret=secret)[0] == 200
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_chat_routes_persist_and_stream_typed_json_lines(server_box):
    server, secret, *_ = server_box
    status, created = request(server, "POST", "/v1/chat", secret=secret, body={"provider": "codex"})
    assert status == 201
    chat_id = created["id"]

    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
    connection.request(
        "POST",
        f"/v1/chat/{chat_id}/turns",
        body=json.dumps({"text": "hello"}),
        headers={"Authorization": f"Bearer {secret}", "Content-Type": "application/json"},
    )
    response = connection.getresponse()
    chunks = [json.loads(line) for line in response.read().splitlines()]
    connection.close()
    assert response.status == 200
    assert response.getheader("Content-Type") == "application/x-ndjson"
    assert response.getheader("Transfer-Encoding") == "chunked"
    assert [chunk["kind"] for chunk in chunks] == ["text", "done"]

    status, loaded = request(server, "GET", f"/v1/chat/{chat_id}", secret=secret)
    assert status == 200
    assert [turn["kind"] for turn in loaded["transcript"]] == ["user", "text", "done"]
    assert request(server, "DELETE", f"/v1/chat/{chat_id}", secret=secret)[0] == 200


def test_provider_device_login_and_broker_redirect_completion_are_drivable(server_box):
    server, secret, *_ = server_box
    status, login = request(server, "POST", "/v1/provider/codex/login", secret=secret)
    assert status == 202
    assert set(login) == {"login_id", "url", "code", "expires_at"}
    assert (
        request(server, "GET", f"/v1/provider/codex/login/{login['login_id']}", secret=secret)[1][
            "state"
        ]
        == "pending"
    )

    status, connect = request(server, "POST", "/v1/broker/connect", secret=secret, body={})
    assert status == 202
    assert "disclosure" in connect
    status, completed = request(
        server,
        "POST",
        f"/v1/broker/connect/{connect['connect_id']}/complete",
        secret=secret,
        body={"redirect_url": connect["redirect_uri"] + "?code=code&state=test"},
    )
    assert status == 202
    assert completed["state"] == "pending"


def test_chat_confirmed_interview_records_the_transcript_hash(server_box):
    server, secret, *_ = server_box
    transcript_hash = "b" * 64
    status, created = request(
        server,
        "POST",
        "/v1/interview",
        secret=secret,
        body={
            "provider": "codex",
            "kind": "rule",
            "model": None,
            "via": "chat",
            "transcript_hash": transcript_hash,
        },
    )
    assert status == 201
    records = list(read(server.context.home / "drafts" / created["id"] / "records.jsonl"))
    assert records[-1].payload["via"] == "chat"
    assert records[-1].payload["transcript_hash"] == transcript_hash


def test_control_center_reads_and_changes_agent_and_local_commons_state(server_box, box_agent):
    server, secret, *_ = server_box
    status, document = request(server, "GET", f"/v1/agents/{box_agent.agent_id}", secret=secret)
    assert status == 200
    assert document["document"]["name"] == box_agent.spec.name

    status, changed = request(
        server,
        "POST",
        f"/v1/agents/{box_agent.agent_id}/approval-mode",
        secret=secret,
        body={"mode": "standing"},
    )
    assert status == 200
    assert changed["approval_mode"] == "standing"

    assert request(server, "GET", "/v1/commons/status", secret=secret)[1]["opted_in"] is False
    assert request(server, "POST", "/v1/commons/keygen", secret=secret)[0] == 201
    assert (
        request(server, "POST", "/v1/commons/opt-in", secret=secret, body={"confirm": True})[0]
        == 200
    )
    assert (
        request(server, "GET", "/v1/commons/pass?ticker=XYZ", secret=secret)[1]["pass"]["ticker"]
        == "XYZ"
    )

    exported = request(
        server,
        "GET",
        f"/v1/agents/{box_agent.agent_id}/ledger/export?for-evidence=true",
        secret=secret,
    )
    assert exported[0] == 200
    assert exported[1]["for_evidence"] is True
    assert request(server, "GET", "/v1/notifications?after=0", secret=secret)[0] == 200


def test_commons_graph_screen_and_credits_forward_the_release_cutoff(server_box, monkeypatch):
    server, secret, *_ = server_box
    monkeypatch.setattr(
        "tick.serve.handlers.AgentRun.list_ids",
        lambda _home: (_ for _ in ()).throw(AssertionError("agent state was inspected")),
    )
    monkeypatch.setattr(
        "tick.serve.handlers.load_profile",
        lambda _home: (_ for _ in ()).throw(AssertionError("broker state was inspected")),
    )
    observed = "2026-09-03T11:00:00Z"

    graph_status, graph = request(
        server,
        "GET",
        f"/v1/commons/graph?ticker=XYZ&depth=2&observed_before={quote(observed)}",
        secret=secret,
    )
    criterion = quote(json.dumps({"predicate_id": "revenue", "op": "gte", "value": "10"}))
    screen_status, screen = request(
        server,
        "GET",
        f"/v1/commons/screen?criterion={criterion}&observed_before={quote(observed)}",
        secret=secret,
    )
    credits_status, credits = request(
        server,
        "GET",
        f"/v1/commons/credits?observed_before={quote(observed)}",
        secret=secret,
    )

    assert graph_status == screen_status == credits_status == 200
    assert graph["release_id"] == screen["release_id"] == "release-1"
    assert credits["balance"] == 2
    calls = server.context.commons_client().calls
    assert calls[0][0:3] == ("graph", "XYZ", 2)
    assert calls[1][0] == "screen"
    assert calls[1][1][0].predicate_id == "revenue"
    assert calls[2][0] == "credits"
    assert all(call[-1].isoformat() == "2026-09-03T11:00:00+00:00" for call in calls)


def test_empty_commons_screen_refuses_before_transport(server_box):
    server, secret, *_ = server_box

    status, payload = request(server, "GET", "/v1/commons/screen", secret=secret)

    assert status == 400
    assert payload == {
        "code": "screen_criteria_required",
        "reason": "add at least one predicate criterion; an empty screen cannot list everything.",
    }
    assert server.context.commons_client().calls == []


def test_commons_read_validation_refusals_name_the_correction(server_box):
    server, secret, *_ = server_box
    incomplete_between = quote(
        json.dumps({"predicate_id": "revenue", "op": "between", "values": ["10"]})
    )

    invalid = (
        (
            "/v1/commons/graph?depth=1",
            "ticker_required",
            "ticker must appear once. Correct it.",
        ),
        (
            "/v1/commons/graph?ticker=XYZ&depth=3",
            "depth_invalid",
            "depth must be 1 or 2. Choose one and retry.",
        ),
        (
            "/v1/commons/credits?observed_before=2026-09-03T11:00:00",
            "observed_before_invalid",
            "observed_before must include Z or an explicit offset. Correct it and retry.",
        ),
        (
            f"/v1/commons/screen?criterion={incomplete_between}",
            "screen_criterion_invalid",
            "Correct the criterion and retry.",
        ),
    )
    for path, code, sentence in invalid:
        status, payload = request(server, "GET", path, secret=secret)
        assert status == 400
        assert payload["code"] == code
        assert sentence in payload["reason"]

    assert server.context.commons_client().calls == []


def test_broker_profile_operations_reuse_the_injected_box_boundary(server_box):
    server, secret, *_ = server_box
    status, proposed = request(
        server,
        "POST",
        "/v1/broker/propose",
        secret=secret,
        body={"account": "account-placeholder", "server_url": "https://agent.robinhood.com"},
    )
    assert status == 201
    assert proposed["action"] == "propose"
    status, proven = request(
        server, "POST", "/v1/broker/prove", secret=secret, body={"probe": {"symbol": "XYZ"}}
    )
    assert status == 200
    assert proven["action"] == "prove"
    assert request(server, "GET", "/v1/broker/profile/diff", secret=secret)[1]["action"] == "diff"


def test_state_summary_still_reads_same_agent(box_agent):
    assert state_summary(box_agent)["agent_id"] == box_agent.agent_id


def test_broker_connect_passes_the_phone_redirect_scheme_and_refuses_odd_ones(server_box):
    server, secret = server_box[0], server_box[1]
    status, connect = request(
        server, "POST", "/v1/broker/connect", secret=secret, body={"redirect_scheme": "tick"}
    )
    assert status == 202 and "authorization_url" in connect
    status, body = request(
        server, "POST", "/v1/broker/connect", secret=secret, body={"redirect_scheme": "Bad Scheme"}
    )
    assert status == 400
    assert body["reason"].startswith(
        "redirect_scheme, when supplied, must be a lowercase URL scheme"
    )
