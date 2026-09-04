from __future__ import annotations

import json

import pytest

from tests.agents.test_task_models import catalog, choices
from tick.agents import task_models
from tick.serve import model_settings
from tick.serve.handlers import APIError

from .conftest import request


def test_catalog_and_credential_routes_require_pairing(server_box, monkeypatch):
    server, secret, *_ = server_box
    monkeypatch.setattr(task_models, "discover_catalog", lambda *_: catalog())
    for method, path, body in [
        ("GET", "/v1/providers", None),
        ("POST", "/v1/providers/anthropic", {"api_key": "fixture-secret"}),
        ("POST", "/v1/model-presets", {"presets": choices()}),
    ]:
        status, _ = request(server, method, path, secret=None, body=body)
        assert status == 401
    status, payload = request(server, "GET", "/v1/providers", secret=secret)
    assert status == 200 and payload == catalog()


def test_connect_and_save_only_record_nonsecret_events(server_box, monkeypatch):
    server, secret, *_ = server_box
    monkeypatch.setattr(
        task_models, "anthropic_models", lambda _: catalog()["providers"][1]["models"]
    )
    monkeypatch.setattr(task_models, "discover_catalog", lambda *_: catalog())
    status, payload = request(
        server, "POST", "/v1/providers/anthropic", secret=secret, body={"api_key": "fixture-secret"}
    )
    assert status == 200 and payload["connected"]
    assert "fixture-secret" not in json.dumps(payload)
    for record in server.context.home.rglob("*.jsonl"):
        assert "fixture-secret" not in record.read_text()
    status, payload = request(
        server, "POST", "/v1/model-presets", secret=secret, body={"presets": choices()}
    )
    assert status == 200 and payload["presets"] == choices()


def test_tier_resolves_before_session_identity_and_explicit_model_stays_pinned(
    server_box, monkeypatch
):
    server, secret, *_ = server_box
    found = catalog() | {"presets": choices()}
    monkeypatch.setattr(task_models, "discover_catalog", lambda *_: found)
    status, chat = request(server, "POST", "/v1/chat", secret=secret, body={"tier": "small"})
    assert status == 201 and chat["provider"] == "codex" and chat["model"] == "fixture-a"
    for body in ({"tier": []}, {"tier": "giant"}, {"tier": "small", "provider": "codex"}):
        with pytest.raises(APIError) as caught:
            model_settings.resolve(server.context, body)
        assert caught.value.status == 400
    found["providers"][0]["connected"] = False
    with pytest.raises(APIError) as caught:
        model_settings.resolve(server.context, {"tier": "small"})
    assert caught.value.status == 409
    assert model_settings.resolve(server.context, {"provider": "codex", "model": "explicit"}) == {
        "provider": "codex",
        "model": "explicit",
    }


def test_simulation_session_is_returned_before_streaming_model_work(server_box):
    from dataclasses import replace

    from tick.serve.handlers import setup_chat_continue

    server, secret, *_ = server_box
    calls = []

    def adapter(*args):
        calls.append(args)
        return ({"kind": "text", "text": "Describe your own rules."}, {"kind": "done"})

    server.context = replace(server.context, setup_chat_adapter=adapter)
    status, saved = request(
        server,
        "POST",
        "/v1/setup/chat",
        secret=secret,
        body={
            "scope": "agent_draft",
            "provider": "codex",
            "model": "fixture-model",
            "resume": False,
            "goal": "simulation",
        },
    )
    assert status == 201 and saved["goal"] == "simulation"
    assert not calls, "a long first turn must not hide the recoverable session ID"
    events = list(setup_chat_continue(server.context, saved["chat"]["id"]))
    assert len(calls) == 1
    assert any(event["kind"] == "progress" for event in events)
