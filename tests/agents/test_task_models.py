"""Model discovery is account-specific metadata, with no inference or approvals."""

from __future__ import annotations

import io
import json
import stat

import httpx2 as httpx
import pytest

from tick.agents import codex_client, task_models
from tick.agents.credentials import anthropic_key, environment_key, save_anthropic_key


def catalog():
    return {
        "providers": [
            {
                "id": "codex",
                "connected": True,
                "models": [{"provider": "codex", "model": "fixture-a"}],
            },
            {
                "id": "anthropic",
                "connected": True,
                "models": [{"provider": "anthropic", "model": "fixture-b"}],
            },
        ],
        "presets": {},
    }


def choices():
    return {
        tier: {
            "provider": "codex" if tier == "small" else "anthropic",
            "model": "fixture-a" if tier == "small" else "fixture-b",
        }
        for tier in task_models.TIERS
    }


def test_presets_mix_providers_and_require_available_models(tmp_path):
    wanted = choices()
    assert task_models.save_presets(tmp_path, wanted, catalog())["presets"] == wanted
    assert task_models.load_presets(tmp_path) == wanted
    assert stat.S_IMODE((tmp_path / "providers/task-presets.json").stat().st_mode) == 0o600
    wanted["small"]["model"] = "invented-model"
    with pytest.raises(ValueError, match="small model is unavailable"):
        task_models.save_presets(tmp_path, wanted, catalog())
    assert task_models.load_presets(tmp_path) == choices(), "failed updates preserve saved choices"
    with pytest.raises(ValueError, match="small, medium, and large"):
        task_models.validate_choices({"small": choices()["small"]}, catalog())


def test_key_stays_private_on_box_and_environment_keeps_precedence(tmp_path, monkeypatch):
    save_anthropic_key(tmp_path, "fixture-private-key")
    path = tmp_path / "providers/anthropic.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert anthropic_key(tmp_path, {}) == "fixture-private-key"
    assert anthropic_key(tmp_path, {"ANTHROPIC_API_KEY": "environment-key"}) == "environment-key"
    monkeypatch.setenv("TICK_HOME", str(tmp_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert environment_key() == "fixture-private-key"
    path.write_text('{"unexpected":"fixture-private-key"}')
    with pytest.raises(ValueError) as caught:
        anthropic_key(tmp_path, {})
    assert "fixture-private-key" not in str(caught.value)


def test_anthropic_paginates_actual_models_and_never_follows_redirects():
    calls = []

    def reply(request):
        assert request.url.host == "api.anthropic.com"
        assert request.headers["x-api-key"] == "fixture-key"
        calls.append(request.url.params)
        if len(calls) == 1:
            return httpx.Response(
                200,
                json={
                    "data": [{"id": "first", "display_name": "First"}],
                    "has_more": True,
                    "last_id": "first",
                },
            )
        return httpx.Response(
            200, json={"data": [{"id": "second", "display_name": "Second"}], "has_more": False}
        )

    result = task_models.anthropic_models("fixture-key", transport=httpx.MockTransport(reply))
    assert [row["model"] for row in result] == ["first", "second"]
    assert calls[1]["after_id"] == "first"
    with pytest.raises(ValueError) as caught:
        task_models.anthropic_models(
            "fixture-secret",
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    302, headers={"location": "https://other.invalid"}, text="fixture-secret"
                )
            ),
        )
    assert "fixture-secret" not in str(caught.value)


def test_failed_provider_does_not_hide_other_provider_or_saved_choices(tmp_path):
    task_models.save_presets(tmp_path, choices(), catalog())

    def unavailable(_):
        raise OSError("private internal failure")

    result = task_models.discover_catalog(
        tmp_path,
        {"ANTHROPIC_API_KEY": "fixture"},
        codex=unavailable,
        anthropic=lambda _: [{"provider": "anthropic", "model": "fixture-b"}],
    )
    assert not result["providers"][0]["connected"]
    assert result["providers"][1]["connected"]
    assert result["presets"] == choices()
    assert "private internal failure" not in json.dumps(result)


def test_codex_rpc_only_reads_metadata_and_refuses_server_tool_requests(monkeypatch):
    class Input(io.StringIO):
        def close(self):
            self.saved = self.getvalue()
            super().close()

    class Process:
        stdin = Input()
        stdout = io.StringIO(
            "\n".join(
                json.dumps(row)
                for row in [
                    {"id": 1, "result": {"userAgent": "fixture"}},
                    {"id": 2, "result": {"account": {"type": "chatgpt"}}},
                    {"id": 99, "method": "item/commandExecution/requestApproval", "params": {}},
                    {
                        "id": 3,
                        "result": {
                            "data": [
                                {"model": "actual-a", "displayName": "Available A"},
                                {"model": "hidden", "hidden": True},
                            ],
                            "nextCursor": "page2",
                        },
                    },
                    {"id": 4, "result": {"data": [{"model": "actual-b"}], "nextCursor": None}},
                ]
            )
            + "\n"
        )
        terminated = False

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            return 0

    process = Process()

    def spawn(argv, **kwargs):
        assert argv == ["codex", "app-server", "--listen", "stdio://"]
        assert kwargs["env"] == {"PATH": "/fixture/bin"}
        return process

    monkeypatch.setattr(codex_client.subprocess, "Popen", spawn)
    models = task_models.codex_models({"PATH": "/fixture/bin"})
    assert [model["model"] for model in models] == ["actual-a", "actual-b"]
    requests = [json.loads(line) for line in process.stdin.saved.splitlines()]
    assert [row["method"] for row in requests if "method" in row] == [
        "initialize",
        "initialized",
        "account/read",
        "model/list",
        "model/list",
    ]
    assert next(row for row in requests if row.get("id") == 99)["error"]["code"] == -32601
    assert process.terminated
