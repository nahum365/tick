"""`tick provider` and the per-order default, through the CLI.

Nothing here reaches a provider: availability is answered from an environment
and a `shutil.which` the test controls.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tick.cli import app
from tick.runtime import AgentRun, ApprovalMode

from .conftest import spec_document

runner = CliRunner()


def invoke(*args: str, **kwargs):
    return runner.invoke(app, list(args), **kwargs)


def test_provider_list_names_the_two_shipped_adapters_and_how_each_is_reached(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("tick.agents.providers.shutil.which", lambda name: "/usr/bin/codex")

    result = invoke("provider", "list")

    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert any(line.startswith("anthropic") and "not available" in line for line in lines)
    assert any(
        line.startswith("codex") and "http_key" not in line and "available" in line
        for line in lines
    )
    assert "needs: ANTHROPIC_API_KEY" in result.output
    assert "needs: codex" in result.output


def test_provider_check_shows_the_terms_note_once_and_exits_two_when_unreachable(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = invoke("provider", "check", "anthropic")
    assert result.exit_code == 2
    assert "set ANTHROPIC_API_KEY in this shell" in result.output
    assert "About anthropic's terms:" in result.output
    assert "YOUR API agreement" in result.output
    assert result.output.count("About anthropic's terms:") == 1

    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    result = invoke("provider", "check", "anthropic")
    assert result.exit_code == 0, result.output
    assert "available" in result.output


def test_provider_check_refuses_a_provider_tick_does_not_ship():
    result = invoke("provider", "check", "mystery")
    assert result.exit_code != 0
    assert "mystery" in result.output


def test_per_order_approval_is_the_default_an_agent_starts_in(tmp_path: Path):
    spec_file = tmp_path / "strategy.json"
    spec_file.write_text(json.dumps(spec_document()), encoding="utf-8")

    result = invoke("agent", "add", str(spec_file), "--max-cancels", "2")

    assert result.exit_code == 0, result.output
    agent_id = result.output.splitlines()[0].strip()
    import os

    home = Path(os.environ["TICK_HOME"])
    assert AgentRun(home, agent_id).state.approval is ApprovalMode.EACH
    assert "paper/each" in invoke("agents").output
