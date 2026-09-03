"""The transcript replay path produces a visible, adoptable draft."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.test_product_constraints import REAL_TICKERS
from tick.cli import app
from tick.runtime import AgentRun

from .conftest import FakeClient, direct_payload, load_conversation

runner = CliRunner()


def test_rule_replay_then_adopt_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, install_fake
):
    home = tmp_path / "home"
    monkeypatch.setenv("TICK_HOME", str(home))
    conversation = load_conversation("rule.json")
    install_fake(
        FakeClient(
            [direct_payload(turn) for turn in conversation["turns"]],
            conversation["model_reported"],
        )
    )
    transcript = tmp_path / "answers.txt"
    transcript.write_text(
        "\n".join(turn["answer"] for turn in conversation["turns"]) + "\n",
        encoding="utf-8",
    )

    interview = runner.invoke(
        app,
        [
            "agent",
            "interview",
            "--provider",
            "codex",
            "--kind",
            "rule",
            "--transcript",
            str(transcript),
        ],
    )

    assert interview.exit_code == 0, interview.output
    draft_id = interview.output.splitlines()[0].split()[1]
    assert "Provenance" in interview.output
    assert "nothing exists as an agent until you adopt it" in interview.output
    assert AgentRun.list_ids(home) == []

    shown = runner.invoke(app, ["agent", "draft", "show", draft_id])
    assert shown.exit_code == 0, shown.output
    assert "price-rule" in shown.output
    assert "transcript_sha256" in shown.output

    adopted = runner.invoke(app, ["agent", "adopt", draft_id, "--max-cancels", "2"])
    assert adopted.exit_code == 0, adopted.output
    assert "adopted in paper mode" in adopted.output
    assert len(AgentRun.list_ids(home)) == 1


@pytest.mark.parametrize("command", [["agent", "interview"], ["agent", "adopt"]])
def test_new_help_is_strategy_free_and_actionable(command: list[str]):
    result = runner.invoke(app, [*command, "--help"])
    assert result.exit_code == 0, result.output
    assert REAL_TICKERS.findall(result.output) == []
    assert "ai agent" not in result.output.lower()
