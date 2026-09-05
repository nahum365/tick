"""Doctor reports every live gate in stable order and never hides a refusal."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from typer.testing import CliRunner

from tick.cli import app
from tick.runtime import RunLease, acknowledge_demotion, run_doctor, save_run_lease
from tick.serve import doctor as doctor_observations

AT = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)


def report(home, *, fragments=("--market broker",)):
    return run_doctor(
        home,
        now=AT,
        provider_status=lambda: (False, "run `codex login`, then retry."),
        loopback_status=lambda: (False, "start `tick serve`, then retry."),
        tunnel_status=lambda: (False, "start `tick tunnel`, then retry."),
        unit_fragments=lambda: (True, "the unit is readable.", fragments),
        pid_alive=lambda _pid: False,
    )


def test_fresh_home_reports_every_check_in_the_documented_order(tick_home):
    tick_home.mkdir(mode=0o700)

    result = report(tick_home)

    assert [check.name for check in result.checks] == [
        "TICK_HOME modes",
        "provider codex",
        "broker grant",
        "broker profile",
        "broker proof",
        "agent ledgers",
        "pairing secret",
        "serve loopback",
        "direct tunnel",
        "persistent launch units",
        "approval window",
        "agent paper/live state",
        "reboot demotion",
    ]
    assert all(check.status == "refuse" for check in result.checks)
    assert not result.ready
    refusal_text = "\n".join(check.line() for check in result.checks if check.status == "refuse")
    assert "codex login" in refusal_text
    assert "tick connect robinhood" in refusal_text
    assert "tick pair new" in refusal_text
    assert "tick serve" in refusal_text


def test_doctor_command_prints_refusals_and_exits_nonzero_on_a_fresh_home(tick_home, monkeypatch):
    monkeypatch.setattr(
        doctor_observations,
        "codex_login_status",
        lambda _home=None: (False, "run `codex login`, then retry."),
    )
    monkeypatch.setattr(
        doctor_observations,
        "loopback_status",
        lambda _home, _port: (False, "start `tick serve`, then retry."),
    )
    monkeypatch.setattr(
        doctor_observations,
        "systemd_unit_fragments",
        lambda: (False, "inspect systemd, then retry.", ()),
    )

    result = CliRunner().invoke(app, ["doctor"])

    assert result.exit_code == 1
    lines = result.output.splitlines()
    assert len(lines) == 13
    assert all(line.startswith("refuse:") for line in lines)
    assert all("." in line for line in lines)


def test_codex_observation_requires_the_code_mode_host(monkeypatch):
    monkeypatch.setattr(
        doctor_observations.shutil,
        "which",
        lambda name: "/fixture/codex" if name == "codex" else None,
    )

    available, reason = doctor_observations.codex_login_status()

    assert available is False
    assert "Code Mode host is missing" in reason
    assert "tick provider install codex" in reason


def test_codex_observation_checks_login_after_both_executables(monkeypatch):
    monkeypatch.setattr(
        doctor_observations.shutil,
        "which",
        lambda name: f"/fixture/{name}",
    )
    calls = []

    def run(argv, **_kwargs):
        calls.append(argv)
        if argv[-1] == "--help":
            return SimpleNamespace(returncode=0, stdout="login status", stderr="")
        return SimpleNamespace(returncode=0, stdout="logged in", stderr="")

    monkeypatch.setattr(doctor_observations.subprocess, "run", run)

    available, reason = doctor_observations.codex_login_status()

    assert available is True
    assert reason == "logged in"
    assert calls == [
        ["/fixture/codex", "login", "--help"],
        ["/fixture/codex", "login", "status"],
    ]


def test_persistent_live_in_a_unit_is_a_refusal_with_the_detection_limit(tick_home):
    tick_home.mkdir(mode=0o700)

    result = report(tick_home, fragments=("ExecStart=tick run XYZ --live",))

    check = next(item for item in result.checks if item.name == "persistent launch units")
    assert check.status == "refuse"
    assert "remove persistent --live" in check.sentence
    assert "cannot detect every root-owned script" in check.sentence


def test_reboot_demotion_remains_a_refusal_until_that_observation_is_acknowledged(tick_home, agent):
    save_run_lease(
        tick_home,
        RunLease(
            agent_id=agent.agent_id,
            run_id="run-before-reboot",
            boot_id="a-previous-boot",
            pid=7001,
            mode="live",
            approval=agent.state.approval,
            launch_source="api",
            started_at=AT,
            previous_run_id=None,
            previous_run_mode=None,
            previous_run_boot_id=None,
        ),
    )
    first = report(tick_home)
    check = next(item for item in first.checks if item.name == "reboot demotion")
    assert check.status == "refuse"
    assert "inspect it" in check.sentence

    acknowledge_demotion(tick_home, "run-before-reboot", check.detail["observations"])
    second = report(tick_home)
    assert next(item for item in second.checks if item.name == "reboot demotion").status == "ok"
