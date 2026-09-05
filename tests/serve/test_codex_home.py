"""Tick's private Codex home: the only home any Tick-started Codex sees."""

from __future__ import annotations

import os
import stat

from tick.serve.codex_home import codex_environment, codex_home, ensure_codex_home
from tick.serve.doctor import codex_login_status


def test_the_home_is_private_and_carries_only_ticks_config(tmp_path):
    directory = ensure_codex_home(tmp_path)

    assert directory == tmp_path / "codex" == codex_home(tmp_path)
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert sorted(p.name for p in directory.iterdir()) == ["config.toml"]
    config = (directory / "config.toml").read_text(encoding="utf-8")
    assert "mcp_servers" not in config, "threads name the box server; nothing else may load"
    assert 'persistence = "none"' in config
    assert stat.S_IMODE((directory / "config.toml").stat().st_mode) == 0o600


def test_a_hand_edited_config_is_rewritten_and_the_login_file_is_left_alone(tmp_path):
    directory = ensure_codex_home(tmp_path)
    (directory / "config.toml").write_text('[mcp_servers.shell]\ncommand = "sh"\n')
    (directory / "auth.json").write_text("{}")

    ensure_codex_home(tmp_path)

    assert "shell" not in (directory / "config.toml").read_text(encoding="utf-8")
    assert (directory / "auth.json").read_text() == "{}"


def test_the_environment_overrides_an_inherited_codex_home_and_leads_with_ticks_bin(tmp_path):
    env = codex_environment(
        {"CODEX_HOME": "/home/dev/.codex", "PATH": "/usr/bin", "OTHER": "kept"}, tmp_path
    )

    assert env["CODEX_HOME"] == str(tmp_path / "codex")
    assert env["PATH"].split(os.pathsep) == [str(tmp_path / "bin"), "/usr/bin"]
    assert env["OTHER"] == "kept"


def test_doctor_looks_for_the_login_in_ticks_home_not_the_users(tmp_path, monkeypatch):
    monkeypatch.setattr("tick.serve.doctor.shutil.which", lambda name: f"/tick/bin/{name}")
    monkeypatch.setattr(
        "tick.serve.doctor.subprocess.run",
        lambda *_a, **_k: type(
            "R", (), {"stdout": "usage: codex login", "stderr": "", "returncode": 1}
        )(),
    )
    ok, reason = codex_login_status(tmp_path)
    assert not ok and "connect Codex from the app" in reason

    ensure_codex_home(tmp_path)
    (tmp_path / "codex" / "auth.json").write_text("{}")
    ok, reason = codex_login_status(tmp_path)
    assert ok and "contents were not read" in reason
