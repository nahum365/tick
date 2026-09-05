"""Tick's own Codex home: the login, config and state Codex keeps for Tick alone.

Every Codex process Tick starts (login, model discovery, app-server chats, the
structured ``codex exec`` proposals) runs with ``CODEX_HOME`` pointing here, so
the person's personal Codex setup, its MCP servers, plugins, hooks, rules and
history, never loads into an agent, and nothing Tick does touches that setup.
Codex writes ``auth.json`` into this directory itself on ``codex login``; Tick
never reads or moves a credential. The one file Tick authors is ``config.toml``,
rewritten on every start so it cannot drift.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

__all__ = ["CODEX_HOME_DIRNAME", "codex_environment", "codex_home", "ensure_codex_home"]

CODEX_HOME_DIRNAME = "codex"

#: Tick-authored, complete. No MCP servers here: each chat thread names the box
#: tool server it may use when it starts, and nothing else exists to load.
_CONFIG = """# Written by Tick on every start. Edits are overwritten.
# This is Tick's private Codex home; your personal ~/.codex is not read here.

[history]
persistence = "none"
"""


def codex_home(home: Path) -> Path:
    """Where Tick's Codex keeps its login and state: ``TICK_HOME/codex``."""
    return home / CODEX_HOME_DIRNAME


def ensure_codex_home(home: Path) -> Path:
    """Create the private home and (re)write Tick's config; the login file is Codex's."""
    directory = codex_home(home)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory.chmod(0o700)
    config = directory / "config.toml"
    if not config.exists() or config.read_text(encoding="utf-8") != _CONFIG:
        partial = config.with_name("config.toml.partial")
        partial.write_text(_CONFIG, encoding="utf-8")
        partial.chmod(0o600)
        partial.replace(config)
    return directory


def codex_environment(environ: Mapping[str, str], home: Path) -> dict[str, str]:
    """The environment for a Codex process: Tick's home, never an inherited one.

    A developer's shell may export its own ``CODEX_HOME``; Tick overrides it so
    the runtime's Codex cannot be redirected to someone else's login or config.
    ``TICK_HOME/bin`` leads ``PATH`` so the box-installed CLI is the one found.
    """
    env = dict(environ)
    env["CODEX_HOME"] = str(codex_home(home))
    home_bin = str(home / "bin")
    parts = [part for part in env.get("PATH", "").split(os.pathsep) if part]
    if home_bin not in parts:
        parts.insert(0, home_bin)
    env["PATH"] = os.pathsep.join(parts)
    return env
