"""Where the runtime's state lives: `TICK_HOME`, and the paths under it.

Everything Tick keeps — the record, and later the token store and the user's
own agent prompts — lives in one directory on the user's machine. That is the
architecture that makes the vendor posture true rather than promised (CLAUDE.md
invariant 1): there is no Tick-side store to which any of it is written, so
there is nothing for a Tick service to hold.

`tick_home` takes the environment as an argument instead of reading
`os.environ` itself. A test that has to monkeypatch a global to avoid writing
into the developer's real `~/.tick` is one forgotten patch away from doing it,
and this function is the one place that decision is made.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

__all__ = [
    "DEFAULT_TICK_HOME",
    "PRIVATE_FILE_MODE",
    "TICK_HOME_ENV",
    "agent_ledger_path",
    "ensure_private_dir",
    "tick_home",
    "write_private_file",
]

#: The environment variable that moves the whole of Tick's state elsewhere.
TICK_HOME_ENV = "TICK_HOME"

#: Where it lives when nothing says otherwise.
DEFAULT_TICK_HOME = "~/.tick"

#: Tick's state is private to the user: it describes what they own and what
#: their agents did with it.
DIR_MODE = 0o700

#: And so is every file Tick writes that describes the user's brokerage.
PRIVATE_FILE_MODE = 0o600


def tick_home(env: Mapping[str, str]) -> Path:
    """The runtime state directory named by `env`, expanded but not created.

    Pure: it makes no directory and touches no disk, so asking where state
    would go never creates it. `env` is the caller's environment mapping —
    `os.environ` in the product, a dict pointing at `tmp_path` in tests.
    """
    raw = env.get(TICK_HOME_ENV) or DEFAULT_TICK_HOME
    if not raw.strip():
        raise ValueError(
            f"{TICK_HOME_ENV} is set to an empty value; unset it to use "
            f"{DEFAULT_TICK_HOME}, or set it to a directory"
        )
    return Path(raw).expanduser()


def ensure_private_dir(path: str | os.PathLike[str]) -> Path:
    """Create `path` and any missing parents, each private to the user.

    `Path.mkdir(parents=True, mode=...)` applies the mode only to the LAST
    directory — the parents it creates on the way get the default permissions,
    which on most machines is world-readable. So the levels are created one at
    a time. Under `TICK_HOME` those parents are `agents/<agent id>`, and the
    list of an account's agents is not public either.

    Directories that already exist are left exactly as they are: tightening
    someone else's directory behind their back is not this function's business.
    """
    resolved = Path(path)
    missing: list[Path] = []
    probe = resolved
    while not probe.exists():
        missing.append(probe)
        if probe.parent == probe:
            break
        probe = probe.parent
    for directory in reversed(missing):
        directory.mkdir(mode=DIR_MODE, exist_ok=True)
    return resolved


def write_private_file(path: str | os.PathLike[str], text: str) -> Path:
    """Write `text` to `path` readable by the owner and by nobody else.

    The mode is applied by `os.open` at creation rather than by a `chmod`
    afterwards: a file created world-readable and tightened a microsecond later
    was world-readable for a microsecond, and on a shared machine that is the
    whole of the exposure. It is re-asserted on every write, because `O_CREAT`
    applies a mode only to a file that did not already exist and these files
    are rewritten — a token on every refresh, a profile on every confirmation.

    Used for everything under `TICK_HOME` that describes the user's brokerage:
    the OAuth grant, the registered client, and the broker profile, which
    carries the account id.
    """
    resolved = Path(path)
    ensure_private_dir(resolved.parent)
    descriptor = os.open(resolved, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, PRIVATE_FILE_MODE)
    try:
        os.fchmod(descriptor, PRIVATE_FILE_MODE)
    except BaseException:
        os.close(descriptor)
        raise
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)
    return resolved


def agent_ledger_path(home: str | os.PathLike[str], agent_id: str) -> Path:
    """The ledger file for one agent: `<home>/agents/<agent_id>/records.jsonl`.

    One file per agent, never one shared file, so an agent's record can be read,
    verified and quoted on its own — and so two agents cannot interleave into a
    chain where a break in one stops the other from recording anything.
    """
    if not agent_id.strip():
        raise ValueError("an agent id is required to name its ledger")
    if agent_id != Path(agent_id).name or agent_id in {".", ".."}:
        raise ValueError(
            f"{agent_id!r} is not a directory name; an agent id must not contain a path "
            f"separator, or it would write its record outside its own directory"
        )
    return Path(home) / "agents" / agent_id / "records.jsonl"
