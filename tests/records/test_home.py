"""Where runtime state goes, and the ways it must not go somewhere else.

Every test here passes its own environment mapping. That is the point of
`tick_home` taking one: a test that has to remember to patch a global is a test
that will one day write into the developer's real `~/.tick`, and the ledger it
would write into is append-only.
"""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from tick.records import (
    DEFAULT_TICK_HOME,
    TICK_HOME_ENV,
    agent_ledger_path,
    ensure_private_dir,
    tick_home,
)


def test_the_environment_decides_where_state_lives(tmp_path: Path):
    assert tick_home({TICK_HOME_ENV: str(tmp_path / "elsewhere")}) == tmp_path / "elsewhere"


def test_an_unset_variable_falls_back_to_the_documented_default():
    assert tick_home({}) == Path(DEFAULT_TICK_HOME).expanduser()
    assert str(tick_home({})).endswith("/.tick")


def test_a_tilde_is_expanded():
    assert tick_home({TICK_HOME_ENV: "~/somewhere"}) == Path("~/somewhere").expanduser()


def test_asking_where_state_would_go_does_not_create_it(tmp_path: Path):
    """Pure: a query about a path is not a decision to occupy it."""
    home = tick_home({TICK_HOME_ENV: str(tmp_path / "unmade")})
    assert not home.exists()


def test_an_empty_value_is_refused_rather_than_silently_meaning_the_default():
    with pytest.raises(ValueError, match="empty value"):
        tick_home({TICK_HOME_ENV: "   "})


def test_a_ledger_path_is_one_file_per_agent(tmp_path: Path):
    assert agent_ledger_path(tmp_path, "dip-buyer") == (
        tmp_path / "agents" / "dip-buyer" / "records.jsonl"
    )


@pytest.mark.parametrize("agent_id", ["../escape", "a/b", "/absolute", "", ".", ".."])
def test_an_agent_id_that_is_not_a_directory_name_is_refused(agent_id, tmp_path: Path):
    """An id with a separator in it writes its record outside its own directory."""
    with pytest.raises(ValueError):
        agent_ledger_path(tmp_path, agent_id)


def test_a_created_directory_is_private_to_the_user(tmp_path: Path):
    created = ensure_private_dir(tmp_path / "home" / "agents" / "one")
    assert created.is_dir()
    assert stat.S_IMODE(created.stat().st_mode) == 0o700


def test_creating_an_existing_directory_is_not_an_error(tmp_path: Path):
    ensure_private_dir(tmp_path / "twice")
    assert ensure_private_dir(tmp_path / "twice").is_dir()


def test_the_parents_it_creates_are_private_too(tmp_path: Path):
    """`mkdir(parents=True, mode=...)` applies the mode only to the last level.

    The intermediate `agents/` directory lists which agents an account runs;
    it is created here, so it is created private.
    """
    ensure_private_dir(tmp_path / "home" / "agents" / "one")
    for level in ("home", "home/agents", "home/agents/one"):
        assert stat.S_IMODE((tmp_path / level).stat().st_mode) == 0o700
