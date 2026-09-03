"""The disclosure: the facts the 2026-08-31 audit requires, in plain words.

Each assertion here is one of those facts. They are asserted on the served text
rather than on a constant, because the requirement is that a *person reading
the terminal* learns them — a fact stated only in a docstring has not been
disclosed to anybody.
"""

from __future__ import annotations

from tick.auth import DISCLOSURE_LINES, disclosure_text

TEXT = disclosure_text().lower()


def test_it_says_the_grant_reads_every_account_the_user_has():
    """Robinhood's grant is wider than what Tick uses, and the user is told first."""
    assert "all of your robinhood accounts" in TEXT
    assert "account numbers" in TEXT


def test_it_says_trading_is_confined_to_the_agentic_account():
    assert "confined" in TEXT and "agentic account" in TEXT


def test_it_says_tick_narrows_its_own_reads_and_that_this_is_ticks_restraint():
    """Voluntary scoping stated as voluntary; anything else overstates the grant."""
    assert "tick reads only the one agentic account you configure" in TEXT
    assert "that is tick restraining itself" in TEXT


def test_it_says_the_token_stays_on_this_machine():
    assert "no tick service receives it" in TEXT
    assert "0600" in TEXT


def test_it_warns_that_the_consent_screen_may_show_a_server_assigned_name():
    assert "client name their server assigned" in TEXT


def test_it_says_how_to_end_the_grant():
    """Fail safe has an exit: the user must always know what they can still do."""
    assert "revoke the grant at robinhood" in TEXT
    assert "tick disconnect robinhood" in TEXT


def test_every_line_is_a_sentence_a_person_could_read_aloud():
    for line in DISCLOSURE_LINES:
        assert line.strip() == line
        assert line.endswith((".", "…"))
