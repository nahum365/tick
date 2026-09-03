"""The ET market clock: sessions, holidays, half days, and the year it refuses.

Every moment in here is timezone-aware and most are written in ET, because
that is the zone the product reasons in. The UTC cases exist to prove the
conversion happens rather than being assumed.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time

import pytest

from tick.runtime import (
    EARLY_CLOSE_TIME,
    EASTERN,
    HOLIDAYS_2026,
    REGULAR_CLOSE,
    REGULAR_OPEN,
    CalendarUnavailable,
    MarketClock,
)


def et(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=EASTERN)


@pytest.fixture
def clock() -> MarketClock:
    return MarketClock.for_2026()


# ----------------------------------------------------------------------
# Session days
# ----------------------------------------------------------------------


def test_a_weekday_that_is_not_a_holiday_is_a_session_day(clock: MarketClock):
    assert clock.is_session_day(date(2026, 9, 1))  # a Tuesday


@pytest.mark.parametrize("day", [date(2026, 9, 5), date(2026, 9, 6)])
def test_the_weekend_is_not_a_session(clock: MarketClock, day: date):
    assert not clock.is_session_day(day)
    assert clock.session_bounds(day) is None


@pytest.mark.parametrize("holiday", sorted(HOLIDAYS_2026))
def test_every_listed_holiday_closes_the_market(clock: MarketClock, holiday: date):
    assert not clock.is_session_day(holiday)
    assert not clock.is_open(datetime.combine(holiday, time(11, 0), tzinfo=EASTERN))


def test_the_holiday_list_is_the_one_documented_as_partial():
    """It is hand-entered 2026 data. If it changes, the docstring's claim changes."""
    assert len(HOLIDAYS_2026) == 10
    assert all(day.year == 2026 for day in HOLIDAYS_2026)


# ----------------------------------------------------------------------
# Bounds and openness
# ----------------------------------------------------------------------


def test_a_regular_session_runs_from_the_open_to_the_close(clock: MarketClock):
    opens_at, closes_at = clock.session_bounds(date(2026, 9, 1))
    assert opens_at == et(2026, 9, 1, 9, 30)
    assert closes_at == et(2026, 9, 1, 16, 0)
    assert (opens_at.timetz().hour, opens_at.timetz().minute) == (
        REGULAR_OPEN.hour,
        REGULAR_OPEN.minute,
    )
    assert closes_at.hour == REGULAR_CLOSE.hour


def test_a_half_day_closes_early(clock: MarketClock):
    """The day after Thanksgiving 2026 ends at 13:00 ET, not 16:00."""
    assert clock.is_early_close(date(2026, 11, 27))
    _, closes_at = clock.session_bounds(date(2026, 11, 27))
    assert closes_at.hour == EARLY_CLOSE_TIME.hour
    assert clock.is_open(et(2026, 11, 27, 12, 59))
    assert not clock.is_open(et(2026, 11, 27, 13, 1))


def test_a_non_session_day_is_never_an_early_close(clock: MarketClock):
    assert not clock.is_early_close(date(2026, 11, 26))
    assert not clock.is_early_close(date(2026, 11, 28))


@pytest.mark.parametrize(
    ("moment", "open_now"),
    [
        (et(2026, 9, 1, 9, 29), False),  # one minute before the bell
        (et(2026, 9, 1, 9, 30), True),  # the bell itself is open
        (et(2026, 9, 1, 12, 0), True),
        (et(2026, 9, 1, 15, 59), True),
        (et(2026, 9, 1, 16, 0), False),  # the close is not open
        (et(2026, 9, 1, 20, 0), False),  # after hours
        (et(2026, 9, 1, 4, 0), False),  # pre-market
    ],
)
def test_openness_is_half_open_around_the_bells(clock: MarketClock, moment, open_now):
    assert clock.is_open(moment) is open_now


def test_a_utc_moment_is_converted_rather_than_read_as_local(clock: MarketClock):
    """13:30 UTC on 2026-09-01 is 09:30 ET — the open, not the middle of the night."""
    assert clock.is_open(datetime(2026, 9, 1, 13, 30, tzinfo=UTC))
    assert not clock.is_open(datetime(2026, 9, 1, 13, 29, tzinfo=UTC))


def test_a_naive_moment_is_refused(clock: MarketClock):
    with pytest.raises(ValueError, match="timezone-aware"):
        clock.is_open(datetime(2026, 9, 1, 10, 0))


def test_the_session_date_is_the_eastern_date(clock: MarketClock):
    """23:00 ET on the 1st is 03:00 UTC on the 2nd; the session is the 1st's."""
    assert clock.session_date(datetime(2026, 9, 2, 3, 0, tzinfo=UTC)) == date(2026, 9, 1)


# ----------------------------------------------------------------------
# The next boundary
# ----------------------------------------------------------------------


def test_the_next_open_before_the_bell_is_todays(clock: MarketClock):
    assert clock.next_open(et(2026, 9, 1, 6, 0)) == et(2026, 9, 1, 9, 30)


def test_the_open_itself_is_the_next_open(clock: MarketClock):
    """At-or-after: a scheduler asked at the bell must not skip the day."""
    assert clock.next_open(et(2026, 9, 1, 9, 30)) == et(2026, 9, 1, 9, 30)


def test_after_the_bell_the_next_open_is_the_following_session(clock: MarketClock):
    assert clock.next_open(et(2026, 9, 1, 9, 31)) == et(2026, 9, 2, 9, 30)


def test_the_weekend_skips_to_monday(clock: MarketClock):
    """Saturday 12 September 2026 → Monday the 14th."""
    assert clock.next_open(et(2026, 9, 12, 10, 0)) == et(2026, 9, 14, 9, 30)


def test_a_long_weekend_skips_the_holiday_too(clock: MarketClock):
    """Labor Day 2026 is Monday 7 September; the next open is Tuesday."""
    assert clock.next_open(et(2026, 9, 4, 16, 0)) == et(2026, 9, 8, 9, 30)


def test_the_next_close_on_a_half_day_is_the_early_one(clock: MarketClock):
    assert clock.next_close(et(2026, 11, 27, 9, 0)) == et(2026, 11, 27, 13, 0)


def test_the_next_close_after_the_close_is_the_following_session(clock: MarketClock):
    assert clock.next_close(et(2026, 9, 1, 16, 1)) == et(2026, 9, 2, 16, 0)


# ----------------------------------------------------------------------
# What the clock will not answer
# ----------------------------------------------------------------------


def test_a_year_outside_the_calendar_refuses_rather_than_guessing(clock: MarketClock):
    """A weekday check is not a trading calendar, and the clock will not pretend."""
    with pytest.raises(CalendarUnavailable, match="2027"):
        clock.is_open(et(2027, 1, 4, 10, 0))
    with pytest.raises(CalendarUnavailable, match="2025"):
        clock.is_session_day(date(2025, 12, 31))


def test_walking_off_the_end_of_the_calendar_refuses(clock: MarketClock):
    """The last session of 2026 has no next open, and that is said, not invented."""
    with pytest.raises(CalendarUnavailable):
        clock.next_open(et(2026, 12, 31, 16, 0))


def test_a_calendar_must_name_its_years():
    with pytest.raises(ValueError, match="years"):
        MarketClock(holidays=[], early_closes=[], years=[])


def test_a_day_cannot_be_both_shut_and_early_closing():
    with pytest.raises(ValueError, match="both closed and early-closing"):
        MarketClock(holidays=[date(2026, 7, 3)], early_closes=[date(2026, 7, 3)], years={2026})


def test_a_calendar_entry_outside_the_covered_years_is_refused():
    """An entry that would never be consulted is a typo, not a feature."""
    with pytest.raises(ValueError, match="outside the calendar's years"):
        MarketClock(holidays=[date(2025, 12, 25)], early_closes=[], years={2026})


def test_a_test_calendar_can_replace_the_shipped_one():
    """The calendar is an argument, so a test never depends on the shipped list."""
    clock = MarketClock(holidays={date(2026, 3, 4)}, early_closes=set(), years={2026})
    assert not clock.is_session_day(date(2026, 3, 4))
    assert clock.is_session_day(date(2026, 1, 1))  # a holiday the shipped list has
