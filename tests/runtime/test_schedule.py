"""The scheduler: a cadence turned into moments, and the floor it refuses below.

`et()` builds Eastern moments; the fixture calendar is the shipped 2026 one, so
the holiday cases are the real dates.
"""

from __future__ import annotations

from datetime import date, datetime

import pytest

from tick.engine import MIN_CADENCE_MINUTES, CadenceRefused
from tick.runtime import (
    DAILY_CLOSE_OFFSET,
    DAILY_OPEN_OFFSET,
    CalendarUnavailable,
    MarketClock,
    Scheduler,
)
from tick.spec import DailyClose, DailyOpen, EveryNMinutes

from .test_clock import et


@pytest.fixture
def scheduler() -> Scheduler:
    return Scheduler(MarketClock.for_2026())


# ----------------------------------------------------------------------
# daily_open / daily_close
# ----------------------------------------------------------------------


def test_daily_open_fires_just_after_the_bell(scheduler: Scheduler):
    assert scheduler.session_ticks(DailyOpen(), date(2026, 9, 1)) == (et(2026, 9, 1, 9, 31),)
    assert DAILY_OPEN_OFFSET.total_seconds() == 60


def test_daily_close_fires_before_the_bell_with_room_to_execute(scheduler: Scheduler):
    assert scheduler.session_ticks(DailyClose(), date(2026, 9, 1)) == (et(2026, 9, 1, 15, 50),)
    assert DAILY_CLOSE_OFFSET.total_seconds() == 600


def test_a_half_day_moves_the_close_tick_with_the_bell(scheduler: Scheduler):
    """13:00 close means 12:50, not 15:50 — three hours after the market shut."""
    assert scheduler.session_ticks(DailyClose(), date(2026, 11, 27)) == (et(2026, 11, 27, 12, 50),)


def test_a_closed_day_has_no_ticks(scheduler: Scheduler):
    assert scheduler.session_ticks(DailyOpen(), date(2026, 9, 5)) == ()  # Saturday
    assert scheduler.session_ticks(DailyClose(), date(2026, 9, 7)) == ()  # Labor Day


# ----------------------------------------------------------------------
# every_n_minutes
# ----------------------------------------------------------------------


def test_the_grid_is_anchored_at_the_open(scheduler: Scheduler):
    ticks = scheduler.session_ticks(EveryNMinutes(n=30), date(2026, 9, 1))
    assert ticks[:3] == (et(2026, 9, 1, 9, 30), et(2026, 9, 1, 10, 0), et(2026, 9, 1, 10, 30))


def test_the_grid_never_reaches_the_close(scheduler: Scheduler):
    """390 minutes at 30-minute steps: 09:30 through 15:30, and 16:00 is not a tick."""
    ticks = scheduler.session_ticks(EveryNMinutes(n=30), date(2026, 9, 1))
    assert ticks[-1] == et(2026, 9, 1, 15, 30)
    assert len(ticks) == 13
    assert all(tick < et(2026, 9, 1, 16, 0) for tick in ticks)


def test_a_half_day_grid_stops_at_the_early_close(scheduler: Scheduler):
    ticks = scheduler.session_ticks(EveryNMinutes(n=60), date(2026, 11, 27))
    assert ticks == tuple(et(2026, 11, 27, hour, 30) for hour in (9, 10, 11, 12))


# ----------------------------------------------------------------------
# The cadence floor
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 4])
def test_a_cadence_below_the_floor_is_refused_before_it_is_scheduled(scheduler: Scheduler, n: int):
    """The connection is what the floor protects, and the scheduler is what would poll it."""
    with pytest.raises(CadenceRefused) as raised:
        scheduler.next_tick(EveryNMinutes(n=n), et(2026, 9, 1, 10, 0))
    message = str(raised.value)
    assert "excessive market data usage" in message
    assert f"every_n_minutes({MIN_CADENCE_MINUTES})" in message


def test_the_floor_itself_is_allowed(scheduler: Scheduler):
    assert scheduler.next_tick(EveryNMinutes(n=MIN_CADENCE_MINUTES), et(2026, 9, 1, 10, 0)) == et(
        2026, 9, 1, 10, 5
    )


def test_the_floor_also_guards_the_per_day_listing(scheduler: Scheduler):
    with pytest.raises(CadenceRefused):
        scheduler.session_ticks(EveryNMinutes(n=1), date(2026, 9, 1))


# ----------------------------------------------------------------------
# next_tick
# ----------------------------------------------------------------------


def test_the_next_tick_is_strictly_after_now(scheduler: Scheduler):
    """A loop that has just ticked must not be handed the moment it just ran."""
    at_open = et(2026, 9, 1, 9, 31)
    assert scheduler.next_tick(DailyOpen(), at_open) == et(2026, 9, 2, 9, 31)


def test_before_the_session_the_next_tick_is_todays(scheduler: Scheduler):
    assert scheduler.next_tick(DailyOpen(), et(2026, 9, 1, 5, 0)) == et(2026, 9, 1, 9, 31)


def test_after_the_close_the_next_tick_is_the_next_session(scheduler: Scheduler):
    assert scheduler.next_tick(EveryNMinutes(n=15), et(2026, 9, 1, 18, 0)) == et(2026, 9, 2, 9, 30)


def test_over_a_weekend_the_next_tick_skips_to_monday(scheduler: Scheduler):
    assert scheduler.next_tick(DailyClose(), et(2026, 9, 11, 16, 30)) == et(2026, 9, 14, 15, 50)


def test_over_a_long_weekend_the_next_tick_skips_the_holiday(scheduler: Scheduler):
    """Friday 4 September, after the close: the next tick is Tuesday the 8th."""
    assert scheduler.next_tick(DailyOpen(), et(2026, 9, 4, 16, 30)) == et(2026, 9, 8, 9, 31)


def test_a_utc_moment_is_placed_in_the_eastern_session(scheduler: Scheduler):
    from datetime import UTC

    assert scheduler.next_tick(EveryNMinutes(n=30), datetime(2026, 9, 1, 13, 45, tzinfo=UTC)) == (
        et(2026, 9, 1, 10, 0)
    )


def test_running_off_the_calendar_refuses_rather_than_inventing_a_session(scheduler: Scheduler):
    with pytest.raises(CalendarUnavailable):
        scheduler.next_tick(DailyOpen(), et(2026, 12, 31, 16, 0))


def test_a_naive_moment_is_refused(scheduler: Scheduler):
    with pytest.raises(ValueError, match="timezone-aware"):
        scheduler.next_tick(DailyOpen(), datetime(2026, 9, 1, 10, 0))
