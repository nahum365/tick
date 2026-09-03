"""The ET market clock: when a US equities regular session is open.

Market logic runs on Eastern time, and every datetime that crosses this
module's surface is timezone-aware. A naive datetime is refused rather than
localised, because "09:30" with no zone is a different instant in every office
that reads it.

**The holiday list is hand-entered and partial.** It covers 2026 only:
`HOLIDAYS_2026` (full closures) and `EARLY_CLOSES_2026` (1:00 pm ET). Asked
about any other year, the clock raises `CalendarUnavailable` rather than
falling back to "every weekday is a session". That refusal is deliberate and it
is the honest direction: a runtime that assumed a session on Thanksgiving would
place orders into a shut market, and a runtime that assumed a 4:00 pm close on
an early-close day would evaluate a `daily_close` cadence three hours after the
bell. A published calendar feed replaces this list before anything runs live;
until then the limit is visible, loud, and dated.

The rules the clock encodes:

- a session day is a weekday that is not in the holiday set;
- regular hours run 09:30 to 16:00 ET, or 09:30 to 13:00 on an early-close day;
- `is_open` is half-open — 09:30 is open, 16:00 is not, because an order timed
  at the closing bell does not execute in the regular session.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from .errors import CalendarUnavailable

__all__ = [
    "EARLY_CLOSES_2026",
    "EARLY_CLOSE_TIME",
    "EASTERN",
    "HOLIDAYS_2026",
    "REGULAR_CLOSE",
    "REGULAR_OPEN",
    "MarketClock",
]

#: Every market judgement is made in this zone.
EASTERN = ZoneInfo("America/New_York")

REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)

#: The close on a half day (the sessions before Independence Day, after
#: Thanksgiving, and before Christmas, when they fall on a weekday).
EARLY_CLOSE_TIME = time(13, 0)

#: US equities full closures in 2026. PARTIAL DATA, hand-entered on 2026-08-31
#: from the NYSE's published rules (fixed dates observed on the nearest weekday
#: when they fall at a weekend, plus Good Friday). It is not a feed and it is
#: not authoritative; see the module docstring.
HOLIDAYS_2026: frozenset[date] = frozenset(
    {
        date(2026, 1, 1),  # New Year's Day (Thursday)
        date(2026, 1, 19),  # Martin Luther King Jr. Day
        date(2026, 2, 16),  # Washington's Birthday
        date(2026, 4, 3),  # Good Friday
        date(2026, 5, 25),  # Memorial Day
        date(2026, 6, 19),  # Juneteenth (Friday)
        date(2026, 7, 3),  # Independence Day observed (July 4 is a Saturday)
        date(2026, 9, 7),  # Labor Day
        date(2026, 11, 26),  # Thanksgiving Day
        date(2026, 12, 25),  # Christmas Day (Friday)
    }
)

#: 2026 half days: the regular session ends at 13:00 ET. Same provenance and
#: the same caveat as `HOLIDAYS_2026`.
EARLY_CLOSES_2026: frozenset[date] = frozenset(
    {
        date(2026, 11, 27),  # the day after Thanksgiving
        date(2026, 12, 24),  # Christmas Eve
    }
)

#: How far `next_open` / `next_close` will walk before giving up. Longer than
#: any run of closures the calendar can contain, short enough that a bug in the
#: calendar surfaces as a refusal rather than a spin.
_SEARCH_DAYS = 10


class MarketClock:
    """A US equities regular-session calendar for a stated set of years.

    The calendar is a constructor argument, not a built-in: tests hand it a
    calendar they wrote, and `for_2026()` is the one place the shipped list is
    chosen. `years` is what the clock claims to know about; a date outside it
    raises rather than being answered from a weekday check.
    """

    def __init__(
        self,
        *,
        holidays: Iterable[date],
        early_closes: Iterable[date],
        years: Iterable[int],
    ) -> None:
        self._holidays = frozenset(holidays)
        self._early_closes = frozenset(early_closes)
        self._years = frozenset(years)
        if not self._years:
            raise ValueError("a market clock must name the years its calendar covers")
        overlap = self._holidays & self._early_closes
        if overlap:
            raise ValueError(
                f"{sorted(overlap)} are listed as both closed and early-closing; a day "
                f"the market is shut has no closing time"
            )
        outside = {
            day for day in self._holidays | self._early_closes if day.year not in self._years
        }
        if outside:
            raise ValueError(
                f"{sorted(outside)} fall outside the calendar's years "
                f"({sorted(self._years)}), so they would never be consulted"
            )

    @classmethod
    def for_2026(cls) -> MarketClock:
        """The shipped calendar: 2026 only, partial, documented above."""
        return cls(holidays=HOLIDAYS_2026, early_closes=EARLY_CLOSES_2026, years={2026})

    @property
    def years(self) -> frozenset[int]:
        """The years this clock will answer about."""
        return self._years

    # ------------------------------------------------------------------
    # Days
    # ------------------------------------------------------------------

    def _require_year(self, day: date) -> None:
        if day.year not in self._years:
            raise CalendarUnavailable(
                f"{day.isoformat()} is in {day.year} and this clock's market calendar "
                f"covers {', '.join(str(year) for year in sorted(self._years))} only. "
                f"Tick will not guess a trading calendar; load one for {day.year} first."
            )

    def is_session_day(self, day: date) -> bool:
        """True when the market holds a regular session on `day`."""
        self._require_year(day)
        return day.weekday() < 5 and day not in self._holidays

    def is_early_close(self, day: date) -> bool:
        """True when `day` is a half day. False for any non-session day."""
        return self.is_session_day(day) and day in self._early_closes

    def session_bounds(self, day: date) -> tuple[datetime, datetime] | None:
        """The ET open and close of `day`'s regular session, or `None` if closed.

        `None` means "there is no session", which is a different answer from
        "the session is empty"; every caller here matches on it explicitly.
        """
        if not self.is_session_day(day):
            return None
        closes_at = EARLY_CLOSE_TIME if day in self._early_closes else REGULAR_CLOSE
        return (
            datetime.combine(day, REGULAR_OPEN, tzinfo=EASTERN),
            datetime.combine(day, closes_at, tzinfo=EASTERN),
        )

    # ------------------------------------------------------------------
    # Moments
    # ------------------------------------------------------------------

    @staticmethod
    def eastern(now: datetime) -> datetime:
        """`now` in Eastern time. A naive moment is refused, never localised."""
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError(
                "the market clock needs a timezone-aware moment; a naive datetime is a "
                "time in an unstated zone and cannot be placed in a session"
            )
        return now.astimezone(EASTERN)

    def session_date(self, now: datetime) -> date:
        """The ET calendar date of `now` — the day a session belongs to."""
        return self.eastern(now).date()

    def is_open(self, now: datetime) -> bool:
        """True when the regular session is running at `now` (open inclusive, close not)."""
        moment = self.eastern(now)
        bounds = self.session_bounds(moment.date())
        if bounds is None:
            return False
        opens_at, closes_at = bounds
        return opens_at <= moment < closes_at

    def next_open(self, now: datetime) -> datetime:
        """The earliest session open at or after `now`, in ET.

        At-or-after, not strictly-after: asked at 09:30:00 on a session day the
        answer is that same instant, because that is when the market opens and
        a scheduler that skipped to tomorrow would lose a day.
        """
        return self._next(now, index=0)

    def next_close(self, now: datetime) -> datetime:
        """The earliest session close at or after `now`, in ET."""
        return self._next(now, index=1)

    def _next(self, now: datetime, *, index: int) -> datetime:
        moment = self.eastern(now)
        day = moment.date()
        for _ in range(_SEARCH_DAYS):
            bounds = self.session_bounds(day)
            if bounds is not None and bounds[index] >= moment:
                return bounds[index]
            day = day + timedelta(days=1)
            self._require_year(day)
        raise CalendarUnavailable(
            f"no session boundary was found in the {_SEARCH_DAYS} days after "
            f"{moment.isoformat()}; the calendar loaded here does not describe a "
            f"market that opens"
        )
