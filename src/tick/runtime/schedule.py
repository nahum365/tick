"""When an agent evaluates next — the cadence turned into moments on a calendar.

The scheduler is a pure function of a cadence, a market calendar and a moment.
It opens nothing, sleeps nothing and remembers nothing; the loop that sleeps
lives in `runner.py`, so that "when is the next tick" can be tested without
time passing.

**The cadence floor is enforced here as well as in the engine.** Robinhood's
Customer Agreement §29 reserves the right to terminate MCP connectivity for
undefined "excessive market data usage", and an undefined limit is one you stay
well clear of. `tick.engine.check_cadence` refuses `every_n_minutes` below five
minutes before a tick evaluates; the scheduler refuses it before a tick is even
scheduled, because the thing being protected is the connection and the
scheduler is what would be hammering it. Both refusals are the same function,
so there is one floor and one sentence.

Three rules turn a cadence into moments, and each is a decision:

- **`daily_open` fires one minute after the bell**, not at it. The open auction
  is still printing at 09:30:00 and a quote taken then is the least
  representative of the day.
- **`daily_close` fires ten minutes before the close**, so that a market order
  it produces has time to reach the book inside the regular session — including
  on a half day, where the close is 13:00 and the tick is 12:50.
- **`every_n_minutes` runs on a grid anchored at the open** — 09:30, 09:30+n,
  … — and never at or after the close. Anchoring at the open rather than at
  whenever the process happened to start is what makes two runs of the same
  agent on the same day evaluate at the same moments.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from tick.engine import MIN_CADENCE_MINUTES, check_cadence
from tick.spec import Cadence, DailyClose, DailyOpen, EveryNMinutes

from .clock import MarketClock
from .errors import CalendarUnavailable

__all__ = [
    "DAILY_CLOSE_OFFSET",
    "DAILY_OPEN_OFFSET",
    "MIN_CADENCE_MINUTES",
    "Scheduler",
]

#: How long after the opening bell a `daily_open` cadence evaluates.
DAILY_OPEN_OFFSET = timedelta(minutes=1)

#: How long before the closing bell a `daily_close` cadence evaluates.
DAILY_CLOSE_OFFSET = timedelta(minutes=10)

#: How many days forward `next_tick` will look for a session before refusing.
_SEARCH_DAYS = 10


class Scheduler:
    """Turns a spec's cadence into the next moment that agent should evaluate."""

    def __init__(self, clock: MarketClock) -> None:
        self._clock = clock

    @property
    def clock(self) -> MarketClock:
        return self._clock

    def next_tick(self, cadence: Cadence, now: datetime) -> datetime:
        """The first scheduled moment strictly after `now`, in ET.

        Strictly after, because the caller asking is usually the loop that has
        just ticked; returning `now` would spin.
        """
        check_cadence(cadence)
        eastern_now = self._clock.eastern(now)
        day = eastern_now.date()
        for _ in range(_SEARCH_DAYS):
            for candidate in self.session_ticks(cadence, day):
                if candidate > eastern_now:
                    return candidate
            day = day + timedelta(days=1)
        raise CalendarUnavailable(
            f"no tick for this cadence falls in the {_SEARCH_DAYS} days after "
            f"{now.isoformat()}; the calendar loaded here does not describe a market "
            f"that opens"
        )

    def session_ticks(self, cadence: Cadence, day: date) -> tuple[datetime, ...]:
        """Every moment this cadence evaluates at on `day`, in order.

        Empty when the market is shut that day. Every moment is inside the
        regular session: at or after the open, strictly before the close, so a
        tick never evaluates into a market that cannot execute it.
        """
        check_cadence(cadence)
        bounds = self._clock.session_bounds(day)
        if bounds is None:
            return ()
        opens_at, closes_at = bounds
        if isinstance(cadence, DailyOpen):
            candidates = [opens_at + DAILY_OPEN_OFFSET]
        elif isinstance(cadence, DailyClose):
            candidates = [closes_at - DAILY_CLOSE_OFFSET]
        elif isinstance(cadence, EveryNMinutes):
            candidates = []
            moment = opens_at
            step = timedelta(minutes=cadence.n)
            while moment < closes_at:
                candidates.append(moment)
                moment = moment + step
        else:  # pragma: no cover - closed union
            raise ValueError(f"unknown cadence {cadence!r}")
        return tuple(candidate for candidate in candidates if opens_at <= candidate < closes_at)
