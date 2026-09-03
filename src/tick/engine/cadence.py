"""How often a spec may run — a floor Tick enforces on itself.

Robinhood's Customer Agreement §29 ("API & MCP") reserves the right to
terminate MCP connectivity for "excessive market data usage" and does not
define the threshold. An undefined limit is one you stay well clear of, so the
conservative cadence is a property of the runtime rather than advice in the
documentation:

- `daily_open` and `daily_close` are the intended cadences and always pass;
- `every_n_minutes` has a floor of `MIN_CADENCE_MINUTES`, and a spec below it
  is refused before the tick runs, with a sentence that says what to change.

The check runs at the start of every evaluation rather than only at spec load,
because the thing being protected is the *connection*, and a spec can reach a
running agent by more paths than the loader (an edited file, a compiled spec,
a future remote). The refusal is an exception, not a per-rule refusal value:
running the spec at all is what is not permitted.
"""

from __future__ import annotations

from tick.spec import Cadence, EveryNMinutes

from .errors import CadenceRefused

__all__ = ["MIN_CADENCE_MINUTES", "check_cadence", "describe_cadence"]

#: The fastest Tick will poll, whatever the spec asks for.
MIN_CADENCE_MINUTES = 5


def check_cadence(cadence: Cadence) -> None:
    """Raise `CadenceRefused` if this cadence polls faster than Tick permits."""
    if isinstance(cadence, EveryNMinutes) and cadence.n < MIN_CADENCE_MINUTES:
        raise CadenceRefused(
            f"cadence every_n_minutes({cadence.n}) is faster than Tick's floor of "
            f"{MIN_CADENCE_MINUTES} minutes: Robinhood may terminate MCP access for "
            f"excessive market data usage, and the threshold is not published. "
            f"Use every_n_minutes({MIN_CADENCE_MINUTES}) or slower, or daily_open / "
            f"daily_close."
        )


def describe_cadence(cadence: Cadence) -> str:
    """Plain words for a cadence, for the CLI and notifications."""
    if isinstance(cadence, EveryNMinutes):
        return f"every {cadence.n} minutes while the market is open"
    if cadence.kind == "daily_open":
        return "once, just after the open"
    return "once, shortly before the close"
