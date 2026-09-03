"""Errors the engine raises — the conditions that stop a tick rather than shape one.

Most things that go wrong in a tick are *values*: a missing quote becomes an
`Unavailable`, and the rule that needed it produces a `Refusal` that is
recorded and reported. The exceptions here are different in kind. They are
conditions under which the engine must not proceed at all — a cadence the
runtime is not allowed to run, a symbol nobody authorised, a market-data port
that broke its own contract. Turning one of those into a per-rule refusal
would let a tick keep running on a footing nobody agreed to.
"""

from __future__ import annotations


class EngineError(Exception):
    """Base class for every engine failure."""


class CadenceRefused(EngineError):
    """The spec asks to run more often than the runtime is permitted to.

    Robinhood may terminate MCP connectivity for undefined "excessive market
    data usage" (Customer Agreement §29). The floor is enforced before a tick
    runs, not after the connection is gone.
    """


class SymbolOutsideScope(EngineError):
    """Market data was requested for a symbol this tick has no business reading.

    The engine reads only the spec's universe and the symbols the account
    actually holds. Everything else is a bug in the caller, and answering it
    would widen what Tick reads beyond what the user's spec named.
    """


class FixtureDataError(EngineError):
    """A market-data fixture file cannot be read as a series.

    Loud rather than lenient: a fixture that half-parsed would produce numbers
    nobody wrote, which is the fabrication invariant 5 forbids, wearing the
    costume of a convenience.
    """


class MarketDataContractError(EngineError):
    """A market-data port returned something its contract does not allow.

    `bars(symbol, n)` returns exactly `n` bars or `Unavailable`. A port that
    returns fewer has handed us a shorter history than we asked for, and
    computing on it would silently change what an indicator means.
    """
