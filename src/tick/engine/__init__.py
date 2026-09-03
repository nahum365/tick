"""The deterministic engine: bars in, decisions out, nothing invented.

Given a spec, a market-data port, the account and the moment, the evaluator
produces one decision per rule per symbol — an order intent, a refusal, or a
recorded "did not fire" — and places nothing.

    from tick.engine import RuleEvaluator, SessionState, apply_cage

    evaluation = RuleEvaluator().evaluate_tick(spec, market, state, now)
    outcome = apply_cage(
        spec.cage,
        [d.intent for d in evaluation.decisions if d.intent],
        held={symbol: p.qty for symbol, p in state.positions.items()},
        prices=evaluation.prices,
        equity=evaluation.equity,
        day_start_equity=day_start_equity,
        session=SessionState.NOT_EVALUATED,
    )

Everything in this package is local computation over values the caller hands
in. It opens no socket, builds no client, and reads no file except the market
fixtures it is explicitly pointed at — the user's positions, balances and
prices are computed on the user's own machine and go nowhere else.

CLAUDE.md invariants this package carries:

- **5, no number is fabricated.** A missing price is an `Unavailable`, a typed
  value with words in it. Nothing defaults to zero, and equity refuses whole
  rather than counting an unpriced holding as nothing.
- **5, again, in the indicators.** An average of "the last 20 closes" computed
  from 12 closes is a different number wearing the same name, so too little
  history is `Unavailable` rather than a shortened window.
- **3, the spec decides.** `RuleEvaluator` executes the spec's grammar and
  nothing else: same spec, same bars, same decisions. `apply_cage` bounds
  intents from ANY source, so a model agent later meets the same limits
  through code it cannot call.
- **Long-only.** A sell larger than the position is refused whole, never
  reduced to a smaller sell and never turned into a short.
- **8, fail safe.** A data failure is answered once and not retried inside a
  tick, so a broken feed stops the runtime rather than hammering the socket;
  and the cage bounds what an agent may take ON while never blocking an exit.
"""

from __future__ import annotations

from .base import (
    CENTS,
    ENGINE_CONTEXT,
    QUANTUM,
    EngineModel,
    ExactDecimal,
    engine_arithmetic,
    quantize_money,
    quantize_value,
)
from .cache import TickMarketCache
from .cadence import MIN_CADENCE_MINUTES, check_cadence, describe_cadence
from .cage import CageCode, CageOutcome, CageRejection, SessionState, apply_cage
from .decisions import Decision, EvaluatedValue, OrderIntent, Refusal, RefusalCode
from .errors import (
    CadenceRefused,
    EngineError,
    FixtureDataError,
    MarketDataContractError,
    SymbolOutsideScope,
)
from .evaluate import EASTERN, RuleEvaluator, TickEvaluation
from .fixture_market import FixtureMarketData
from .indicators import (
    EMA_WINDOW_MULTIPLE,
    change_pct,
    crosses_above,
    crosses_below,
    ema,
    latest_close,
    required_bars,
    sma,
)
from .market import Bar, BarsResult, MarketDataPort, Quote, QuoteResult, Unavailable
from .portfolio import PortfolioState, Position

__all__ = [
    "CENTS",
    "EASTERN",
    "EMA_WINDOW_MULTIPLE",
    "ENGINE_CONTEXT",
    "MIN_CADENCE_MINUTES",
    "QUANTUM",
    "Bar",
    "BarsResult",
    "CadenceRefused",
    "CageCode",
    "CageOutcome",
    "CageRejection",
    "Decision",
    "EngineError",
    "EngineModel",
    "EvaluatedValue",
    "ExactDecimal",
    "FixtureDataError",
    "FixtureMarketData",
    "MarketDataContractError",
    "MarketDataPort",
    "OrderIntent",
    "PortfolioState",
    "Position",
    "Quote",
    "QuoteResult",
    "Refusal",
    "RefusalCode",
    "RuleEvaluator",
    "SessionState",
    "SymbolOutsideScope",
    "TickEvaluation",
    "TickMarketCache",
    "Unavailable",
    "apply_cage",
    "change_pct",
    "crosses_above",
    "check_cadence",
    "crosses_below",
    "describe_cadence",
    "ema",
    "engine_arithmetic",
    "latest_close",
    "quantize_money",
    "quantize_value",
    "required_bars",
    "sma",
]
