"""The rule evaluator — the deterministic half of the product.

Given a spec, a market-data port, the account, and the moment, this produces
one `Decision` per rule per symbol and nothing else. It places no orders, it
reaches no network, and it has no opinion of its own: the same spec against the
same bars produces the same decisions, which is what makes the record
reviewable and the runtime supervisable (CLAUDE.md invariant 3).

Three decisions about *meaning* are worth reading before the code.

**Conditions are evaluated on bars; money is converted at the quote.** Every
indicator — `price` included — is computed from the closing series, so a
condition is reproducible from data that does not change under you. The quote
is used for exactly two things: turning a notional or a percentage into whole
shares, and valuing the account. The intent records which quote it used, when
that quote was taken, and from where.

**Unavailability refuses only where it could change the answer.** `all_of`
with one child definitely false is false, even if a sibling could not be
evaluated; `any_of` with one child definitely true is true. Anywhere else, a
missing number produces a `Refusal` rather than a decision — never a zero.

**A firing rule may still refuse.** Long-only is enforced here, not only at the
broker: a sell larger than the position is refused whole, never reduced to a
smaller sell and never allowed to become a short. A size that floors to zero
shares refuses too, because an order for nothing is not what the rule asked
for.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import ROUND_FLOOR, Decimal
from zoneinfo import ZoneInfo

from tick.spec import (
    AllOf,
    AllSize,
    AnyOf,
    Cash,
    ChangePct,
    Compare,
    ComparisonOp,
    Condition,
    DayOfWeek,
    Ema,
    IndicatorNode,
    Not,
    NotionalSize,
    NumberLiteral,
    PctOfEquitySize,
    PositionPctOfEquity,
    PositionQty,
    Price,
    Rule,
    SharesSize,
    Side,
    Sma,
    StrategySpec,
)

from .base import EngineModel, engine_arithmetic, quantize_money, quantize_value
from .cache import TickMarketCache
from .cadence import check_cadence
from .decisions import Decision, EvaluatedValue, OrderIntent, Refusal, RefusalCode
from .errors import EngineError
from .indicators import (
    change_pct,
    crosses_above,
    crosses_below,
    ema,
    latest_close,
    required_bars,
    sma,
)
from .market import MarketDataPort, Quote, QuoteResult, Unavailable
from .portfolio import PortfolioState

__all__ = ["EASTERN", "RuleEvaluator", "TickEvaluation"]

#: Market logic runs on Eastern time; `day_of_week` is the ET weekday.
EASTERN = ZoneInfo("America/New_York")

_PREV_SUFFIX = "@prev"


class TickEvaluation(EngineModel):
    """Everything one tick produced: the decisions, and the numbers behind them.

    The runtime needs the prices and the equity again for the cage, and asking
    the market twice for the same tick is exactly what the cache exists to
    prevent — so they are handed back rather than re-fetched.
    """

    decisions: tuple[Decision, ...]
    prices: Mapping[str, Decimal]
    equity: Decimal | Unavailable
    quote_calls: int
    bars_calls: int


class _SymbolContext:
    """One symbol's view of the tick: its bars, its quote, and the account."""

    def __init__(
        self,
        cache: TickMarketCache,
        symbol: str,
        state: PortfolioState,
        now,
    ) -> None:
        self.cache = cache
        self.symbol = symbol
        self.state = state
        self.now = now

    def closes(self, n: int) -> tuple[Decimal, ...] | Unavailable:
        bars = self.cache.bars(self.symbol, n)
        if isinstance(bars, Unavailable):
            return bars
        return tuple(bar.close for bar in bars)

    def quote(self) -> QuoteResult:
        return self.cache.quote(self.symbol)

    def equity(self) -> Decimal | Unavailable:
        quotes = {symbol: self.cache.quote(symbol) for symbol in sorted(self.state.symbols)}
        return self.state.equity(quotes)


class RuleEvaluator:
    """Evaluates a spec's rules over its universe. Holds no state between ticks."""

    def provenance(self) -> dict[str, object]:
        """Who decided, for the record. For a rule agent: the document did.

        The runner asks whatever produced a tick's decisions for this, and
        writes it into the decision record. A rule agent's answer carries no
        identity beyond its kind, because the spec is already named in the same
        record and the spec IS the decision procedure — reproducible from the
        document and the bars, with nothing else to pin.
        """
        return {"kind": "rule_agent"}

    def evaluate(
        self,
        spec: StrategySpec,
        market: MarketDataPort,
        state: PortfolioState,
        now,
    ) -> list[Decision]:
        """One decision per rule per symbol, in spec order."""
        return list(self.evaluate_tick(spec, market, state, now).decisions)

    def evaluate_tick(
        self,
        spec: StrategySpec,
        market: MarketDataPort,
        state: PortfolioState,
        now,
    ) -> TickEvaluation:
        """`evaluate`, plus the prices and equity the cage will need."""
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now must be timezone-aware; market logic runs on ET sessions")
        check_cadence(spec.cadence)

        permitted = frozenset(spec.universe) | state.symbols
        cache = TickMarketCache(market, permitted)

        decisions: list[Decision] = []
        for symbol in spec.universe:
            context = _SymbolContext(cache, symbol, state, now)
            for rule in spec.rules:
                decisions.append(self._evaluate_rule(rule, context, now))

        # Equity is priced every tick because the cage's drawdown limit needs it.
        # Prices, by contrast, are only what this tick already had to ask for:
        # a symbol no rule needed is not quoted just to fill in a table.
        equity = state.equity({symbol: cache.quote(symbol) for symbol in sorted(state.symbols)})
        prices = {
            symbol: quote.price
            for symbol, quote in sorted(cache.cached_quotes().items())
            if isinstance(quote, Quote)
        }
        return TickEvaluation(
            decisions=tuple(decisions),
            prices=prices,
            equity=equity,
            quote_calls=cache.quote_calls,
            bars_calls=cache.bars_calls,
        )

    # ------------------------------------------------------------------
    # One rule
    # ------------------------------------------------------------------

    def _evaluate_rule(self, rule: Rule, context: _SymbolContext, now) -> Decision:
        values: list[EvaluatedValue] = []
        outcome = _evaluate_condition(rule.when, context, values)
        source = f"rule:{rule.id}"

        if isinstance(outcome, Unavailable):
            refusal = Refusal(
                source=source,
                symbol=context.symbol,
                code=_refusal_code_for(outcome),
                reason=(
                    f"rule {rule.id!r} could not be evaluated for {context.symbol}: "
                    f"{outcome}. No order is placed on a number we do not have."
                ),
            )
            return Decision(
                rule_id=rule.id,
                symbol=context.symbol,
                at=now,
                fired=False,
                values=tuple(values),
                intent=None,
                refusal=refusal,
            )

        if not outcome:
            return Decision(
                rule_id=rule.id,
                symbol=context.symbol,
                at=now,
                fired=False,
                values=tuple(values),
                intent=None,
                refusal=None,
            )

        result = _size_order(rule, context)
        return Decision(
            rule_id=rule.id,
            symbol=context.symbol,
            at=now,
            fired=True,
            values=tuple(values),
            intent=result if isinstance(result, OrderIntent) else None,
            refusal=result if isinstance(result, Refusal) else None,
        )


# ----------------------------------------------------------------------
# Conditions
# ----------------------------------------------------------------------


def _refusal_code_for(unavailable: Unavailable) -> RefusalCode:
    if unavailable.what.startswith("equity"):
        return RefusalCode.EQUITY_UNAVAILABLE
    if unavailable.what.startswith("bars"):
        return RefusalCode.BARS_UNAVAILABLE
    if unavailable.what.startswith("quote"):
        return RefusalCode.QUOTE_UNAVAILABLE
    return RefusalCode.INDICATOR_UNAVAILABLE


def _record(values: list[EvaluatedValue], label: str, result: Decimal | Unavailable) -> None:
    """Keep every operand the condition looked at, once, in evaluation order."""
    if any(existing.label == label for existing in values):
        return
    values.append(EvaluatedValue.of(label, result))


def _evaluate_condition(
    condition: Condition,
    context: _SymbolContext,
    values: list[EvaluatedValue],
) -> bool | Unavailable:
    if isinstance(condition, Compare):
        return _evaluate_compare(condition, context, values)
    if isinstance(condition, Not):
        inner = _evaluate_condition(condition.of, context, values)
        if isinstance(inner, Unavailable):
            return inner
        return not inner
    # all_of / any_of: every child is evaluated so the record is complete, then
    # unavailability is allowed to matter only where it could change the answer.
    results = [_evaluate_condition(child, context, values) for child in condition.of]
    decided = [result for result in results if not isinstance(result, Unavailable)]
    missing = [result for result in results if isinstance(result, Unavailable)]
    if isinstance(condition, AllOf):
        if any(result is False for result in decided):
            return False
        return missing[0] if missing else True
    if isinstance(condition, AnyOf):
        if any(result is True for result in decided):
            return True
        return missing[0] if missing else False
    raise EngineError(f"unknown condition kind {condition!r}")  # pragma: no cover - closed union


def _evaluate_compare(
    condition: Compare,
    context: _SymbolContext,
    values: list[EvaluatedValue],
) -> bool | Unavailable:
    left_now = _operand(condition.left, context, at_previous=False)
    _record(values, condition.left.label(), left_now)
    right_now = _operand(condition.right, context, at_previous=False)
    _record(values, condition.right.label(), right_now)

    if not condition.op.is_cross:
        if isinstance(left_now, Unavailable):
            return left_now
        if isinstance(right_now, Unavailable):
            return right_now
        return _compare(left_now, condition.op, right_now)

    left_prev = _operand(condition.left, context, at_previous=True)
    _record(values, condition.left.label() + _PREV_SUFFIX, left_prev)
    right_prev = _operand(condition.right, context, at_previous=True)
    _record(values, condition.right.label() + _PREV_SUFFIX, right_prev)

    for result in (left_now, right_now, left_prev, right_prev):
        if isinstance(result, Unavailable):
            return result
    if condition.op is ComparisonOp.CROSSES_ABOVE:
        return crosses_above(left_now, left_prev, right_now, right_prev)
    return crosses_below(left_now, left_prev, right_now, right_prev)


def _compare(left: Decimal, op: ComparisonOp, right: Decimal) -> bool:
    if op is ComparisonOp.GT:
        return left > right
    if op is ComparisonOp.LT:
        return left < right
    if op is ComparisonOp.GTE:
        return left >= right
    if op is ComparisonOp.LTE:
        return left <= right
    raise EngineError(f"{op} is not a value comparison")  # pragma: no cover - closed enum


def _operand(
    node: IndicatorNode,
    context: _SymbolContext,
    *,
    at_previous: bool,
) -> Decimal | Unavailable:
    """One operand's value, on this bar or on the one before it."""
    if isinstance(node, NumberLiteral):
        return node.value

    if isinstance(node, Price | Sma | Ema | ChangePct):
        series = context.closes(required_bars(node, previous=at_previous))
        if isinstance(series, Unavailable):
            # The port's answer already names the symbol; scoping it again
            # would read "bars for XYZ for XYZ".
            return series
        if at_previous:
            series = series[:-1]
        value = _series_value(node, series)
        if isinstance(value, Unavailable):
            return value.scoped(context.symbol)
        return value

    if at_previous:  # pragma: no cover - the grammar refuses crosses on these
        raise EngineError(
            f"{node.label()} has no per-bar history; the spec grammar should have refused this"
        )

    if isinstance(node, PositionQty):
        return Decimal(context.state.qty(context.symbol))
    if isinstance(node, Cash):
        return context.state.cash
    if isinstance(node, DayOfWeek):
        return Decimal(context.now.astimezone(EASTERN).isoweekday())
    if isinstance(node, PositionPctOfEquity):
        return _position_pct_of_equity(context)
    raise EngineError(f"unknown operand {node!r}")  # pragma: no cover - closed union


def _series_value(node: IndicatorNode, closes: Sequence[Decimal]) -> Decimal | Unavailable:
    if isinstance(node, Price):
        return latest_close(closes)
    if isinstance(node, Sma):
        return sma(closes, node.n)
    if isinstance(node, Ema):
        return ema(closes, node.n)
    return change_pct(closes, node.n_bars)


def _position_pct_of_equity(context: _SymbolContext) -> Decimal | Unavailable:
    equity = context.equity()
    if isinstance(equity, Unavailable):
        return equity
    if equity == 0:
        return Unavailable(
            what="position_pct_of_equity",
            reason="account equity is 0, so a percentage of it has no value",
        )
    quote = context.quote()
    value = context.state.market_value(context.symbol, quote)
    if isinstance(value, Unavailable):
        return value
    with engine_arithmetic():
        return quantize_value(value / equity * 100)


# ----------------------------------------------------------------------
# Sizing
# ----------------------------------------------------------------------


def _floor_shares(notional: Decimal, price: Decimal) -> int:
    """Whole shares affordable at `price`, always rounded DOWN.

    Rounding down is the only direction that cannot overspend: a rule asking
    for $1,000 of a $184.20 stock gets 5 shares ($921.00), never 6 ($1,105.20).
    """
    with engine_arithmetic():
        return int((notional / price).to_integral_value(rounding=ROUND_FLOOR))


def _size_order(rule: Rule, context: _SymbolContext) -> OrderIntent | Refusal:
    source = f"rule:{rule.id}"
    symbol = context.symbol
    action = rule.then

    quote = context.quote()
    if isinstance(quote, Unavailable):
        return Refusal(
            source=source,
            symbol=symbol,
            code=RefusalCode.QUOTE_UNAVAILABLE,
            reason=(
                f"rule {rule.id!r} fired for {symbol} but {quote}. An order is not "
                f"placed at a price we do not have."
            ),
        )

    price = quote.price
    held = context.state.qty(symbol)
    size = action.size

    if action.side is Side.SELL and held == 0:
        return Refusal(
            source=source,
            symbol=symbol,
            code=RefusalCode.NO_POSITION_TO_SELL,
            reason=(
                f"rule {rule.id!r} fired to sell {size.label()} of {symbol} but the "
                f"account holds none. Tick is long-only: it closes positions, it never "
                f"opens a short."
            ),
        )

    if isinstance(size, SharesSize):
        qty = size.shares
    elif isinstance(size, NotionalSize):
        qty = _floor_shares(size.notional, price)
    elif isinstance(size, PctOfEquitySize):
        equity = context.equity()
        if isinstance(equity, Unavailable):
            return Refusal(
                source=source,
                symbol=symbol,
                code=RefusalCode.EQUITY_UNAVAILABLE,
                reason=(
                    f"rule {rule.id!r} fired to trade {size.label()} of {symbol} but "
                    f"{equity}. A percentage of an unknown number is not a size."
                ),
            )
        with engine_arithmetic():
            qty = _floor_shares(equity * size.pct_of_equity / 100, price)
    elif isinstance(size, AllSize):
        qty = held if action.side is Side.SELL else _floor_shares(context.state.cash, price)
    else:  # pragma: no cover - closed union
        raise EngineError(f"unknown size {size!r}")

    if action.side is Side.SELL and qty > held:
        return Refusal(
            source=source,
            symbol=symbol,
            code=RefusalCode.SELL_EXCEEDS_POSITION,
            reason=(
                f"rule {rule.id!r} fired to sell {qty} {symbol} but the account holds "
                f"{held}. Tick is long-only, so the order is refused whole — it is never "
                f"reduced to a smaller sell nobody asked for, and never allowed to short."
            ),
        )

    if qty < 1:
        basis = "buying power" if action.side is Side.BUY else "the position"
        return Refusal(
            source=source,
            symbol=symbol,
            code=RefusalCode.SIZE_ROUNDS_TO_ZERO,
            reason=(
                f"rule {rule.id!r} fired to trade {size.label()} of {symbol}, which is "
                f"less than one whole share at ${price} ({basis} does not cover one). "
                f"Tick places whole shares only."
            ),
        )

    with engine_arithmetic():
        est_notional = quantize_money(Decimal(qty) * price)

    return OrderIntent(
        source=source,
        symbol=symbol,
        side=action.side,
        qty=qty,
        est_price=price,
        est_notional=est_notional,
        price_asof=quote.asof,
        price_source=quote.source,
        reason=f"rule {rule.id!r} fired: {action.label()} of {symbol}",
    )
