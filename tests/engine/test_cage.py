"""The cage: what an agent may take on, and what it may always still close.

Two properties are tested harder than the arithmetic. First, the cage judges
intents by their *content*, not their author — the same limits apply to a spec
rule and to a model agent, which is what makes invariant 3 an enforcement
rather than an intention. Second, every limit binds buys and none of them binds
an exit: a ceiling that blocks a sell would trap the user in the position the
ceiling existed to prevent.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from tick.engine import CageCode, OrderIntent, SessionState, Unavailable, apply_cage
from tick.spec import Cage, Session, Side

from .conftest import LAST_BAR

CAGE = Cage(
    max_position_pct=Decimal("20.00"),
    max_positions=2,
    max_order_notional=Decimal("1000.00"),
    max_daily_drawdown_pct=Decimal("3.00"),
    allowed_session=Session.REGULAR_HOURS,
)

EQUITY = Decimal("10000.00")
DAY_START = Decimal("10000.00")


def intent(
    symbol: str,
    side: Side,
    qty: int,
    price: str,
    *,
    source: str = "rule:r",
) -> OrderIntent:
    unit = Decimal(price)
    return OrderIntent(
        source=source,
        symbol=symbol,
        side=side,
        qty=qty,
        est_price=unit,
        est_notional=unit * qty,
        price_asof=LAST_BAR,
        price_source="fixture",
        reason="test intent",
    )


def judge(
    *intents: OrderIntent,
    cage: Cage = CAGE,
    held: dict[str, int] | None = None,
    prices: dict[str, Decimal] | None = None,
    equity: Decimal | Unavailable = EQUITY,
    day_start_equity: Decimal = DAY_START,
    session: SessionState = SessionState.NOT_EVALUATED,
):
    return apply_cage(
        cage,
        intents,
        held=held if held is not None else {},
        prices=prices
        if prices is not None
        else {"XYZ": Decimal("100.00"), "ABCD": Decimal("40.00")},
        equity=equity,
        day_start_equity=day_start_equity,
        session=session,
    )


def test_an_intent_inside_every_limit_is_allowed():
    buy = intent("XYZ", Side.BUY, 5, "100.00")
    outcome = judge(buy)
    assert outcome.allowed == (buy,)
    assert outcome.rejected == ()


def test_the_cage_judges_a_model_agents_intent_by_the_same_rule():
    """The cage takes intents from any source; authority does not depend on who asked."""
    from_rule = intent("XYZ", Side.BUY, 50, "100.00", source="rule:greedy")
    from_model = intent("XYZ", Side.BUY, 50, "100.00", source="model:some-model-id")
    rule_outcome = judge(from_rule)
    model_outcome = judge(from_model)
    assert rule_outcome.rejected[0].code == model_outcome.rejected[0].code
    assert model_outcome.allowed == ()
    # and the rejection is traceable back to whoever proposed it
    assert model_outcome.rejected[0].intent.source == "model:some-model-id"


# ----------------------------------------------------------------------
# The session
# ----------------------------------------------------------------------


def test_a_closed_market_stops_both_sides():
    outcome = judge(
        intent("XYZ", Side.BUY, 1, "100.00"),
        intent("ABCD", Side.SELL, 1, "40.00"),
        held={"ABCD": 5},
        session=SessionState.CLOSED,
    )
    assert outcome.allowed == ()
    assert {rejection.code for rejection in outcome.rejected} == {CageCode.SESSION_CLOSED}
    assert "regular_hours" in outcome.rejected[0].reason


def test_an_unevaluated_session_does_not_block_by_itself():
    outcome = judge(intent("XYZ", Side.BUY, 1, "100.00"), session=SessionState.NOT_EVALUATED)
    assert len(outcome.allowed) == 1


# ----------------------------------------------------------------------
# Exits are never blocked
# ----------------------------------------------------------------------


def test_a_sell_passes_an_order_ceiling_it_would_break_as_a_buy():
    big_sell = intent("XYZ", Side.SELL, 90, "100.00")
    assert judge(big_sell, held={"XYZ": 90}).allowed == (big_sell,)
    assert judge(intent("XYZ", Side.BUY, 90, "100.00")).allowed == ()


def test_a_sell_passes_while_a_drawdown_halt_stops_every_buy():
    equity = Decimal("9500.00")  # 5% down against a 3% limit
    outcome = judge(
        intent("XYZ", Side.BUY, 1, "100.00"),
        intent("XYZ", Side.SELL, 1, "100.00"),
        held={"XYZ": 10},
        equity=equity,
    )
    assert [i.side for i in outcome.allowed] == [Side.SELL]
    assert outcome.rejected[0].code is CageCode.DAILY_DRAWDOWN_HALT
    assert "down 5.00000000%" in outcome.rejected[0].reason
    assert "all buying is halted" in outcome.rejected[0].reason


def test_a_sell_passes_when_the_account_cannot_be_priced_at_all():
    missing = Unavailable(what="equity", reason="a held symbol has no price")
    outcome = judge(
        intent("XYZ", Side.BUY, 1, "100.00"),
        intent("XYZ", Side.SELL, 1, "100.00"),
        held={"XYZ": 10},
        equity=missing,
    )
    assert [i.side for i in outcome.allowed] == [Side.SELL]
    assert outcome.rejected[0].code is CageCode.EQUITY_UNAVAILABLE
    assert "Buys stop; exits do not" in outcome.rejected[0].reason


# ----------------------------------------------------------------------
# The limits themselves
# ----------------------------------------------------------------------


def test_an_order_over_the_notional_ceiling_is_rejected():
    outcome = judge(intent("XYZ", Side.BUY, 11, "100.00"))
    rejection = outcome.rejected[0]
    assert rejection.code is CageCode.MAX_ORDER_NOTIONAL
    assert "$1100.00" in rejection.reason
    assert "$1000.00 per-order ceiling" in rejection.reason


def test_a_buy_with_no_price_this_tick_cannot_be_checked_so_it_is_refused():
    outcome = judge(intent("XYZ", Side.BUY, 1, "100.00"), prices={})
    assert outcome.rejected[0].code is CageCode.PRICE_UNAVAILABLE
    assert "not placed" in outcome.rejected[0].reason


def test_a_position_over_the_concentration_cap_is_rejected():
    # 25 shares at $100 is $2,500 of a $10,000 account: 25%, over the 20% cap.
    outcome = judge(intent("XYZ", Side.BUY, 25, "100.00"), cage=_cage(max_order_notional="5000.00"))
    rejection = outcome.rejected[0]
    assert rejection.code is CageCode.MAX_POSITION_PCT
    assert "25.00000000% of equity" in rejection.reason


def test_concentration_counts_what_is_already_held():
    outcome = judge(
        intent("XYZ", Side.BUY, 5, "100.00"),
        held={"XYZ": 18},
        cage=_cage(max_positions=3),
    )
    assert outcome.rejected[0].code is CageCode.MAX_POSITION_PCT


def test_concentration_counts_earlier_intents_in_the_same_batch():
    first = intent("XYZ", Side.BUY, 15, "100.00")
    second = intent("XYZ", Side.BUY, 15, "100.00")
    # 15% of equity each: the first fits under the 20% cap, the pair does not.
    outcome = judge(first, second, cage=_cage(max_order_notional="5000.00"))
    assert outcome.allowed == (first,)
    assert outcome.rejected[0].intent is second
    assert outcome.rejected[0].code is CageCode.MAX_POSITION_PCT


def test_one_position_too_many_is_rejected():
    outcome = judge(
        intent("ABCD", Side.BUY, 1, "40.00"),
        held={"XYZ": 1, "WXY": 1},
        prices={"ABCD": Decimal("40.00")},
    )
    rejection = outcome.rejected[0]
    assert rejection.code is CageCode.MAX_POSITIONS
    assert "allows 2" in rejection.reason


def test_adding_to_a_position_at_the_position_limit_is_allowed():
    buy = intent("XYZ", Side.BUY, 1, "100.00")
    outcome = judge(buy, held={"XYZ": 1, "ABCD": 1})
    assert outcome.allowed == (buy,)


def test_two_new_symbols_in_one_batch_count_against_the_limit():
    first = intent("XYZ", Side.BUY, 1, "100.00")
    second = intent("ABCD", Side.BUY, 1, "40.00")
    outcome = judge(first, second, held={"WXY": 1}, cage=_cage(max_positions=2))
    assert outcome.allowed == (first,)
    assert outcome.rejected[0].code is CageCode.MAX_POSITIONS


# ----------------------------------------------------------------------
# Precedence and inputs
# ----------------------------------------------------------------------


def test_the_session_outranks_every_other_reason():
    outcome = judge(
        intent("XYZ", Side.BUY, 100, "100.00"),
        equity=Decimal("1000.00"),
        session=SessionState.CLOSED,
    )
    assert outcome.rejected[0].code is CageCode.SESSION_CLOSED


def test_a_drawdown_halt_outranks_a_size_ceiling():
    outcome = judge(intent("XYZ", Side.BUY, 100, "100.00"), equity=Decimal("9000.00"))
    assert outcome.rejected[0].code is CageCode.DAILY_DRAWDOWN_HALT


def test_equity_at_exactly_the_drawdown_floor_halts_buying():
    outcome = judge(intent("XYZ", Side.BUY, 1, "100.00"), equity=Decimal("9700.00"))
    assert outcome.rejected[0].code is CageCode.DAILY_DRAWDOWN_HALT


def test_equity_a_cent_above_the_floor_still_buys():
    outcome = judge(intent("XYZ", Side.BUY, 1, "100.00"), equity=Decimal("9700.01"))
    assert len(outcome.allowed) == 1


@pytest.mark.parametrize("day_start", [Decimal("0"), Decimal("-1")])
def test_a_drawdown_cannot_be_computed_from_a_zero_day_start(day_start):
    with pytest.raises(ValueError, match="must be > 0"):
        judge(intent("XYZ", Side.BUY, 1, "100.00"), day_start_equity=day_start)


def test_every_rejection_names_the_order_it_stopped():
    outcome = judge(intent("XYZ", Side.BUY, 100, "100.00"))
    rejection = outcome.rejected[0]
    assert "buy 100 XYZ at $100.00" in str(rejection)


def _cage(**overrides) -> Cage:
    values = {
        "max_position_pct": Decimal("20.00"),
        "max_positions": 2,
        "max_order_notional": Decimal("1000.00"),
        "max_daily_drawdown_pct": Decimal("3.00"),
        "allowed_session": Session.REGULAR_HOURS,
    }
    values.update(
        {
            key: Decimal(value) if isinstance(value, str) else value
            for key, value in overrides.items()
        }
    )
    return Cage(**values)
