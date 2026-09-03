"""The paper broker: the invariants a simulated account must never break.

Paper is where every agent starts (CLAUDE.md invariant 2), so its arithmetic is
the first thing a user ever sees a record of. Cash never goes negative,
quantities never go negative, nothing fills without a price, and the sequence
of order ids is the same on every replay.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from tests.engine.conftest import LAST_BAR, market
from tick.broker import BrokerPort, Fill, PaperBroker, RejectCode, Rejected
from tick.engine import OrderIntent, Quote, Unavailable
from tick.spec import Side


def intent(symbol: str, side: Side, qty: int, price: str = "40.00") -> OrderIntent:
    unit = Decimal(price)
    return OrderIntent(
        source="rule:test",
        symbol=symbol,
        side=side,
        qty=qty,
        est_price=unit,
        est_notional=unit * qty,
        price_asof=LAST_BAR,
        price_source="fixture",
        reason="test intent",
    )


def broker(cash: str = "10000.00", max_cancels: int = 3) -> PaperBroker:
    return PaperBroker(market(), Decimal(cash), max_cancels)


class OneQuoteMarket:
    """A market with a single controllable price, for arithmetic tests."""

    def __init__(self, price: str | None) -> None:
        self.price = price

    def quote(self, symbol: str):
        if self.price is None:
            return Unavailable(what=f"quote for {symbol}", reason="the feed is down")
        return Quote(
            symbol=symbol,
            price=Decimal(self.price),
            asof=LAST_BAR,
            source="stub",
        )

    def bars(self, symbol: str, n: int):  # pragma: no cover - not used here
        return Unavailable(what=f"bars for {symbol}", reason="this stub has no history")


def test_the_paper_broker_satisfies_the_broker_protocol():
    assert isinstance(broker(), BrokerPort)


def test_a_new_account_is_its_starting_cash_and_nothing_else():
    account = broker("2500.00").state()
    assert account.cash == Decimal("2500.00")
    assert account.positions == {}


@pytest.mark.parametrize("cash", [Decimal("0"), Decimal("-1")])
def test_an_account_must_be_funded(cash):
    with pytest.raises(ValueError, match="must be > 0"):
        PaperBroker(market(), cash, 3)


def test_starting_cash_is_never_a_binary_float():
    with pytest.raises(TypeError, match="not exact money"):
        PaperBroker(market(), 10000.55, 3)


def test_max_cancels_is_required():
    with pytest.raises(TypeError):
        PaperBroker(market(), Decimal("100"))  # type: ignore[call-arg]


# ----------------------------------------------------------------------
# Buying
# ----------------------------------------------------------------------


def test_a_buy_fills_at_the_current_quote_not_at_the_intents_estimate():
    paper = PaperBroker(OneQuoteMarket("41.50"), Decimal("1000.00"), 3)
    fill = paper.place(intent("ABCD", Side.BUY, 2, price="40.00"))
    assert isinstance(fill, Fill)
    assert fill.price == Decimal("41.50")
    assert fill.qty == 2
    assert fill.ts == LAST_BAR
    assert paper.state().cash == Decimal("917.00")


def test_a_buy_creates_a_position_at_what_was_paid():
    paper = PaperBroker(OneQuoteMarket("40.00"), Decimal("1000.00"), 3)
    paper.place(intent("ABCD", Side.BUY, 5))
    position = paper.state().positions["ABCD"]
    assert position.qty == 5
    assert position.avg_cost == Decimal("40.000000")


def test_a_second_buy_averages_the_cost_by_what_was_actually_paid():
    stub = OneQuoteMarket("40.00")
    paper = PaperBroker(stub, Decimal("1000.00"), 3)
    paper.place(intent("ABCD", Side.BUY, 5))
    stub.price = "46.00"
    paper.place(intent("ABCD", Side.BUY, 5))
    position = paper.state().positions["ABCD"]
    assert position.qty == 10
    assert position.avg_cost == Decimal("43.000000")
    assert paper.state().cash == Decimal("570.00")


def test_a_buy_beyond_the_cash_on_hand_is_refused_whole():
    paper = PaperBroker(OneQuoteMarket("40.00"), Decimal("100.00"), 3)
    result = paper.place(intent("ABCD", Side.BUY, 5))
    assert isinstance(result, Rejected)
    assert result.code is RejectCode.INSUFFICIENT_CASH
    assert "never overdrawn" in result.reason
    assert paper.state().cash == Decimal("100.00")
    assert paper.state().positions == {}


def test_a_buy_that_spends_the_last_cent_is_allowed():
    paper = PaperBroker(OneQuoteMarket("40.00"), Decimal("80.00"), 3)
    assert isinstance(paper.place(intent("ABCD", Side.BUY, 2)), Fill)
    assert paper.state().cash == Decimal("0.00")


# ----------------------------------------------------------------------
# Selling — long-only
# ----------------------------------------------------------------------


def test_a_sell_reduces_the_position_and_leaves_the_basis_alone():
    stub = OneQuoteMarket("40.00")
    paper = PaperBroker(stub, Decimal("1000.00"), 3)
    paper.place(intent("ABCD", Side.BUY, 5))
    stub.price = "50.00"
    fill = paper.place(intent("ABCD", Side.SELL, 2))
    assert isinstance(fill, Fill)
    assert fill.price == Decimal("50.00")
    position = paper.state().positions["ABCD"]
    assert position.qty == 3
    assert position.avg_cost == Decimal("40.000000")
    assert paper.state().cash == Decimal("900.00")


def test_selling_everything_removes_the_position():
    paper = PaperBroker(OneQuoteMarket("40.00"), Decimal("1000.00"), 3)
    paper.place(intent("ABCD", Side.BUY, 5))
    paper.place(intent("ABCD", Side.SELL, 5))
    assert paper.state().positions == {}
    assert paper.state().cash == Decimal("1000.00")


def test_a_sell_larger_than_the_position_is_refused_never_truncated():
    paper = PaperBroker(OneQuoteMarket("40.00"), Decimal("1000.00"), 3)
    paper.place(intent("ABCD", Side.BUY, 3))
    result = paper.place(intent("ABCD", Side.SELL, 10))
    assert isinstance(result, Rejected)
    assert result.code is RejectCode.SELL_EXCEEDS_POSITION
    assert "never allowed to go short" in result.reason
    assert paper.state().positions["ABCD"].qty == 3


def test_selling_what_is_not_held_never_opens_a_short():
    paper = PaperBroker(OneQuoteMarket("40.00"), Decimal("1000.00"), 3)
    result = paper.place(intent("ABCD", Side.SELL, 1))
    assert isinstance(result, Rejected)
    assert result.code is RejectCode.NO_POSITION_TO_SELL
    assert paper.state().positions == {}


def test_a_long_sequence_never_produces_a_negative_cash_or_quantity():
    stub = OneQuoteMarket("40.00")
    paper = PaperBroker(stub, Decimal("500.00"), 20)
    script = [
        (Side.BUY, 5, "40.00"),
        (Side.SELL, 2, "44.00"),
        (Side.BUY, 100, "44.00"),  # refused: not enough cash
        (Side.SELL, 50, "44.00"),  # refused: not enough shares
        (Side.BUY, 3, "30.00"),
        (Side.SELL, 6, "30.00"),
    ]
    for side, qty, price in script:
        stub.price = price
        paper.place(intent("ABCD", side, qty))
        account = paper.state()
        assert account.cash >= 0
        assert all(position.qty >= 1 for position in account.positions.values())
    assert paper.state().positions == {}


# ----------------------------------------------------------------------
# No price, no fill
# ----------------------------------------------------------------------


def test_an_order_is_never_filled_at_a_price_nobody_quoted():
    paper = PaperBroker(OneQuoteMarket(None), Decimal("1000.00"), 3)
    result = paper.place(intent("ABCD", Side.BUY, 1))
    assert isinstance(result, Rejected)
    assert result.code is RejectCode.QUOTE_UNAVAILABLE
    assert "fabricated number" in result.reason
    assert paper.state().cash == Decimal("1000.00")


# ----------------------------------------------------------------------
# Order ids and cancels
# ----------------------------------------------------------------------


def test_order_ids_are_deterministic():
    paper = PaperBroker(OneQuoteMarket("40.00"), Decimal("1000.00"), 3)
    ids = [
        paper.place(intent("ABCD", Side.BUY, 1)).order_id,
        paper.place(intent("ABCD", Side.BUY, 1)).order_id,
        paper.place(intent("ABCD", Side.SELL, 1)).order_id,
    ]
    assert ids == ["paper-000001", "paper-000002", "paper-000003"]


def test_a_replay_of_the_same_script_produces_the_same_record():
    def run() -> list[tuple[str, str, int, Decimal]]:
        paper = PaperBroker(market(), Decimal("1000.00"), 3)
        paper.place(intent("ABCD", Side.BUY, 5))
        paper.place(intent("ABCD", Side.SELL, 2))
        return [(f.order_id, f.symbol, f.qty, f.price) for f in paper.fills]

    assert run() == run()


def test_a_filled_paper_order_cannot_be_cancelled():
    paper = PaperBroker(OneQuoteMarket("40.00"), Decimal("1000.00"), 3)
    fill = paper.place(intent("ABCD", Side.BUY, 1))
    result = paper.cancel(fill.order_id)
    assert isinstance(result, Rejected)
    assert result.code is RejectCode.NOT_CANCELLABLE


def test_cancelling_an_unknown_order_says_so():
    paper = broker()
    result = paper.cancel("paper-999999")
    assert isinstance(result, Rejected)
    assert result.code is RejectCode.UNKNOWN_ORDER


def test_cancels_are_rate_limited_so_a_loop_cannot_hammer_the_broker():
    paper = broker(max_cancels=2)
    paper.cancel("paper-000001")
    paper.cancel("paper-000002")
    result = paper.cancel("paper-000003")
    assert isinstance(result, Rejected)
    assert result.code is RejectCode.CANCEL_LIMIT_REACHED
    assert "terminate connectivity" in result.reason
    assert paper.cancels == 2


def test_a_fill_describes_itself_in_mechanical_past_tense():
    paper = PaperBroker(OneQuoteMarket("184.20"), Decimal("1000.00"), 3)
    fill = paper.place(intent("XYZ", Side.BUY, 2))
    assert fill.describe() == "bought 2 XYZ at $184.20"


def test_the_state_a_broker_hands_out_cannot_be_edited_behind_its_back():
    paper = PaperBroker(OneQuoteMarket("40.00"), Decimal("1000.00"), 3)
    paper.place(intent("ABCD", Side.BUY, 1))
    snapshot = paper.state()
    paper.place(intent("ABCD", Side.BUY, 1))
    assert snapshot.positions["ABCD"].qty == 1
    assert paper.state().positions["ABCD"].qty == 2


def test_the_broker_reads_nothing_but_the_market_it_was_given():
    """A paper account exists only on this machine: no id, no account lookup, no I/O."""
    paper = broker()
    assert not hasattr(paper, "account_id")
    assert paper.state() == PaperBroker(market(), Decimal("10000.00"), 3).state()


def test_fills_are_timestamped_from_the_quote_that_priced_them():
    stub = OneQuoteMarket("40.00")
    paper = PaperBroker(stub, Decimal("1000.00"), 3)
    fill = paper.place(intent("ABCD", Side.BUY, 1))
    assert fill.ts == LAST_BAR
    assert fill.ts.tzinfo is not None
    assert fill.ts != datetime.now(UTC)
