"""The live-readiness check, on its own: what is missing, and who is asking.

Everything here reads files under `tmp_path`. Nothing opens a connection —
which is the point of checking readiness separately in the first place: a
machine that is not set up never reaches the broker at all.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.broker.test_profile import Session, confirmed, profile, tool
from tick.broker import (
    Category,
    ToolState,
    confirm_profile,
    load_profile,
    profile_path,
    verify_session_profile,
)
from tick.records import write_private_file
from tick.runtime import (
    LIVE_CAPABILITIES,
    LiveReadiness,
    check_local_live_ready,
    local_actor,
    record_first_live_place_proof,
)

AT = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)


def grant(home: Path) -> None:
    write_private_file(
        home / "robinhood" / "tokens.json", json.dumps({"access_token": "placeholder"})
    )


def test_a_bare_machine_is_not_ready_and_names_the_connect_command(tmp_path: Path):
    readiness = check_local_live_ready(tmp_path, approval_mode="each")

    assert not readiness.ready
    assert readiness.account_id is None
    assert any("tick connect robinhood" in step for step in readiness.missing)


def test_a_connected_machine_with_no_profile_names_the_ceremony(tmp_path: Path):
    grant(tmp_path)
    readiness = check_local_live_ready(tmp_path, approval_mode="each")

    assert not readiness.ready
    assert len(readiness.missing) == 1
    assert "no broker profile has been confirmed" in readiness.missing[0]
    assert "tick broker propose --account" in readiness.missing[0]


def test_an_unreadable_profile_is_reported_rather_than_ignored(tmp_path: Path):
    grant(tmp_path)
    write_private_file(profile_path(tmp_path), "{ this is not json")

    readiness = check_local_live_ready(tmp_path, approval_mode="each")

    assert not readiness.ready
    assert "could not be read" in readiness.missing[0]


def test_every_missing_step_is_reported_at_once(tmp_path: Path):
    """A refusal that names one of two missing steps sends someone round twice."""
    readiness = check_local_live_ready(tmp_path, approval_mode="each")
    assert not readiness.ready
    # No grant AND no profile: both checks still report.
    assert len(readiness.missing) == 2


def test_trading_dependency_graph_includes_reconciliation():
    """Every fact place depends on, including order reconciliation."""
    assert [capability.value for capability in LIVE_CAPABILITIES] == [
        "order.place",
        "read.quote",
        "read.positions",
        "read.balances",
        "read.orders",
    ]


def test_a_readiness_cannot_claim_to_be_ready_with_nothing_to_trade_through():
    with pytest.raises(ValueError, match="broker profile"):
        LiveReadiness(ready=True, missing=(), account_id="agentic-0001")


def test_a_readiness_that_is_not_ready_must_say_why():
    with pytest.raises(ValueError, match="must say what is missing"):
        LiveReadiness(ready=False, missing=(), account_id=None)


def test_the_actor_is_read_from_the_environment_and_never_invented():
    assert local_actor({"TICK_ACTOR": "someone"}) == "someone"
    assert local_actor({"USER": "someone-else"}) == "someone-else"
    assert local_actor({"LOGNAME": "third"}) == "third"


def test_an_environment_that_says_nothing_produces_the_word_for_that():
    assert local_actor({}) == "unknown"
    assert local_actor({"USER": "   "}) == "unknown"


def complete_profile():
    discovered = {
        category: tool(
            category.value.replace(".", "_"),
            input_schema=(
                {
                    "type": "object",
                    "properties": {
                        "account": {"type": "string"},
                        "symbol": {"type": "string"},
                        "side": {"type": "string"},
                        "quantity": {"type": "integer"},
                    },
                    "required": ["account", "symbol", "side", "quantity"],
                }
                if category is Category.ORDER_PLACE
                else {"type": "object", "properties": {}}
            ),
        )
        for category in LIVE_CAPABILITIES
    }
    results = {
        Category.ORDER_PLACE: {
            "order_id": "id",
            "quantity": "qty",
            "price": "price",
            "filled_at": "at",
        },
        Category.READ_QUOTE: {"price": "price", "asof": "at"},
        Category.READ_POSITIONS: {
            "items": "rows",
            "account": "account",
            "symbol": "symbol",
            "quantity": "qty",
            "average_cost": "cost",
        },
        Category.READ_BALANCES: {"items": "rows", "account": "account", "cash": "cash"},
        Category.READ_ORDERS: {"items": "rows", "account": "account", "order_id": "id"},
    }
    mappings = []
    for category, discovered_tool in discovered.items():
        arguments = (
            {
                "account": "{account_id}",
                "symbol": "{symbol}",
                "side": "{side}",
                "quantity": "{qty}",
            }
            if category is Category.ORDER_PLACE
            else {}
        )
        mappings.append(
            confirmed(
                discovered_tool,
                category,
                arguments=arguments,
                result=results[category],
                proved=category is not Category.ORDER_PLACE,
            )
        )
    return profile(*mappings), tuple(discovered.values())


def test_first_each_live_launch_needs_confirmed_place_but_not_a_fabricated_probe(tmp_path):
    grant(tmp_path)
    stored, _tools = complete_profile()
    confirm_profile(tmp_path, stored, actor="terminal", at=AT)

    assert check_local_live_ready(tmp_path, approval_mode="each").ready
    standing = check_local_live_ready(tmp_path, approval_mode="standing")
    assert not standing.ready
    assert any(
        "order.place" in sentence and "not proven" in sentence for sentence in standing.missing
    )


def test_terminal_first_live_outcome_records_place_proof_for_the_exact_hashes(tmp_path):
    grant(tmp_path)
    stored, tools = complete_profile()
    confirm_profile(tmp_path, stored, actor="terminal", at=AT)
    session = Session(tools)
    verified = verify_session_profile(
        stored,
        session,
        server=stored.server,
        account_id=stored.account_id,
        confirmation_recorded=True,
    )
    assert all(state is ToolState.CONFIRMED for state in verified.states.values())

    evidence = record_first_live_place_proof(tmp_path, verified, at=AT, outcome="fill")

    assert evidence["proved_by"] == "first_live_fill"
    assert (
        evidence["contract_hash"] == stored.mapping_for(Category.ORDER_PLACE).contract.contract_hash
    )
    assert evidence["mapping_hash"] == stored.mapping_for(Category.ORDER_PLACE).mapping_hash
    assert load_profile(tmp_path).mapping_for(Category.ORDER_PLACE).proved
    assert check_local_live_ready(tmp_path, approval_mode="standing").ready
