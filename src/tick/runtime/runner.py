"""One tick, end to end — and the loop that repeats it.

    outcome = Runner(
        stamp=utc_clock, market_source=..., broker_source=..., evaluator=RuleEvaluator()
    ).tick(
        agent, market=market, broker=broker, clock=clock,
        notify=print, approve=ask, now=datetime.now(UTC),
    )

The order of a tick is fixed, and every step of it is a refusal point:

1. **The ledger must verify.** Before anything is evaluated and before anything
   is appended. A runtime that cannot record must not trade, so a chain that
   does not verify places nothing, records nothing, and raises
   `LedgerQuarantined` naming the failing seq and the one command that moves
   things forward (`tick ledger new <agent_id>`). No `--force`, no
   skip-the-bad-line, no automatic roll.
2. **The kill switch wins.** If `STOP` is present the tick records a `stop`,
   notifies, and places NOTHING. It is checked on every tick including the one
   after a failure, which is what makes "the user can still stop it" true
   rather than hopeful (invariant 8).
3. **The mode and the broker must agree.** A `live` agent must be holding a
   broker whose data source is `robinhood`, and a `paper` one the local
   simulation. Paper is the default and live is an explicit, logged act
   (invariant 2), so a `--live` that quietly ran paper — or a paper run that
   reached a real brokerage — would make the record a lie about what happened
   to somebody's money. Either mismatch records a `note` and refuses.
4. **The session must be open.** Outside regular hours the tick reads NO market
   data and writes no record. The cheapest way to stay clear of Robinhood's
   undefined "excessive market data usage" is not to ask.
5. **Evaluate, then cage.** `evaluator` produces the tick's decisions — one per
   rule per symbol for a rule agent, one per proposal for a caged model agent —
   and the cage bounds what may be taken on. The cage is slice 02's code,
   unchanged, and it is the SAME code either way: this module decides nothing
   about a trade and does not know which kind of agent proposed one.
6. **Approve, place, record, notify** — in that order, per intent.

**A broker failure stops the run and does not retry.** Any exception from
`place` — or from the `state()` read that opens the tick, or from the model a
caged agent asks — records a `note` carrying the error, notifies, and ends the
tick with `halted=True`; the intents behind it are not attempted and the failed
one is not re-sent. The run is left stoppable: the ledger is intact, `STOP` still
works, and the next tick checks it first. Blind-retrying an order whose fate
is unknown is how one order becomes two.

**Records, and how many.** One `decision` record per tick carries the whole
evaluation — every rule, every symbol, the numbers each was judged on, the
prices and the equity. That answers the volume question slice 03 left open: 20
symbols × 5 rules is one record, not a hundred. Anything that stopped an order
a rule actually asked for gets a record of its OWN as well — a fired rule that
refused, a cage rejection, a declined approval — because those are the rows a
person goes looking for.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from pydantic import AwareDatetime, BaseModel, ConfigDict

from tick.agents import ModelAgentError, model_id_of
from tick.broker import BrokerPort, CancelResult, Fill, ProfileBroker, RejectCode, Rejected
from tick.engine import (
    Decision,
    MarketDataPort,
    OrderIntent,
    PortfolioState,
    SessionState,
    TickEvaluation,
    Unavailable,
    apply_cage,
)
from tick.records import DataSource, Ledger, RecordKind, normalize_payload
from tick.spec import canonical_encode, sha256_hex

from . import notify as grammar
from .approvals import ApprovalOutcome, ApprovalQueue, boot_id
from .clock import MarketClock
from .errors import ModeNotWired, NotificationRefused
from .launch import load_run_lease
from .live import record_first_live_place_proof
from .modes import ApprovalMode, Mode
from .reconcile import reconcile_unknown_orders
from .schedule import Scheduler
from .state import AgentRun

__all__ = ["ApprovalDecision", "IntentSource", "Runner", "TickOutcome", "rule_id_of"]

#: The prefix `tick.engine` puts on a spec rule's intents.
_RULE_PREFIX = "rule:"

#: Which broker each mode's orders must be going to. Paper orders go to the
#: local simulation and live orders go to the brokerage; there is no third
#: pairing, and a mismatch refuses rather than being reconciled.
_BROKER_FOR_MODE: dict[Mode, DataSource] = {
    Mode.PAPER: DataSource.PAPER,
    Mode.LIVE: DataSource.ROBINHOOD,
}


@dataclass(frozen=True, slots=True)
class ApprovalDecision:
    """A terminal approval result; its source is evidence, not a person identity."""

    approved: bool
    decided_via: str | None
    outcome: str
    note: str
    approval_id: str | None
    run_id: str | None
    boot_id: str | None
    intent_hash: str | None
    evidence_hash: str | None

    def __post_init__(self) -> None:
        if self.decided_via not in {None, "terminal", "api", "chat"}:
            raise ValueError("decided_via must be terminal, api, chat, or None for expiry/STOP")
        if not self.note.strip():
            raise ValueError("an approval decision must say what happened")
        if not self.outcome.strip():
            raise ValueError("an approval decision must name its terminal outcome")


def rule_id_of(source: str) -> str:
    """The rule id inside an intent's `source` (`rule:golden-cross`)."""
    return source[len(_RULE_PREFIX) :] if source.startswith(_RULE_PREFIX) else source


@runtime_checkable
class IntentSource(Protocol):
    """Whatever produces a tick's decisions: a spec's rules, or a caged model.

    Two methods, and the shape is `RuleEvaluator`'s because that came first.
    Everything after this seam — the cage, approval, placement, the record, the
    notification grammar — is one code path, which is what makes "a model agent
    meets the same limits" a fact about the code rather than a claim about two
    of them.

    `provenance()` is what the decision record says about WHO decided. For a
    rule agent that is its kind and nothing else, because the spec is already
    named in the same record and the spec IS the procedure. For a model agent
    it is the model id, the model the provider reported, and the hashes of the
    instructions and of the exact prompt — the things that make a judgment
    nobody can re-derive reviewable anyway.
    """

    def evaluate_tick(
        self,
        spec: Any,
        market: MarketDataPort,
        state: PortfolioState,
        now: datetime,
    ) -> TickEvaluation:  # pragma: no cover - protocol
        ...

    def provenance(self) -> dict[str, Any]:  # pragma: no cover - protocol
        ...


class TickOutcome(BaseModel):
    """What one tick did. A value, so a caller never has to re-read the ledger."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_id: str
    at: AwareDatetime
    mode: Mode
    #: True when the rules were evaluated. False when the market was shut or
    #: the kill switch was set — in both cases nothing was read and nothing
    #: was placed.
    evaluated: bool
    session_open: bool
    #: The kill switch was set; a `stop` was recorded and nothing was placed.
    stopped: bool
    #: A failure ended the run. The loop stops; the agent stays stoppable.
    halted: bool
    halt_reason: str | None
    fills: tuple[Fill, ...]
    #: One line per order a rule asked for that was not placed.
    not_placed: tuple[str, ...]
    notifications: tuple[str, ...]
    records: int

    @property
    def placed_anything(self) -> bool:
        return bool(self.fills)


class Runner:
    """Runs an agent's ticks. Holds no per-agent state between them.

    `stamp` is the wall clock every record is timestamped with; `market_source`
    and `broker_source` are where the numbers in a record came from; `evaluator`
    is what produces the tick's decisions. All four are required and none has a
    default: a record whose provenance was inherited from a library default is a
    record nobody can trust to say where its numbers are from, and a runner that
    built its own evaluator would decide for the caller which kind of agent this
    is.
    """

    def __init__(
        self,
        *,
        stamp: Callable[[], datetime],
        market_source: DataSource,
        broker_source: DataSource,
        evaluator: IntentSource,
    ) -> None:
        self._stamp = stamp
        self._market_source = DataSource(market_source)
        self._broker_source = DataSource(broker_source)
        self._evaluator = evaluator

    # ------------------------------------------------------------------
    # One tick
    # ------------------------------------------------------------------

    def tick(
        self,
        agent: AgentRun,
        *,
        market: MarketDataPort,
        broker: BrokerPort,
        clock: MarketClock,
        notify: Callable[[str], None],
        approve: Callable[[OrderIntent], bool | ApprovalDecision],
        now: datetime,
    ) -> TickOutcome:
        """Evaluate `agent` once at `now` and act on what comes out.

        `now` is passed in rather than read from a clock in here, so a tick is
        reproducible: the same agent, the same fixtures and the same moment
        produce the same decisions and the same record.
        """
        # 1. The record comes first. Nothing is placed and nothing is appended
        #    on a chain that does not verify.
        agent.require_verified_ledger()
        ledger = agent.ledger(clock=self._stamp)
        state = agent.state
        lease = load_run_lease(agent.home, agent.agent_id)
        mode = (
            lease.mode
            if lease is not None and lease.pid == os.getpid() and lease.boot_id == boot_id()
            else state.mode
        )
        appended = _Counter()

        # 2. The kill switch, before any market read and any order.
        if agent.stop_requested():
            reason = agent.stop_reason()
            self._append(
                ledger,
                appended,
                RecordKind.STOP,
                {"event": "kill_switch", "reason": reason, "at": now},
                source=DataSource.RUNTIME,
            )
            sent = self._send(
                notify,
                ledger,
                appended,
                mode=mode,
                compose=lambda: grammar.run_stopped(
                    agent_id=agent.agent_id, reason=reason, mode=mode
                ),
                fallback=lambda: grammar.run_stopped(
                    agent_id=agent.agent_id,
                    reason="the record carries what happened",
                    mode=mode,
                ),
            )
            return self._outcome(
                agent,
                now,
                mode,
                evaluated=False,
                session_open=False,
                stopped=True,
                halted=False,
                halt_reason=reason,
                notifications=(sent,),
                records=appended.count,
            )

        # 3. The mode and the broker have to agree about what this run is.
        expected = _BROKER_FOR_MODE[mode]
        if self._broker_source is not expected:
            message = (
                f"agent {agent.agent_id} is in {mode.value} mode, whose orders go to a "
                f"{expected.value} broker, but this run was built with a "
                f"{self._broker_source.value} one. Nothing was placed: a record that "
                f"labelled a simulated fill live — or a real one simulated — would be "
                f"worse than no record."
            )
            self._append(
                ledger,
                appended,
                RecordKind.NOTE,
                {
                    "event": "mode_broker_mismatch",
                    "mode": mode.value,
                    "broker_source": self._broker_source.value,
                    "expected_broker_source": expected.value,
                    "reason": message,
                    "at": now,
                },
                source=DataSource.RUNTIME,
            )
            raise ModeNotWired(message)

        if mode is Mode.LIVE:
            try:
                appended.count += reconcile_unknown_orders(agent, broker, now=now)
            except Exception as exc:  # noqa: BLE001 - reconciliation always fails closed
                return self._halt(
                    agent,
                    ledger,
                    appended,
                    notify,
                    now=now,
                    mode=mode,
                    event="order_reconciliation_failed",
                    detail={"action": "read.orders"},
                    reason=str(exc),
                )

        # 4. The session. Outside it, nothing is read and nothing is recorded.
        if not clock.is_open(now):
            return self._outcome(
                agent,
                now,
                mode,
                evaluated=False,
                session_open=False,
                stopped=False,
                halted=False,
                halt_reason=None,
                notifications=(),
                records=appended.count,
            )

        # 5. Evaluate. Quotes are fetched only for the spec's universe and the
        #    symbols the account holds, and each is asked for once (slice 02's
        #    per-tick cache).
        #
        #    Both reads here can fail against a real brokerage or a real model,
        #    and both stop the run rather than being asked again: an account
        #    Tick could not read is not an empty account, and a model that did
        #    not answer has not said "do nothing".
        spec = agent.spec
        try:
            portfolio = broker.state()
        except Exception as exc:  # noqa: BLE001 - any broker failure stops the run
            return self._halt(
                agent,
                ledger,
                appended,
                notify,
                now=now,
                mode=mode,
                event="broker_failed",
                detail={"action": "read_state"},
                reason=(
                    f"the account could not be read: {type(exc).__name__}: {exc}. Nothing "
                    f"was evaluated and nothing was placed; the read was not retried."
                ),
            )
        try:
            evaluation = self._evaluator.evaluate_tick(spec, market, portfolio, now)
        except ModelAgentError as exc:
            return self._halt(
                agent,
                ledger,
                appended,
                notify,
                now=now,
                mode=mode,
                event="model_failed",
                detail={"action": "decide"},
                reason=(
                    f"the model agent could not decide: {type(exc).__name__}: {exc}. "
                    f"Nothing was placed and the model was not asked again."
                ),
            )

        session_date = clock.session_date(now)

        # 6. The whole evaluation, as one record — written before the session
        #    gate below, so a tick that stops for want of an opening equity
        #    still records what it saw.
        self._append(
            ledger,
            appended,
            RecordKind.DECISION,
            _decision_payload(
                agent,
                evaluation,
                now,
                session_date,
                self._evaluator.provenance(),
                broker,
            ),
            source=self._market_source,
        )

        # 7. The session's opening equity. Never fabricated: if the account
        #    cannot be priced, the session does not open and nothing is placed.
        if state.session_date != session_date:
            if isinstance(evaluation.equity, Unavailable):
                reason = (
                    f"the session could not be opened: {evaluation.equity}. The daily "
                    f"drawdown limit is a percentage of the opening equity, and Tick "
                    f"does not invent one. Nothing was placed."
                )
                self._append(
                    ledger,
                    appended,
                    RecordKind.NOTE,
                    {"event": "session_not_opened", "reason": reason, "at": now},
                    source=self._market_source,
                )
                sent = self._send(
                    notify,
                    ledger,
                    appended,
                    mode=mode,
                    compose=lambda: grammar.run_stopped(
                        agent_id=agent.agent_id, reason=reason, mode=mode
                    ),
                    fallback=lambda: grammar.run_stopped(
                        agent_id=agent.agent_id,
                        reason="the record carries what happened",
                        mode=mode,
                    ),
                )
                return self._outcome(
                    agent,
                    now,
                    mode,
                    evaluated=True,
                    session_open=True,
                    stopped=False,
                    halted=True,
                    halt_reason=reason,
                    not_placed=(reason,),
                    notifications=(sent,),
                    records=appended.count,
                )
            state = agent.save_state(
                state.opened(session_date=session_date, equity=evaluation.equity)
            )

        day_start_equity = state.day_start_equity
        assert day_start_equity is not None  # set together with session_date

        notifications: list[str] = []
        not_placed: list[str] = []

        # 8. A fired rule that could not be sized is an order the user asked
        #    for and did not get; it gets its own record and its own sentence.
        for decision in evaluation.decisions:
            if decision.fired and decision.refusal is not None:
                self._append(
                    ledger,
                    appended,
                    RecordKind.REFUSAL,
                    {"stage": "engine", "refusal": decision.refusal, "at": now},
                    source=self._market_source,
                )
                not_placed.append(decision.refusal.reason)
                notifications.append(
                    self._not_placed(
                        notify,
                        ledger,
                        appended,
                        mode=mode,
                        rule_id=decision.rule_id,
                        reason=decision.refusal.reason,
                    )
                )

        # 9. The cage. Slice 02's code, over intents from any source.
        outcome = apply_cage(
            spec.cage,
            [decision.intent for decision in evaluation.decisions if decision.intent is not None],
            held={symbol: position.qty for symbol, position in portfolio.positions.items()},
            prices=evaluation.prices,
            equity=evaluation.equity,
            day_start_equity=day_start_equity,
            session=SessionState.OPEN,
        )
        for rejection in outcome.rejected:
            self._append(
                ledger,
                appended,
                RecordKind.REFUSAL,
                {"stage": "cage", "rejection": rejection, "at": now},
                source=self._market_source,
            )
            not_placed.append(rejection.reason)
            notifications.append(
                self._not_placed(
                    notify,
                    ledger,
                    appended,
                    mode=mode,
                    rule_id=rule_id_of(rejection.intent.source),
                    reason=rejection.reason,
                )
            )

        # 10. Approve and place.
        fills: list[Fill] = []
        halted = False
        halt_reason: str | None = None
        for intent in outcome.allowed:
            rule = rule_id_of(intent.source)
            if state.approval is ApprovalMode.EACH:
                raw_answer = approve(intent)
                approval_result = (
                    raw_answer
                    if isinstance(raw_answer, ApprovalDecision)
                    else ApprovalDecision(
                        approved=bool(raw_answer),
                        decided_via="terminal",
                        outcome="approved" if raw_answer else "declined",
                        note=(
                            "approved from the terminal"
                            if raw_answer
                            else "you declined from the terminal"
                        ),
                        approval_id=None,
                        run_id=None,
                        boot_id=None,
                        intent_hash=None,
                        evidence_hash=None,
                    )
                )
                answered = approval_result.approved
                self._append(
                    ledger,
                    appended,
                    RecordKind.ORDER,
                    {
                        "intent": intent,
                        "approval": ApprovalMode.EACH.value,
                        "approved": answered,
                        "decided_via": approval_result.decided_via,
                        "note": approval_result.outcome,
                        "reason": approval_result.note,
                        "at": now,
                    },
                    source=self._broker_source,
                )
                if not answered:
                    reason = (
                        f"{approval_result.note}: {intent.describe()} was not approved, so nothing "
                        f"was placed for rule {rule!r}."
                    )
                    self._append(
                        ledger,
                        appended,
                        RecordKind.REFUSAL,
                        {"stage": "approval", "intent": intent, "reason": reason, "at": now},
                        source=DataSource.RUNTIME,
                    )
                    not_placed.append(reason)
                    notifications.append(
                        self._not_placed(
                            notify, ledger, appended, mode=mode, rule_id=rule, reason=reason
                        )
                    )
                    continue
            else:
                self._append(
                    ledger,
                    appended,
                    RecordKind.ORDER,
                    {
                        "intent": intent,
                        "approval": ApprovalMode.STANDING.value,
                        "approved": True,
                        "at": now,
                    },
                    source=self._broker_source,
                )

            try:
                invalid = (
                    self._approval_recheck(agent, intent, approval_result)
                    if state.approval is ApprovalMode.EACH
                    else None
                )
                if invalid is not None:
                    self._append(
                        ledger,
                        appended,
                        RecordKind.REFUSAL,
                        {
                            "stage": "dispatch",
                            "code": "approval_recheck_failed",
                            "intent": intent,
                            "reason": invalid,
                            "at": now,
                        },
                        source=DataSource.RUNTIME,
                    )
                    not_placed.append(invalid)
                    halted = True
                    halt_reason = invalid
                    break
                # An approval is not a reservation. The record and STOP are
                # checked again at the final local boundary before dispatch.
                with agent.dispatch_gate() as dispatch_allowed:
                    if not dispatch_allowed:
                        reason = (
                            f"the kill switch was set before {intent.describe()} entered "
                            f"the broker. Nothing was placed for rule {rule!r}; remove STOP "
                            "only when a later tick should evaluate again."
                        )
                        self._append(
                            ledger,
                            appended,
                            RecordKind.REFUSAL,
                            {
                                "stage": "dispatch",
                                "code": "stopped_before_dispatch",
                                "intent": intent,
                                "reason": reason,
                                "at": now,
                            },
                            source=DataSource.RUNTIME,
                        )
                        not_placed.append(reason)
                        halted = True
                        halt_reason = reason
                        break
                    agent.require_verified_ledger()
                    result = broker.place(intent)
            except Exception as exc:  # noqa: BLE001 - any broker failure stops the run
                halt_reason = (
                    f"the broker failed while placing {intent.describe()} for rule "
                    f"{rule!r}: {type(exc).__name__}: {exc}. The order was not retried "
                    f"and no further order was placed this tick."
                )
                self._append(
                    ledger,
                    appended,
                    RecordKind.NOTE,
                    {
                        "event": "broker_failed",
                        "intent": intent,
                        "error": f"{type(exc).__name__}: {exc}",
                        "reason": halt_reason,
                        "at": now,
                    },
                    source=DataSource.RUNTIME,
                )
                not_placed.append(halt_reason)
                notifications.append(
                    self._send(
                        notify,
                        ledger,
                        appended,
                        mode=mode,
                        compose=lambda reason=halt_reason: grammar.run_stopped(
                            agent_id=agent.agent_id, reason=reason, mode=mode
                        ),
                        fallback=lambda: grammar.run_stopped(
                            agent_id=agent.agent_id,
                            reason="the record carries what happened",
                            mode=mode,
                        ),
                    )
                )
                halted = True
                break

            if isinstance(result, Fill):
                self._append(
                    ledger,
                    appended,
                    RecordKind.FILL,
                    {"fill": result, "intent": intent, "at": now},
                    source=self._broker_source,
                )
                fills.append(result)
                notifications.append(
                    self._filled(notify, ledger, appended, mode=mode, rule_id=rule, fill=result)
                )
            else:
                self._append(
                    ledger,
                    appended,
                    RecordKind.REJECTED,
                    {"rejected": result, "intent": intent, "at": now},
                    source=self._broker_source,
                )
                not_placed.append(result.reason)
                notifications.append(
                    self._not_placed(
                        notify, ledger, appended, mode=mode, rule_id=rule, reason=result.reason
                    )
                )

            proof_error = self._first_live_place_proof(
                agent,
                ledger,
                appended,
                broker,
                result,
                mode=mode,
                approval=state.approval,
                now=now,
            )
            if proof_error is not None:
                not_placed.append(proof_error)
                halted = True
                halt_reason = proof_error
                break

        agent.save_state(state.ticked(now))
        return self._outcome(
            agent,
            now,
            mode,
            evaluated=True,
            session_open=True,
            stopped=False,
            halted=halted,
            halt_reason=halt_reason,
            fills=tuple(fills),
            not_placed=tuple(not_placed),
            notifications=tuple(notifications),
            records=appended.count,
        )

    def _first_live_place_proof(
        self,
        agent: AgentRun,
        ledger: Ledger,
        appended: _Counter,
        broker: BrokerPort,
        result,
        *,
        mode: Mode,
        approval: ApprovalMode,
        now: datetime,
    ) -> str | None:
        """Persist first-live evidence only after an exact call returns terminally."""
        if (
            mode is not Mode.LIVE
            or approval is not ApprovalMode.EACH
            or not isinstance(broker, ProfileBroker)
            or not broker.last_place_called
            or (isinstance(result, Rejected) and result.code is RejectCode.ORDER_OUTCOME_UNKNOWN)
        ):
            return None
        outcome = "fill" if isinstance(result, Fill) else "rejection"
        try:
            evidence = record_first_live_place_proof(
                agent.home,
                broker.verified_profile,
                at=now,
                outcome=outcome,
            )
            if evidence is not None:
                self._append(
                    ledger,
                    appended,
                    RecordKind.NOTE,
                    {"event": "first_live_place_proven", **evidence, "at": now},
                    source=DataSource.RUNTIME,
                )
        except Exception as exc:  # noqa: BLE001 - failure must stop later orders
            reason = (
                f"the first live place outcome was recorded, but its exact proof could not "
                f"be persisted ({type(exc).__name__}: {exc}). No further order was placed; "
                "inspect the ledger and broker profile before continuing."
            )
            self._append(
                ledger,
                appended,
                RecordKind.NOTE,
                {"event": "first_live_place_proof_failed", "reason": reason, "at": now},
                source=DataSource.RUNTIME,
            )
            return reason
        return None

    def _approval_recheck(
        self, agent: AgentRun, intent: OrderIntent, decision: ApprovalDecision
    ) -> str | None:
        """Re-bind queued approval to this run and unchanged intent before dispatch."""
        if decision.approval_id is None:
            return None
        if None in {
            decision.run_id,
            decision.boot_id,
            decision.intent_hash,
            decision.evidence_hash,
        }:
            return (
                "the approval lost its run or evidence binding. Nothing was placed; "
                "wait for a newly evaluated intent."
            )
        try:
            request, resolution = ApprovalQueue.system(agent.home, agent.agent_id).get(
                decision.approval_id
            )
        except Exception as exc:  # noqa: BLE001 - corrupt authority always fails closed
            return (
                f"the approval cannot be re-read ({exc}). Nothing was placed; wait for "
                "a newly evaluated intent."
            )
        lease = load_run_lease(agent.home, agent.agent_id)
        if (
            resolution is None
            or resolution.outcome is not ApprovalOutcome.APPROVED
            or request.run_id != decision.run_id
            or request.boot_id != decision.boot_id
            or boot_id() != decision.boot_id
            or (
                lease is not None
                and (lease.run_id, lease.boot_id) != (decision.run_id, decision.boot_id)
            )
        ):
            return (
                "the approval no longer belongs to this active run and boot. Nothing was "
                "placed; wait for the current run to evaluate again."
            )
        intent_hash = sha256_hex(
            canonical_encode(normalize_payload(intent.model_dump(mode="json")))
        )
        evidence = normalize_payload(
            {
                "price_asof": intent.price_asof,
                "price_source": intent.price_source,
                "est_price": intent.est_price,
                "est_notional": intent.est_notional,
            }
        )
        evidence_hash = sha256_hex(canonical_encode(evidence))
        if (
            intent_hash != decision.intent_hash
            or evidence_hash != decision.evidence_hash
            or request.intent_hash != decision.intent_hash
            or request.evidence_hash != decision.evidence_hash
        ):
            return (
                "the approved intent or its evidence changed before dispatch. Nothing was "
                "placed; wait for a newly evaluated intent."
            )
        return None

    # ------------------------------------------------------------------
    # The loop
    # ------------------------------------------------------------------

    def run(
        self,
        agent: AgentRun,
        *,
        market: MarketDataPort,
        broker: BrokerPort,
        clock: MarketClock,
        scheduler: Scheduler,
        notify: Callable[[str], None],
        approve: Callable[[OrderIntent], bool | ApprovalDecision],
        now: Callable[[], datetime],
        sleep: Callable[[float], None],
        poll_seconds: float,
        max_ticks: int | None,
    ) -> list[TickOutcome]:
        """Tick on the schedule until the agent is stopped or something halts it.

        `sleep` never waits longer than `poll_seconds` at a stretch, whatever
        the schedule says. That is the kill switch's liveness: an agent whose
        next tick is tomorrow morning still notices a `STOP` within one poll,
        rather than at the bell.

        `now` and `sleep` are both injected and both required, so the loop can
        be tested without time passing.
        """
        if poll_seconds <= 0:
            raise ValueError(f"poll_seconds ({poll_seconds}) must be > 0")
        outcomes: list[TickOutcome] = []
        cadence = agent.spec.cadence
        due = scheduler.next_tick(cadence, now())
        while max_ticks is None or len(outcomes) < max_ticks:
            moment = now()
            if agent.stop_requested():
                outcomes.append(
                    self.tick(
                        agent,
                        market=market,
                        broker=broker,
                        clock=clock,
                        notify=notify,
                        approve=approve,
                        now=moment,
                    )
                )
                break
            if moment < due:
                sleep(min((due - moment).total_seconds(), poll_seconds))
                continue
            outcome = self.tick(
                agent,
                market=market,
                broker=broker,
                clock=clock,
                notify=notify,
                approve=approve,
                now=moment,
            )
            outcomes.append(outcome)
            if outcome.halted:
                break
            due = scheduler.next_tick(cadence, moment)
        return outcomes

    # ------------------------------------------------------------------
    # Cancels
    # ------------------------------------------------------------------

    def cancel(self, agent: AgentRun, *, broker: BrokerPort, order_id: str) -> CancelResult:
        """Cancel a working order, unless this session has cancelled enough.

        The guard is the runtime's, not the broker's: `PaperBroker` has one of
        its own, but a cancel loop is a pattern brokers terminate connectivity
        over and the limit has to hold whichever broker is behind the port.
        Beyond the limit the broker is not called at all.
        """
        agent.require_verified_ledger()
        ledger = agent.ledger(clock=self._stamp)
        state = agent.state
        remaining = agent.cancels_remaining(state)
        if remaining <= 0:
            rejection = Rejected(
                code=RejectCode.CANCEL_LIMIT_REACHED,
                reason=(
                    f"agent {agent.agent_id} has already cancelled "
                    f"{state.cancels_this_session} times this session, the configured "
                    f"limit of {state.max_cancels_per_session}. The broker was not "
                    f"asked: repeated cancellation is a pattern brokers terminate "
                    f"connectivity over."
                ),
            )
            ledger.append(
                RecordKind.REFUSAL,
                {
                    "stage": "cancel_guard",
                    "order_id": order_id,
                    "reason": rejection.reason,
                    "cancels_this_session": state.cancels_this_session,
                    "max_cancels_per_session": state.max_cancels_per_session,
                },
                source=DataSource.RUNTIME,
            )
            return rejection

        agent.save_state(state.cancelled())
        try:
            result = broker.cancel(order_id)
        except Exception as exc:  # noqa: BLE001 - any broker failure stops the run
            ledger.append(
                RecordKind.NOTE,
                {
                    "event": "broker_failed",
                    "action": "cancel",
                    "order_id": order_id,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                source=DataSource.RUNTIME,
            )
            raise
        ledger.append(
            RecordKind.NOTE,
            {"event": "cancel", "order_id": order_id, "result": result},
            source=self._broker_source,
        )
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _append(
        self,
        ledger: Ledger,
        counter: _Counter,
        kind: RecordKind,
        payload: dict[str, Any],
        *,
        source: DataSource,
    ) -> None:
        ledger.append(kind, payload, source=source)
        counter.count += 1

    def _send(
        self,
        notify: Callable[[str], None],
        ledger: Ledger,
        counter: _Counter,
        *,
        mode: Mode,
        compose: Callable[[], str],
        fallback: Callable[[], str],
    ) -> str:
        """Compose, send, and never send a sentence the grammar refused.

        A refusal is a bug in a reason string somewhere upstream, so it is
        recorded rather than swallowed — and the user still gets told something
        happened, which is the half of the fail-safe that is easy to drop.
        """
        try:
            sentence = compose()
        except NotificationRefused as exc:
            self._append(
                ledger,
                counter,
                RecordKind.NOTE,
                {"event": "notification_withheld", "reason": str(exc)},
                source=DataSource.RUNTIME,
            )
            sentence = fallback()
        notify(sentence)
        return sentence

    def _not_placed(
        self,
        notify: Callable[[str], None],
        ledger: Ledger,
        counter: _Counter,
        *,
        mode: Mode,
        rule_id: str,
        reason: str,
    ) -> str:
        """Say an order a rule or a model asked for was not placed.

        `rule_id` is the decision's own actor: a rule id, or `model:<id>`. Which
        sentence is composed follows from that, so a model-driven agent is never
        described as a rule firing and a rule agent is never described as a
        model — the naming rule, at the one place a sentence is built.
        """
        model = model_id_of(rule_id)
        if model is not None:
            return self._send(
                notify,
                ledger,
                counter,
                mode=mode,
                compose=lambda: grammar.model_not_placed(model=model, reason=reason, mode=mode),
                fallback=lambda: grammar.model_withheld(model=model, mode=mode),
            )
        return self._send(
            notify,
            ledger,
            counter,
            mode=mode,
            compose=lambda: grammar.fired_not_placed(rule_id=rule_id, reason=reason, mode=mode),
            fallback=lambda: grammar.withheld(rule_id=rule_id, mode=mode),
        )

    def _filled(
        self,
        notify: Callable[[str], None],
        ledger: Ledger,
        counter: _Counter,
        *,
        mode: Mode,
        rule_id: str,
        fill: Fill,
    ) -> str:
        """Say an order executed, in the words that fit who proposed it."""
        model = model_id_of(rule_id)
        if model is not None:
            return self._send(
                notify,
                ledger,
                counter,
                mode=mode,
                compose=lambda: grammar.model_filled(
                    model=model, description=fill.describe(), mode=mode
                ),
                fallback=lambda: grammar.model_withheld(model=model, mode=mode),
            )
        return self._send(
            notify,
            ledger,
            counter,
            mode=mode,
            compose=lambda: grammar.fired_filled(
                rule_id=rule_id, description=fill.describe(), mode=mode
            ),
            fallback=lambda: grammar.withheld(rule_id=rule_id, mode=mode),
        )

    def _halt(
        self,
        agent: AgentRun,
        ledger: Ledger,
        counter: _Counter,
        notify: Callable[[str], None],
        *,
        now: datetime,
        mode: Mode,
        event: str,
        detail: dict[str, Any],
        reason: str,
    ) -> TickOutcome:
        """Record why the run stopped, tell the user, and leave it stoppable.

        The one shape every mid-tick failure takes. Nothing is retried, the
        ledger is intact, `STOP` still works, and the next tick checks it
        first — which is what makes invariant 8's "the user can still stop it"
        true after a failure rather than hopeful.
        """
        self._append(
            ledger,
            counter,
            RecordKind.NOTE,
            {"event": event, "reason": reason, "at": now, **detail},
            source=DataSource.RUNTIME,
        )
        sent = self._send(
            notify,
            ledger,
            counter,
            mode=mode,
            compose=lambda: grammar.run_stopped(agent_id=agent.agent_id, reason=reason, mode=mode),
            fallback=lambda: grammar.run_stopped(
                agent_id=agent.agent_id,
                reason="the record carries what happened",
                mode=mode,
            ),
        )
        return self._outcome(
            agent,
            now,
            mode,
            evaluated=False,
            session_open=True,
            stopped=False,
            halted=True,
            halt_reason=reason,
            not_placed=(reason,),
            notifications=(sent,),
            records=counter.count,
        )

    @staticmethod
    def _outcome(
        agent: AgentRun,
        now: datetime,
        mode: Mode,
        *,
        evaluated: bool,
        session_open: bool,
        stopped: bool,
        halted: bool,
        halt_reason: str | None,
        records: int,
        fills: tuple[Fill, ...] = (),
        not_placed: tuple[str, ...] = (),
        notifications: tuple[str, ...] = (),
    ) -> TickOutcome:
        return TickOutcome(
            agent_id=agent.agent_id,
            at=now,
            mode=mode,
            evaluated=evaluated,
            session_open=session_open,
            stopped=stopped,
            halted=halted,
            halt_reason=halt_reason,
            fills=fills,
            not_placed=not_placed,
            notifications=notifications,
            records=records,
        )


class _Counter:
    """How many records a tick appended. A box, so helpers can add to it."""

    __slots__ = ("count",)

    def __init__(self) -> None:
        self.count = 0


def _decision_payload(
    agent: AgentRun,
    evaluation: TickEvaluation,
    now: datetime,
    session_date: Any,
    provenance: dict[str, Any],
    broker: BrokerPort,
) -> dict[str, Any]:
    """The whole evaluation as one record's payload.

    One record per tick rather than one per rule per symbol. A tick is one
    thing that happened and this is what it saw; splitting it into a hundred
    rows would make the chain longer without making it say more.

    `agent` is who decided, from the intent source itself. For a rule agent it
    says only that a rule agent decided — the spec is named two lines up and is
    the whole procedure. For a model agent it carries the model id, the model
    the provider reported having answered, and the hashes of the instructions
    and of the exact prompt, because none of that can be re-derived from the
    record afterwards.
    """
    return {
        "event": "tick",
        "agent_id": agent.agent_id,
        "spec_id": agent.state.spec_id,
        "agent": provenance,
        "at": now,
        "session_date": session_date.isoformat(),
        "decisions": [_decision_row(decision) for decision in evaluation.decisions],
        "prices": dict(sorted(evaluation.prices.items())),
        "equity": evaluation.equity,
        "quote_calls": evaluation.quote_calls,
        "bars_calls": evaluation.bars_calls,
        "broker": getattr(broker, "broker_name", "paper"),
        "profile_hash": getattr(broker, "profile_hash", None),
        "inventory_hash": getattr(broker, "inventory_hash", None),
        "profile_sanction": getattr(broker, "profile_sanction", None),
        "data_class": getattr(broker, "data_class", None),
    }


def _decision_row(decision: Decision) -> dict[str, Any]:
    return {
        "rule_id": decision.rule_id,
        "symbol": decision.symbol,
        "fired": decision.fired,
        "values": list(decision.values),
        "intent": decision.intent,
        "refusal": decision.refusal,
    }
