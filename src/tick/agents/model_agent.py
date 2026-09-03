"""The caged model agent: judgment from a model, authority from the runtime.

    agent = ModelAgent(spec, client=client, instructions=instructions)
    evaluation = agent.evaluate_tick(spec, market, portfolio, now)

One tick: the market snapshot for the spec's universe, the account, and the
cage go to the model together with the USER's own instructions file; the model
answers with intents in a strict schema; every intent then meets the same
`apply_cage` a rule agent's intents meet, in the runner, through code the model
cannot call (CLAUDE.md invariant 3).

`evaluate_tick` has deliberately the same signature as `RuleEvaluator`'s, and
returns the same `TickEvaluation`. That is what lets the runner stay ignorant
of which kind of agent it is ticking: approval, the cage, placement, the record
and the notification grammar are one code path for both, so a limit cannot hold
for a spec agent and quietly not hold for a model one.

**Tick contributes the schema, the snapshot and the cage. Nothing else.**
There is no default instruction text, no selection heuristic, no example
strategy, no ranking of the universe and no "candidates" list anywhere in this
package. An agent whose instructions file is missing or empty refuses to run
rather than acquiring a strategy nobody wrote.

**What the model says is a proposal, and every part of it is checked.** A
symbol outside the user's own universe, a sell larger than the position, a
quantity that is not a whole share, a symbol with no quote this tick: each is a
`Refusal` with words in it, carried forward and recorded, never a silently
dropped intent and never an order reduced to one that would fit.

**Nothing here retries.** A reply that cannot be read stops the tick
(`ModelReplyError`). Asking a second time after an answer nobody could read is
a machine arguing with itself on the user's money, and an intent whose fate is
unknown must not be sent twice.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from decimal import Decimal
from typing import Any

from tick.engine import (
    Decision,
    MarketDataPort,
    OrderIntent,
    PortfolioState,
    Quote,
    Refusal,
    RefusalCode,
    TickEvaluation,
    TickMarketCache,
    Unavailable,
    check_cadence,
    engine_arithmetic,
    quantize_money,
)
from tick.spec import Side, canonical_encode, sha256_hex

from .client import ModelClient, ModelRequest
from .errors import InstructionsMissing
from .schema import MAX_REASON_LENGTH, tool_definitions
from .snapshot import build_snapshot, snapshot_json
from .spec import ModelAgentSpec, agent_spec_id

__all__ = ["MAX_OUTPUT_TOKENS", "MODEL_SOURCE_PREFIX", "ModelAgent", "model_id_of"]

#: What an intent proposed by a model carries as its `source`. The runner reads
#: it to choose the sentence it sends, so a model agent's notification names the
#: model rather than pretending a rule fired.
MODEL_SOURCE_PREFIX = "model:"

#: Headroom for one reply. A list of intents is a small document.
MAX_OUTPUT_TOKENS = 4000

#: The one join between the user's instructions and the snapshot. Two newlines
#: and nothing else: every other character in the composed prompt is either the
#: user's own text or the snapshot Tick built from the tick's own numbers.
PROMPT_JOIN = "\n\n"


def model_id_of(source: str) -> str | None:
    """The model id inside an intent's `source`, or `None` if a rule proposed it."""
    if source.startswith(MODEL_SOURCE_PREFIX):
        return source[len(MODEL_SOURCE_PREFIX) :]
    return None


class ModelAgent:
    """One model agent, for one document, with one user-authored instruction set.

    `spec`, `client` and `instructions` are all required and none has a
    default. An instructions text Tick supplied would be Tick authoring a
    strategy; a client Tick constructed would be Tick operating a model
    endpoint; and a spec inferred from the client would be an agent whose cage
    nobody chose.
    """

    def __init__(
        self,
        spec: ModelAgentSpec,
        *,
        client: ModelClient,
        instructions: str,
    ) -> None:
        if not instructions.strip():
            raise InstructionsMissing(
                f"agent {spec.name!r} has no instructions. A model agent runs the "
                f"instructions YOU wrote; Tick ships none and will not invent one. "
                f"Write them to the agent's instructions.md and run it again."
            )
        self._spec = spec
        self._client = client
        self._instructions = instructions
        self._last: dict[str, Any] | None = None

    @property
    def spec(self) -> ModelAgentSpec:
        return self._spec

    @property
    def instructions(self) -> str:
        """The user's own instructions, exactly as written. Never edited here."""
        return self._instructions

    @property
    def instructions_sha256(self) -> str:
        """The hash of the instructions this agent is running.

        Recorded with every decision, so a record read later says which version
        of the user's own words produced it — the instructions file is not
        hashed into the agent id, because a person is expected to edit it.
        """
        return sha256_hex(self._instructions.encode("utf-8"))

    # ------------------------------------------------------------------
    # The prompt
    # ------------------------------------------------------------------

    def compose(self, snapshot: Mapping[str, Any]) -> ModelRequest:
        """The whole request: the user's words, this tick's snapshot, the schema.

        One user message, in one order, every time. `tests/agents/test_prompt.py`
        subtracts the instructions and the snapshot from the composed text and
        asserts that what remains is whitespace — which is the audit's claim
        checked rather than asserted.
        """
        content = self._instructions + PROMPT_JOIN + snapshot_json(snapshot)
        return ModelRequest(
            model=self._spec.model,
            messages=({"role": "user", "content": content},),
            tools=tool_definitions(),
            max_tokens=MAX_OUTPUT_TOKENS,
        )

    def provenance(self) -> dict[str, Any]:
        """What the last tick's decision record says about who decided.

        Empty of numbers and full of identity: the model the document pins, the
        model the provider reported having answered, the hash of the user's
        instructions, and the hash of the exact prompt that was sent. Together
        they are what makes a model's decision reviewable six weeks later.
        """
        if self._last is None:
            return {
                "kind": "model_agent",
                "provider": self._spec.provider,
                "model": self._spec.model,
                "instructions_sha256": self.instructions_sha256,
                "text": "this agent has not been asked anything yet",
            }
        return dict(self._last)

    # ------------------------------------------------------------------
    # One tick
    # ------------------------------------------------------------------

    def evaluate_tick(
        self,
        spec: ModelAgentSpec,
        market: MarketDataPort,
        state: PortfolioState,
        now: datetime,
    ) -> TickEvaluation:
        """Ask the model once, and turn its answer into decisions.

        `spec` is passed in by the runner and must be the document this agent
        was built for — an agent running one cage while its record cites
        another would make the record a lie, so the two ids are compared.
        """
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now must be timezone-aware; market logic runs on ET sessions")
        if agent_spec_id(spec) != agent_spec_id(self._spec):
            raise ValueError(
                f"this model agent was built for {agent_spec_id(self._spec)} and was asked "
                f"to tick {agent_spec_id(spec)}. An agent executes the document it was "
                f"created for; a changed document is a different agent."
            )
        check_cadence(spec.cadence)

        permitted = frozenset(spec.universe) | state.symbols
        cache = TickMarketCache(market, permitted)
        quotes = {symbol: cache.quote(symbol) for symbol in sorted(permitted)}
        equity = state.equity({symbol: quotes[symbol] for symbol in sorted(state.symbols)})

        snapshot = build_snapshot(
            universe=list(spec.universe),
            cadence_kind=spec.cadence.kind,
            quotes=quotes,
            portfolio=state,
            equity=equity,
            cage=spec.cage,
            now=now,
        )
        request = self.compose(snapshot)
        reply = self._client.propose(request)
        self._last = {
            "kind": "model_agent",
            "provider": self._spec.provider,
            "model": self._spec.model,
            "model_reported": reply.model,
            "instructions_sha256": self.instructions_sha256,
            "prompt_sha256": _prompt_hash(request),
            "intents_proposed": len(reply.intents),
        }

        decisions = tuple(
            self._decision(raw, spec=spec, state=state, quotes=quotes, now=now, model=reply.model)
            for raw in reply.intents
        )
        prices = {
            symbol: quote.price
            for symbol, quote in sorted(quotes.items())
            if isinstance(quote, Quote)
        }
        return TickEvaluation(
            decisions=decisions,
            prices=prices,
            equity=equity,
            quote_calls=cache.quote_calls,
            bars_calls=cache.bars_calls,
        )

    # ------------------------------------------------------------------
    # One proposed intent
    # ------------------------------------------------------------------

    def _decision(
        self,
        raw: Any,
        *,
        spec: ModelAgentSpec,
        state: PortfolioState,
        quotes: Mapping[str, Any],
        now: datetime,
        model: str,
    ) -> Decision:
        """Check one proposed intent, and answer with an intent or a refusal.

        Every rejection is a `Refusal` carried into the record. A proposal the
        runtime will not act on is exactly the row a person goes looking for
        later, and dropping it silently would make the record shorter and less
        true.
        """
        rule_id = f"{MODEL_SOURCE_PREFIX}{self._spec.model}"
        symbol = _symbol_of(raw)
        refusal_code, problem = _problem_with(raw, spec=spec, state=state, quotes=quotes)
        if refusal_code is not None:
            return _refused(rule_id, symbol, refusal_code, problem, now)

        assert isinstance(raw, Mapping)
        side = Side(str(raw["side"]))
        qty = int(raw["qty"])
        quote = quotes[symbol]
        assert isinstance(quote, Quote)
        with engine_arithmetic():
            notional = quantize_money(Decimal(qty) * quote.price)
        intent = OrderIntent(
            source=rule_id,
            symbol=symbol,
            side=side,
            qty=qty,
            est_price=quote.price,
            est_notional=notional,
            price_asof=quote.asof,
            price_source=quote.source,
            reason=_reason_of(raw, model=model),
        )
        return Decision(
            rule_id=rule_id,
            symbol=symbol,
            at=now,
            fired=True,
            values=(),
            intent=intent,
            refusal=None,
        )


# ----------------------------------------------------------------------
# Checking a proposal, one reason at a time
# ----------------------------------------------------------------------


def _symbol_of(raw: Any) -> str:
    """The symbol the proposal names, for the record, even when it is nonsense."""
    if isinstance(raw, Mapping):
        value = raw.get("symbol")
        if isinstance(value, str) and value.strip():
            return value.strip()[:16]
    return "unnamed"


def _reason_of(raw: Mapping[str, Any], *, model: str) -> str:
    reason = str(raw["reason"]).strip()[:MAX_REASON_LENGTH]
    return f"{model} proposed this: {reason}"


def _problem_with(
    raw: Any,
    *,
    spec: ModelAgentSpec,
    state: PortfolioState,
    quotes: Mapping[str, Any],
) -> tuple[RefusalCode | None, str]:
    """The first thing wrong with a proposed intent, in a fixed order.

    Precedence is stated so that two simultaneously-true reasons always produce
    the same recorded one: shape → universe → quantity → quote → long-only.
    """
    if not isinstance(raw, Mapping):
        return (
            RefusalCode.MODEL_OUTPUT_INVALID,
            f"the model answered with a {type(raw).__name__} where the schema declares "
            f"an order intent object. Nothing was placed for it.",
        )
    unknown = sorted(set(raw) - {"symbol", "side", "qty", "reason"})
    if unknown:
        return (
            RefusalCode.MODEL_OUTPUT_INVALID,
            f"the proposed intent carries {unknown}, which the schema does not declare. "
            f"Tick reads an order from the shape it offered and never from extra keys.",
        )
    missing = sorted({"symbol", "side", "qty", "reason"} - set(raw))
    if missing:
        return (
            RefusalCode.MODEL_OUTPUT_INVALID,
            f"the proposed intent is missing {missing}. An order missing one of those "
            f"is not an under-specified order, it is a different one.",
        )

    symbol = raw.get("symbol")
    if not isinstance(symbol, str) or symbol.strip() not in spec.universe:
        return (
            RefusalCode.SYMBOL_OUTSIDE_UNIVERSE,
            f"{_symbol_of(raw)} is not in this agent's universe "
            f"({', '.join(spec.universe)}). The universe is the user's own document and "
            f"Tick will not trade outside it, whatever the model proposes.",
        )
    symbol = symbol.strip()

    if str(raw.get("side")) not in {side.value for side in Side}:
        return (
            RefusalCode.MODEL_OUTPUT_INVALID,
            f"{raw.get('side')!r} is not a side. There is buy and there is sell, and "
            f"there is no short side anywhere in this runtime.",
        )
    qty = raw.get("qty")
    if not isinstance(qty, int) or isinstance(qty, bool) or qty < 1:
        return (
            RefusalCode.MODEL_OUTPUT_INVALID,
            f"{qty!r} is not a whole number of shares. Tick trades whole shares and "
            f"refuses an order for a fraction or for nothing.",
        )
    reason = raw.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return (
            RefusalCode.MODEL_OUTPUT_INVALID,
            "the proposed intent carries no reason. A decision with nothing recorded "
            "beside it cannot be reviewed, so it is not placed.",
        )

    quote = quotes.get(symbol)
    if not isinstance(quote, Quote):
        why = quote.reason if isinstance(quote, Unavailable) else "it was not fetched this tick"
        return (
            RefusalCode.QUOTE_UNAVAILABLE,
            f"there is no price for {symbol} this tick ({why}), so the order cannot be "
            f"sized or checked against the cage. Nothing was placed and no price was "
            f"guessed.",
        )

    if str(raw["side"]) == Side.SELL.value:
        held = state.qty(symbol)
        if held == 0:
            return (
                RefusalCode.NO_POSITION_TO_SELL,
                f"the account holds no {symbol}, so there is nothing to sell. Tick is "
                f"long-only and never opens a short.",
            )
        if int(raw["qty"]) > held:
            return (
                RefusalCode.SELL_EXCEEDS_POSITION,
                f"selling {raw['qty']} {symbol} exceeds the {held} held. The order is "
                f"refused whole — never reduced to a smaller sell, and never allowed to "
                f"go short.",
            )
    return (None, "")


def _refused(rule_id: str, symbol: str, code: RefusalCode, reason: str, now: datetime) -> Decision:
    return Decision(
        rule_id=rule_id,
        symbol=symbol,
        at=now,
        fired=True,
        values=(),
        intent=None,
        refusal=Refusal(source=rule_id, symbol=symbol, code=code, reason=reason),
    )


def _prompt_hash(request: ModelRequest) -> str:
    """The hash of the exact prompt sent, messages and tool schema together.

    Both halves, because a change to the schema changes what the model was
    asked as surely as a change to the instructions does.
    """
    return sha256_hex(
        canonical_encode(
            {
                "messages": [dict(message) for message in request.messages],
                "tools": [dict(tool) for tool in request.tools],
            }
        )
    )
