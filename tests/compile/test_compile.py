"""The compile flow: success, refusal, the one retry, and the double failure.

The refusal tests are the ones that matter most. Tick may translate a person's
words and may not author a strategy, so the interesting cases are the two ways
a strategy could be authored — the model asked to invent and did, or the model
invented without being asked — and both end as questions to the user, not a
spec on disk.
"""

from __future__ import annotations

import pytest

from tick.compile import (
    FROM_MODEL,
    FROM_TRACEABILITY,
    CompileError,
    CompileRefusal,
    CompileResult,
    ModelReplyError,
    compile_text,
)

from .conftest import FakeAnthropic


def compile_fixture(name: str) -> tuple[FakeAnthropic, object]:
    fake = FakeAnthropic.replaying(name)
    return fake, compile_text(fake.text, fake, model=fake.model)


def test_a_recorded_exchange_compiles_to_a_validated_spec():
    fake, outcome = compile_fixture("simple-cross-buy.json")

    assert isinstance(outcome, CompileResult)
    assert outcome.spec.universe == ["XYZ"]
    assert outcome.spec.rules[0].id == "cross-up"
    assert outcome.spec.rules[0].then.size.shares == 10
    assert outcome.attempts == 1
    assert outcome.model == fake.model
    assert len(fake.calls) == 1


def test_the_explanation_covers_every_rule_id():
    _, outcome = compile_fixture("simple-cross-buy.json")

    assert isinstance(outcome, CompileResult)
    explained = [item.rule_id for item in outcome.explanation]
    assert explained == [rule.id for rule in outcome.spec.rules]
    assert all(item.what_it_does for item in outcome.explanation)
    assert all(item.what_it_cannot_know for item in outcome.explanation)


def test_a_model_that_asks_instead_of_inventing_returns_its_questions():
    """The words name no security and no limits, so nothing is compiled."""
    fake, outcome = compile_fixture("missing-everything.json")

    assert isinstance(outcome, CompileRefusal)
    assert outcome.origin == FROM_MODEL
    assert "Which symbols should this trade?" in outcome.questions
    assert len(fake.calls) == 1, "a refusal is an answer, not something to retry"


def test_an_invented_threshold_is_caught_by_the_traceability_check():
    """The user said 250; the model answered with a 200-bar average nobody named."""
    fake, outcome = compile_fixture("invented-threshold.json")

    assert isinstance(outcome, CompileRefusal)
    assert outcome.origin == FROM_TRACEABILITY
    assert outcome.questions == (
        "How many bars should the lookback in the rule 'above-trend' cover? "
        "Nothing in what you wrote says 200.",
    )
    assert len(fake.calls) == 1


def test_an_invented_symbol_is_caught_by_the_traceability_check():
    """The prompt cannot stop this. The code must, and does."""
    _, outcome = compile_fixture("invented-symbol.json")

    assert isinstance(outcome, CompileRefusal)
    assert outcome.origin == FROM_TRACEABILITY
    assert outcome.questions == (
        "Which symbols should this trade? The compiled spec names ABCD, which you did not name.",
    )


def test_every_number_the_user_did_supply_traces_and_produces_no_question():
    """The success path is the same check passing, not the check being skipped."""
    _, outcome = compile_fixture("simple-cross-buy.json")

    assert isinstance(outcome, CompileResult)
    assert outcome.spec.cage.max_positions == 3
    assert str(outcome.spec.cage.max_order_notional) == "5000"


def test_a_malformed_spec_is_retried_once_with_the_error_and_then_accepted():
    fake, outcome = compile_fixture("invalid-then-valid.json")

    assert isinstance(outcome, CompileResult)
    assert outcome.attempts == 2
    assert len(fake.calls) == 2
    retry = dict(fake.calls[1].messages[-1])
    assert "sma(0)" in retry["content"]
    assert "n must be >= 1" in retry["content"]


def test_the_retry_conversation_adds_only_the_model_answer_and_the_error():
    fake, _ = compile_fixture("invalid-then-valid.json")

    first, second = fake.calls
    assert len(first.messages) == 1
    assert len(second.messages) == 3
    assert [dict(m)["role"] for m in second.messages] == ["user", "assistant", "user"]
    assert dict(second.messages[0]) == dict(first.messages[0])
    assert second.system == first.system


def test_two_malformed_specs_raise_carrying_both_errors():
    fake = FakeAnthropic.replaying("invalid-twice.json")

    with pytest.raises(CompileError) as caught:
        compile_text(fake.text, fake, model=fake.model)

    problems = caught.value.problems
    assert len(problems) == 2
    assert problems[0].startswith("attempt 1:") and "sma(0)" in problems[0]
    assert problems[1].startswith("attempt 2:") and "sma(-3)" in problems[1]
    assert "sma(0)" in str(caught.value) and "sma(-3)" in str(caught.value)
    assert len(fake.calls) == 2, "two attempts, never a third"


def test_a_reply_that_calls_no_tool_is_not_read_as_a_spec():
    fake = FakeAnthropic.replaying("prose-not-a-tool-call.json")

    with pytest.raises(ModelReplyError) as caught:
        compile_text(fake.text, fake, model=fake.model)

    assert "emit_strategy_spec" in str(caught.value)


def test_empty_words_are_refused_before_the_model_is_asked():
    fake = FakeAnthropic.replaying("simple-cross-buy.json")

    with pytest.raises(CompileError):
        compile_text("   \n ", fake, model=fake.model)

    assert fake.calls == []


def test_the_compiler_never_places_or_records_anything(tmp_path, monkeypatch):
    """Invariant 3: the compiler produces a document. It touches no agent, no ledger."""
    monkeypatch.setenv("TICK_HOME", str(tmp_path / "home"))
    _, outcome = compile_fixture("simple-cross-buy.json")

    assert isinstance(outcome, CompileResult)
    assert not (tmp_path / "home").exists()
