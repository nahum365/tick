"""The notification grammar: four templates, always tagged, never advising."""

from __future__ import annotations

import pytest

from tick.runtime import FORBIDDEN_PHRASES, Mode, NotificationRefused, notify


def test_a_fill_reads_as_a_mechanical_past_tense_sentence():
    assert (
        notify.fired_filled(
            rule_id="dip-buyer", description="bought 12 XYZ at $184.20", mode=Mode.PAPER
        )
        == "Your rule 'dip-buyer' fired: bought 12 XYZ at $184.20 — simulated."
    )


def test_a_live_fill_is_tagged_live():
    assert notify.fired_filled(
        rule_id="dip-buyer", description="sold 3 XYZ at $12.00", mode=Mode.LIVE
    ).endswith(" — live.")


def test_a_refused_order_names_the_rule_and_the_reason():
    assert notify.fired_not_placed(
        rule_id="dip-buyer", reason="the account holds no XYZ", mode=Mode.PAPER
    ) == (
        "Your rule 'dip-buyer' fired but the order was rejected: "
        "the account holds no XYZ — simulated."
    )


def test_a_stop_says_which_agent_stopped():
    assert notify.run_stopped(
        agent_id="a1b2c3", reason="the kill switch is set", mode=Mode.PAPER
    ) == ("Tick stopped agent 'a1b2c3': the kill switch is set — simulated.")


@pytest.mark.parametrize("mode", list(Mode))
def test_every_sentence_carries_its_mode_tag(mode: Mode):
    """A simulated run always says so; there is no untagged path (invariant 6)."""
    sentences = [
        notify.fired_filled(rule_id="r", description="bought 1 XYZ at $1.00", mode=mode),
        notify.fired_not_placed(rule_id="r", reason="a reason", mode=mode),
        notify.withheld(rule_id="r", mode=mode),
        notify.run_stopped(agent_id="a", reason="a reason", mode=mode),
    ]
    for sentence in sentences:
        assert sentence.endswith(f" — {mode.tag}.")


@pytest.mark.parametrize("phrase", FORBIDDEN_PHRASES)
def test_no_notification_can_carry_a_forbidden_phrase(phrase: str):
    """The vocabulary of a recommendation is refused wherever it enters."""
    for compose in (
        lambda text: notify.fired_filled(rule_id="r", description=text, mode=Mode.PAPER),
        lambda text: notify.fired_not_placed(rule_id="r", reason=text, mode=Mode.PAPER),
        lambda text: notify.run_stopped(agent_id="a", reason=text, mode=Mode.PAPER),
    ):
        with pytest.raises(NotificationRefused):
            compose(f"the agent {phrase} something")


@pytest.mark.parametrize(
    "text",
    [
        "we RECOMMEND selling",  # case does not launder it
        "recommended a trim",  # nor does an inflection
        "considering the position",
        "two opportunities were seen",
        "you should rebalance",
    ],
)
def test_inflections_and_casing_do_not_get_through(text: str):
    with pytest.raises(NotificationRefused, match="vocabulary of a recommendation"):
        notify.fired_not_placed(rule_id="r", reason=text, mode=Mode.PAPER)


@pytest.mark.parametrize("text", ["the foundation of the position", "the shoulder of the range"])
def test_a_word_that_merely_contains_a_forbidden_one_is_allowed(text: str):
    """`foundation` is not `found`; the check is whole words, not substrings."""
    assert notify.fired_not_placed(rule_id="r", reason=text, mode=Mode.PAPER)


def test_the_check_over_blocks_rather_than_under_blocks():
    """`considerate` is refused with `consider`. Deliberate: the safe direction."""
    with pytest.raises(NotificationRefused):
        notify.fired_not_placed(rule_id="r", reason="a considerate price", mode=Mode.PAPER)


def test_the_withheld_fallback_says_less_and_says_that_it_says_less():
    sentence = notify.withheld(rule_id="dip-buyer", mode=Mode.PAPER)
    assert "withheld" in sentence
    assert "the record carries what happened" in sentence


def test_a_long_reason_is_trimmed_rather_than_sent_whole():
    sentence = notify.fired_not_placed(rule_id="r", reason="x" * 500, mode=Mode.PAPER)
    assert len(sentence) < 320
    assert sentence.endswith("… — simulated.")


def test_a_multi_line_reason_becomes_one_line():
    sentence = notify.fired_not_placed(rule_id="r", reason="one\n  two\tthree", mode=Mode.PAPER)
    assert "one two three" in sentence


def test_a_notification_must_name_its_rule():
    with pytest.raises(ValueError, match="names the rule"):
        notify.fired_not_placed(rule_id="  ", reason="a reason", mode=Mode.PAPER)


def test_a_refusal_notification_must_carry_a_reason():
    with pytest.raises(ValueError, match="carry the reason"):
        notify.fired_not_placed(rule_id="r", reason="   ", mode=Mode.PAPER)


# ----------------------------------------------------------------------
# Model agents: the model id is shown, and no rule is claimed to have fired
# ----------------------------------------------------------------------


def test_a_model_agents_fill_names_the_model_and_never_a_rule():
    sentence = notify.model_filled(
        model="claude-opus-5", description="bought 12 XYZ at $184.20", mode=Mode.PAPER
    )
    assert sentence == ("Your model agent (claude-opus-5) bought 12 XYZ at $184.20 — simulated.")
    assert "rule" not in sentence


def test_a_model_agents_live_fill_is_tagged_live():
    assert notify.model_filled(
        model="claude-opus-5", description="bought 1 XYZ at $10.00", mode=Mode.LIVE
    ).endswith("— live.")


def test_a_model_agents_rejection_says_what_was_proposed_was_rejected():
    sentence = notify.model_not_placed(
        model="claude-opus-5", reason="the cage refused it", mode=Mode.PAPER
    )
    assert sentence == (
        "Your model agent (claude-opus-5) proposed an order that was rejected: "
        "the cage refused it — simulated."
    )


def test_a_model_agents_sentence_must_name_the_model():
    with pytest.raises(ValueError, match="names the model"):
        notify.model_filled(model="  ", description="bought 1 XYZ at $1.00", mode=Mode.PAPER)


def test_a_model_agents_forbidden_phrase_is_refused_like_any_other():
    """One `compose`, so the vocabulary rule cannot hold for one kind and not the other."""
    with pytest.raises(NotificationRefused):
        notify.model_not_placed(
            model="claude-opus-5", reason="you should consider it", mode=Mode.PAPER
        )


def test_the_model_withheld_fallback_says_less_and_says_that_it_says_less():
    sentence = notify.model_withheld(model="claude-opus-5", mode=Mode.PAPER)
    assert "withheld" in sentence
    assert "the record carries what happened" in sentence
    assert "claude-opus-5" in sentence
