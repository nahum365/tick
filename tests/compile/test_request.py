"""What actually goes up: the user's words, and nothing about the account.

This file is the audit's "bring your own model" and "no hosted LLM path"
constraints, tested rather than asserted. A request is a plain frozen record,
so a test can look at exactly what would have been sent — and what is NOT in it
is the point: no positions, no balances, no orders, no account id, no snapshot
of anything the user owns.
"""

from __future__ import annotations

from tick.compile import ASK_TOOL_NAME, EMIT_TOOL_NAME, SYSTEM_PROMPT, compile_text

from .conftest import FakeAnthropic

#: Words that would mean account state had reached the request.
ACCOUNT_WORDS = ("balance", "buying power", "your positions", "holdings", "portfolio value")


def test_the_request_carries_only_the_users_words_and_the_grammar(cross_buy: FakeAnthropic):
    compile_text(cross_buy.text, cross_buy, model=cross_buy.model)

    request = cross_buy.calls[0]
    assert request.system == SYSTEM_PROMPT
    assert [dict(message) for message in request.messages] == [
        {"role": "user", "content": cross_buy.text}
    ]
    assert [tool["name"] for tool in request.tools] == [EMIT_TOOL_NAME, ASK_TOOL_NAME]
    assert request.model == cross_buy.model


def test_no_account_state_is_anywhere_in_the_request(cross_buy: FakeAnthropic):
    """Robinhood data never leaves the machine: none of it is even assembled here."""
    compile_text(cross_buy.text, cross_buy, model=cross_buy.model)

    request = cross_buy.calls[0]
    wire = (request.system + " " + " ".join(str(dict(m)) for m in request.messages)).lower()
    for word in ACCOUNT_WORDS:
        assert word not in wire, f"the request mentions {word!r}"
