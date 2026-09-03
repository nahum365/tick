"""What the user is told before the browser opens — in plain words, first.

The 2026-08-31 terms audit made this a required step rather than good manners.
Robinhood's grant is wider than what Tick uses: authorising it gives the
holder of the token **read access to every Robinhood account you have**,
account numbers included, while *trading* is confined to the Agentic account.
Tick narrows its own reads to the one account you configure, but that is
Tick's restraint, not the grant's — and a restraint the user was never told
about is not informed consent.

So the disclosure is printed **before** the authorization URL, every time, and
a test asserts the order. It is deliberately not a link to a document: a
person about to hand over a brokerage grant should not have to go and read
something to find out what they are handing over.

The last line exists because of a specific confusion this ceremony produces.
Tick registers itself dynamically with Robinhood's authorization server, so
the consent screen shows whatever name that server assigns the registration —
which may not say "Tick". A user who expects their own product's name and sees
something else is right to hesitate, so they are told in advance.
"""

from __future__ import annotations

from collections.abc import Iterable

__all__ = ["DISCLOSURE_LINES", "disclosure_text"]

#: One line per fact, so a caller can render them as bullets or as prose and a
#: test can assert on the fact rather than on the paragraph it sits in.
DISCLOSURE_LINES: tuple[str, ...] = (
    "You are about to authorise Tick against your Robinhood account.",
    (
        "Robinhood's grant is wider than what Tick uses: it gives the holder of the "
        "token READ access to ALL of your Robinhood accounts, including your account "
        "numbers, not only the Agentic one."
    ),
    (
        "Trading is confined by Robinhood to your Agentic account. No order can be "
        "placed in any other account with this grant."
    ),
    (
        "Tick reads only the one Agentic account you configure. Positions, balances "
        "and orders are requested for that account id alone, and anything belonging "
        "to another account is dropped before it reaches the rest of the runtime. "
        "That is Tick restraining itself; the grant itself is wider."
    ),
    (
        "The token is written to this machine, mode 0600, and is used from this "
        "machine only. No Tick service receives it, and no account data leaves here."
    ),
    (
        "Tick registers itself with Robinhood as it connects, so the consent screen "
        "may show a client name their server assigned rather than the word 'Tick'."
    ),
    (
        "You can end this at any time: revoke the grant at Robinhood, and run "
        "`tick disconnect robinhood` to delete the copy on this machine."
    ),
)


def disclosure_text(lines: Iterable[str] = DISCLOSURE_LINES) -> str:
    """The disclosure as one block of text, ready to print."""
    return "\n\n".join(line for line in lines)
