"""The guard: nothing in these tests may open a socket.

`tests/compile/conftest.py` makes `socket.socket` raise for every test in this
package. This file proves the guard is real (a socket raises), and then proves
the compiler runs a whole exchange underneath it — so "the tests replay
fixtures, they never call a provider" is enforced rather than promised. It
matters beyond tidiness: a test that reached the real API would bill whoever
ran it and would make the suite depend on a model's mood.
"""

from __future__ import annotations

import socket

import pytest

from tick.compile import API_KEY_ENV, AnthropicSpecProposer, CompileResult, compile_text

from .conftest import FakeAnthropic, NetworkUsedInATest


def test_the_guard_itself_refuses_a_socket():
    with pytest.raises(NetworkUsedInATest):
        socket.socket()

    with pytest.raises(NetworkUsedInATest):
        socket.create_connection(("example.invalid", 443))


def test_a_whole_compile_runs_with_the_network_closed(cross_buy: FakeAnthropic):
    outcome = compile_text(cross_buy.text, cross_buy, model=cross_buy.model)

    assert isinstance(outcome, CompileResult)
    assert len(cross_buy.calls) == 1


def test_building_a_provider_client_opens_nothing_by_itself(monkeypatch):
    """Constructing the SDK client is not a connection; only a call would be."""
    monkeypatch.setenv(API_KEY_ENV, "sk-ant-not-a-real-key")

    assert AnthropicSpecProposer.for_environment() is not None
