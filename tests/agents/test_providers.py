"""The provider registry is closed, honest about what it needs, and builds nothing of ours."""

from __future__ import annotations

import pytest

from tick.agents import (
    PROVIDERS,
    AnthropicModelClient,
    CodexModelClient,
    ModelAgentError,
    Provider,
    ProviderShape,
    availability,
    client_for,
    parse_agent_spec,
)
from tick.spec import SpecError

from .conftest import model_spec_document


def test_the_registry_covers_exactly_the_shipped_providers():
    assert set(PROVIDERS) == set(Provider) == {Provider.ANTHROPIC, Provider.CODEX}


def test_each_provider_says_what_it_needs_and_which_documented_path_it_uses():
    for info in PROVIDERS.values():
        assert info.requires
        assert info.documented_path
        assert info.terms_note and "YOUR" in info.terms_note
        assert "per-order approval" in info.terms_note


def test_availability_reads_the_environment_it_is_given(monkeypatch: pytest.MonkeyPatch):
    ok, line = availability(Provider.ANTHROPIC, {"ANTHROPIC_API_KEY": "k"})
    assert ok and "available" in line
    ok, line = availability(Provider.ANTHROPIC, {})
    assert not ok and "ANTHROPIC_API_KEY" in line

    monkeypatch.setattr("tick.agents.providers.shutil.which", lambda name: "/usr/bin/codex")
    assert availability(Provider.CODEX, {})[0] is True
    monkeypatch.setattr("tick.agents.providers.shutil.which", lambda name: None)
    ok, line = availability(Provider.CODEX, {})
    assert not ok and "install the `codex` command" in line


def test_client_for_builds_the_pinned_adapter_and_never_another(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    assert isinstance(client_for(Provider.ANTHROPIC), AnthropicModelClient)
    monkeypatch.setattr("tick.agents.codex_client.shutil.which", lambda name: "/usr/bin/codex")
    assert isinstance(client_for(Provider.CODEX), CodexModelClient)


def test_client_for_refuses_with_the_fix_when_the_machine_cannot_reach_the_provider(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ModelAgentError, match="ANTHROPIC_API_KEY"):
        client_for(Provider.ANTHROPIC)
    monkeypatch.setattr("tick.agents.codex_client.shutil.which", lambda name: None)
    with pytest.raises(ModelAgentError, match="codex"):
        client_for(Provider.CODEX)


def test_a_document_pins_its_provider_and_only_a_shipped_one():
    spec = parse_agent_spec(model_spec_document(provider="codex", model="gpt-5.6-terra"))
    assert spec.provider == "codex"
    with pytest.raises(SpecError, match="provider"):
        parse_agent_spec(model_spec_document(provider="mystery"))
    document = model_spec_document()
    del document["provider"]
    with pytest.raises(SpecError, match="provider"):
        parse_agent_spec(document)


def test_the_provider_is_part_of_the_documents_identity():
    from tick.agents import agent_spec_id

    one = parse_agent_spec(model_spec_document(provider="anthropic"))
    two = parse_agent_spec(model_spec_document(provider="codex"))
    assert agent_spec_id(one) != agent_spec_id(two)


def test_shapes_are_the_two_shipped_ones():
    assert PROVIDERS[Provider.ANTHROPIC].shape is ProviderShape.HTTP_KEY
    assert PROVIDERS[Provider.CODEX].shape is ProviderShape.CLI
