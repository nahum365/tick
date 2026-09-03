"""The model providers Tick ships an adapter for, and what each one needs.

A closed set. The rule that decides membership (owner ruling, 2026-09-02) is
about **what we ship**, not about what the user may do: the official build
carries an adapter only for a path the provider documents for programmatic or
third-party use, never one whose purpose is to route around a consumer-terms
bar, and never one that alters client identity. Everything else is the user's
own business under the unsanctioned-adapter mode.

Two shapes are shipped in this slice:

- **`http_key`** — the provider's own SDK, on the user's API key, read from the
  user's environment for one call (`anthropic_client.py`).
- **`cli`** — a subprocess the user already authenticated, run once per tick
  with a JSON schema for its answer (`codex_client.py`).

A provider's `terms_note` is the sentence Tick shows ONCE, at `tick provider
check`. It is informational. The provider's terms are a contract between the
provider and the user, Tick is not a party to it, and nothing in the runtime
enforces it: per-order approval is the default every agent starts in for
every provider, and standing mode is the user's choice under the user's own
agreement.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum

from .anthropic_client import API_KEY_ENV, AnthropicModelClient
from .client import ModelClient
from .codex_client import CODEX_BINARY, CodexModelClient
from .errors import ProviderUnavailable

__all__ = [
    "PROVIDERS",
    "Provider",
    "ProviderInfo",
    "ProviderShape",
    "availability",
    "client_for",
]


class Provider(StrEnum):
    """A provider the official build ships an adapter for. Pinned in the agent's document."""

    ANTHROPIC = "anthropic"
    CODEX = "codex"


class ProviderShape(StrEnum):
    """How the adapter reaches the provider."""

    #: The provider's SDK on the user's API key.
    HTTP_KEY = "http_key"
    #: A locally installed, user-authenticated command-line tool.
    CLI = "cli"


@dataclass(frozen=True, slots=True)
class ProviderInfo:
    """What a person needs to know to connect one provider, in plain words."""

    provider: Provider
    shape: ProviderShape
    #: What must exist on this machine: an environment variable, or a binary.
    requires: str
    #: The documented path Tick uses. Stated so the shipped-adapter rule is checkable.
    documented_path: str
    #: The one sentence about the provider's terms shown at `tick provider check`.
    terms_note: str

    def available(self, environ: Mapping[str, str]) -> bool:
        if self.shape is ProviderShape.HTTP_KEY:
            return bool(environ.get(self.requires))
        return shutil.which(self.requires) is not None

    def how_to_make_available(self) -> str:
        if self.shape is ProviderShape.HTTP_KEY:
            return f"set {self.requires} in this shell; Tick reads it per call and stores nothing."
        return f"install the `{self.requires}` command and log in with it; Tick stores nothing."


PROVIDERS: Mapping[Provider, ProviderInfo] = {
    Provider.ANTHROPIC: ProviderInfo(
        provider=Provider.ANTHROPIC,
        shape=ProviderShape.HTTP_KEY,
        requires=API_KEY_ENV,
        documented_path="the Anthropic API through the anthropic SDK, on your own key",
        terms_note=(
            "Anthropic's Usage Policy (2025-09-15) treats finance as a high-risk use and "
            "asks for human review of decisions. That is a term of YOUR API agreement; Tick "
            "starts every agent in per-order approval, and standing mode is your choice."
        ),
    ),
    Provider.CODEX: ProviderInfo(
        provider=Provider.CODEX,
        shape=ProviderShape.CLI,
        requires=CODEX_BINARY,
        documented_path=(
            "`codex exec` on your own ChatGPT login, which OpenAI documents for scripted and "
            "scheduled use"
        ),
        terms_note=(
            "OpenAI's Usage Policies (2025-10-29) bar automating high-stakes financial "
            "decisions without human review, on the API and on a ChatGPT subscription alike. "
            "That is a term of YOUR agreement; Tick starts every agent in per-order approval, "
            "and standing mode is your choice."
        ),
    ),
}

_BUILDERS: Mapping[Provider, Callable[[], ModelClient]] = {
    Provider.ANTHROPIC: AnthropicModelClient.for_environment,
    Provider.CODEX: CodexModelClient.for_environment,
}


def availability(provider: Provider, environ: Mapping[str, str] | None = None) -> tuple[bool, str]:
    """Whether this machine can reach `provider` now, and what to do if not."""
    info = PROVIDERS[provider]
    if info.available(os.environ if environ is None else environ):
        return True, f"{provider.value}: available ({info.requires} present)."
    return False, f"{provider.value}: not available. To fix: {info.how_to_make_available()}"


def client_for(provider: Provider) -> ModelClient:
    """The one place a model client is built, from the user's own environment.

    Raises `ProviderUnavailable` (a `ModelAgentError`) when the key or the
    binary is missing, with the fix in the message. Never falls back to a
    different provider than the document pins: an agent whose record says
    `codex` and whose decision came from somewhere else is not a record.
    """
    try:
        builder = _BUILDERS[provider]
    except KeyError as exc:  # pragma: no cover - the enum is the registry
        raise ProviderUnavailable(f"{provider!r} is not a provider Tick ships") from exc
    return builder()
