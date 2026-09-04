"""Account-discovered model catalog and explicit, box-local task presets.

Codex model/list is a metadata RPC: no thread or model turn is created.
Anthropic models come from its authenticated Models API. There is no static
model-name fallback and no Tick-hosted credential or inference path.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from tick.records import write_private_file

from .anthropic_client import available_models as anthropic_models
from .codex_client import codex_models
from .credentials import anthropic_key

TIERS = ("small", "medium", "large")


class ModelCatalogError(ValueError):
    pass


class ModelChoice(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    provider: str = Field(pattern=r"^(codex|anthropic)$")
    model: str = Field(min_length=1, max_length=200)


def discover_catalog(
    home: Path,
    env: Mapping[str, str],
    *,
    codex: Callable[[Mapping[str, str]], list[dict[str, str]]] = codex_models,
    anthropic: Callable[[str], list[dict[str, str]]] = anthropic_models,
) -> dict[str, Any]:
    providers = []
    for provider in ("codex", "anthropic"):
        try:
            if provider == "codex":
                models = codex(env)
            else:
                key = anthropic_key(home, env)
                models = anthropic(key) if key else []
            providers.append(
                {"id": provider, "connected": bool(models), "models": models, "reason": None}
            )
        except (OSError, ValueError):
            providers.append(
                {
                    "id": provider,
                    "connected": False,
                    "models": [],
                    "reason": f"Couldn’t load {provider.capitalize()} models. "
                    "Check the connection and retry.",
                }
            )
    try:
        presets = load_presets(home)
        presets_error = None
    except (ValueError, OSError):
        presets = {}
        presets_error = "Saved task presets could not be read. Choose and save each model again."
    return {"providers": providers, "presets": presets, "presets_error": presets_error}


def load_presets(home: Path) -> dict[str, dict[str, str]]:
    try:
        raw = json.loads((home / "providers" / "task-presets.json").read_text())
    except FileNotFoundError:
        return {}
    if not isinstance(raw, dict) or set(raw) != set(TIERS):
        raise ModelCatalogError("Saved task presets are unreadable. Choose your models again.")
    return {tier: ModelChoice.model_validate(raw[tier]).model_dump() for tier in TIERS}


def validate_choices(
    choices: Mapping[str, Any], catalog: Mapping[str, Any]
) -> dict[str, dict[str, str]]:
    if set(choices) != set(TIERS):
        raise ModelCatalogError("Choose a model for small, medium, and large tasks.")
    available = {
        (model["provider"], model["model"])
        for provider in catalog["providers"]
        if provider["connected"]
        for model in provider["models"]
    }
    result = {}
    for tier in TIERS:
        choice = ModelChoice.model_validate(choices[tier])
        if (choice.provider, choice.model) not in available:
            raise ModelCatalogError(
                f"The selected {tier} model is unavailable. Choose a replacement."
            )
        result[tier] = choice.model_dump()
    return result


def save_presets(
    home: Path, choices: Mapping[str, Any], catalog: Mapping[str, Any]
) -> dict[str, Any]:
    result = validate_choices(choices, catalog)
    write_private_file(
        home / "providers" / "task-presets.json", json.dumps(result, sort_keys=True) + "\n"
    )
    return {"presets": result}
