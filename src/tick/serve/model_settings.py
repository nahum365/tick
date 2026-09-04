"""Paired-device model settings; no route is exposed through the model's MCP."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx

from tick.agents import task_models
from tick.agents.credentials import save_anthropic_key


def catalog(context: Any) -> dict[str, Any]:
    from .handlers import APIError

    try:
        return task_models.discover_catalog(context.home, context.env)
    except (ValueError, OSError) as exc:
        raise APIError(
            409,
            "model_settings_unavailable",
            "Model settings could not be read. Reconnect your providers and retry.",
        ) from exc


def connect_anthropic(context: Any, body: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
    from .handlers import APIError, _record_api_mutation

    key = body.get("api_key")
    if set(body) != {"api_key"} or not isinstance(key, str) or not 1 <= len(key.strip()) <= 4096:
        raise APIError(400, "provider_key_required", "Enter your Anthropic API key.")
    if context.env.get("ANTHROPIC_API_KEY"):
        raise APIError(
            409,
            "provider_environment_key",
            "This server already uses ANTHROPIC_API_KEY. Refresh available models to use it, "
            "or remove that environment setting before connecting a different key.",
        )
    try:
        models = task_models.anthropic_models(key.strip())
        if not models:
            raise ValueError("No available models")
        save_anthropic_key(context.home, key.strip())
        _record_api_mutation(context, "providers", "anthropic_connected")
    except (ValueError, OSError, httpx.HTTPError) as exc:
        raise APIError(
            409,
            "provider_connection_failed",
            "Anthropic could not be connected. "
            "Check your key, account access, and server connection.",
        ) from exc
    return 200, {"id": "anthropic", "connected": True, "models": models, "reason": None}


def save(context: Any, body: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
    from .handlers import APIError, _record_api_mutation

    if set(body) != {"presets"} or not isinstance(body["presets"], dict):
        raise APIError(400, "model_presets_invalid", "Choose a model for each task size.")
    try:
        result = task_models.save_presets(context.home, body["presets"], catalog(context))
        _record_api_mutation(context, "providers", "task_presets_saved")
        return 200, result
    except (ValueError, OSError) as exc:
        raise APIError(409, "model_presets_unavailable", str(exc)) from exc


def resolve(context: Any, body: Mapping[str, Any]) -> dict[str, Any]:
    """A tier is resolved once per task and the existing session pins the result."""
    from .handlers import APIError

    if "tier" not in body:
        return dict(body)
    tier = body["tier"]
    if (
        not isinstance(tier, str)
        or tier not in task_models.TIERS
        or "provider" in body
        or "model" in body
    ):
        raise APIError(
            400, "model_tier_invalid", "Choose a task preset or an explicit provider and model."
        )
    found = catalog(context)
    try:
        choices = task_models.validate_choices(found["presets"], found)
    except ValueError as exc:
        raise APIError(409, "model_choice_required", str(exc)) from exc
    result = dict(body)
    result.pop("tier")
    result.update(choices[tier])
    return result
