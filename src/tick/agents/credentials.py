"""Provider keys owned by the box; never sent to a Tick-operated service."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path

from tick.records import write_private_file


def anthropic_key(home: Path, env: Mapping[str, str]) -> str | None:
    # An explicit process environment retains precedence for existing installations.
    if env.get("ANTHROPIC_API_KEY"):
        return env["ANTHROPIC_API_KEY"]
    path = home / "providers" / "anthropic.json"
    try:
        value = json.loads(path.read_text())["api_key"]
    except FileNotFoundError:
        return None
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "The saved provider credential cannot be read. Reconnect Anthropic."
        ) from exc
    if not isinstance(value, str) or not value:
        raise ValueError("The saved provider credential cannot be read. Reconnect Anthropic.")
    return value


def environment_key() -> str | None:
    home = Path(os.environ.get("TICK_HOME", str(Path.home() / ".tick")))
    return anthropic_key(home, os.environ)


def save_anthropic_key(home: Path, key: str) -> None:
    write_private_file(home / "providers" / "anthropic.json", json.dumps({"api_key": key}) + "\n")
