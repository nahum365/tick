"""Private local state describing the currently running direct tunnel."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from tick.records import write_private_file


@dataclass(frozen=True, slots=True)
class TunnelInfo:
    endpoint_id: str
    udp_port: int
    relay_url: str | None
    since: datetime
    direct_addresses: tuple[str, ...]

    def json(self) -> dict[str, object]:
        payload = asdict(self)
        payload["since"] = self.since.isoformat()
        payload["direct_addresses"] = list(self.direct_addresses)
        return payload


def tunnel_info_path(home: str | os.PathLike[str]) -> Path:
    return Path(home) / "tunnel" / "endpoint.json"


def write_tunnel_info(home: Path, info: TunnelInfo) -> Path:
    if info.since.tzinfo is None or info.since.utcoffset() is None:
        raise ValueError("tunnel start time must be timezone-aware")
    return write_private_file(
        tunnel_info_path(home), json.dumps(info.json(), separators=(",", ":")) + "\n"
    )


def load_tunnel_info(home: str | os.PathLike[str]) -> TunnelInfo:
    path = tunnel_info_path(home)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return TunnelInfo(
        endpoint_id=str(payload["endpoint_id"]),
        udp_port=int(payload["udp_port"]),
        relay_url=(str(payload["relay_url"]) if payload.get("relay_url") is not None else None),
        since=datetime.fromisoformat(str(payload["since"])),
        direct_addresses=tuple(str(value) for value in payload["direct_addresses"]),
    )


def tunnel_status(home: Path) -> tuple[bool, str]:
    """Let doctor report local tunnel state without probing any network."""
    try:
        info = load_tunnel_info(home)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        return False, f"start `tick tunnel --udp-port 7434 --no-relay` ({exc})."
    if info.since.tzinfo is None or info.since.utcoffset() is None:
        return False, "the tunnel timestamp has no timezone. Restart `tick tunnel`."
    return (
        True,
        f"tunnel endpoint {info.endpoint_id[:10]}… is configured on UDP {info.udp_port}; "
        "the full id remains available with `tick tunnel-info`.",
    )
