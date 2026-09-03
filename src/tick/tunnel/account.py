"""Fetch the rendezvous configuration without retaining the account bearer."""

from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass(frozen=True, slots=True)
class RelayAccountConfig:
    url: str
    token: str


def fetch_relay(control_plane_url: str, account_session: str) -> RelayAccountConfig:
    """Use a bearer for this call only; neither returned value is written to disk."""
    target = control_plane_url.rstrip("/") + "/v1/relay"
    response = httpx.get(
        target,
        headers={"Authorization": f"Bearer {account_session}"},
        timeout=20.0,
    )
    if response.is_error:
        raise ValueError(
            "the account relay configuration was refused. Sign in again or pass --relay-url."
        )
    payload = response.json()
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("url"), str)
        or not isinstance(payload.get("token"), str)
    ):
        raise ValueError(
            "the account relay configuration was unreadable. Retry or pass --relay-url."
        )
    return RelayAccountConfig(url=payload["url"], token=payload["token"])
