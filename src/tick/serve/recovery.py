"""Digital Ocean metadata proof for recovering a lost pairing capability."""

from __future__ import annotations

import json
import re
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

METADATA_TAGS_URL = "http://169.254.169.254/metadata/v1/tags"
NONCE = re.compile(r"^[a-f0-9]{32}$")


class MetadataPort(Protocol):
    def tags(self) -> frozenset[str]: ...


class DigitalOceanMetadata:
    """Read only the droplet's local metadata address, never a public service."""

    def tags(self) -> frozenset[str]:
        try:
            with urlopen(METADATA_TAGS_URL, timeout=2) as response:  # noqa: S310 - link-local
                raw = response.read().decode("utf-8")
        except (HTTPError, URLError, TimeoutError, OSError, UnicodeError) as exc:
            raise RuntimeError(
                "Digital Ocean metadata tags are unavailable. Retry recovery on the droplet."
            ) from exc
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = [line.strip() for line in raw.splitlines() if line.strip()]
        if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
            raise RuntimeError(
                "Digital Ocean metadata returned unreadable tags. Retry recovery on the droplet."
            )
        return frozenset(parsed)


def recovery_tag(nonce: str) -> str:
    if not NONCE.fullmatch(nonce):
        raise ValueError(
            "nonce must be 32 lowercase hexadecimal characters. Start recovery again from the app."
        )
    return f"tick-recover-{nonce}"
