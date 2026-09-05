"""Digital Ocean metadata proof for recovering a lost pairing capability."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

METADATA_TAGS_URL = "http://169.254.169.254/metadata/v1/tags"
NONCE = re.compile(r"^[a-f0-9]{32}$")

# Digital Ocean applies a tag through its API before the droplet's metadata
# service reflects it. Recovery is checked right after the app tags the droplet,
# so the box keeps looking for a bounded time instead of refusing on the first
# read. The bound stays under the tunnel connection's idle timeout so the phone
# still receives the verdict.
RECOVERY_TAG_WAIT_SECONDS = 25.0
RECOVERY_TAG_POLL_SECONDS = 2.0


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


def wait_for_recovery_tag(
    metadata: MetadataPort,
    expected: str,
    *,
    wait_seconds: float | None = None,
    poll_seconds: float | None = None,
    sleep: Callable[[float], None] | None = None,
    monotonic: Callable[[], float] | None = None,
) -> bool:
    """Return True once ``expected`` is visible in droplet metadata, else False at the deadline.

    Every metadata read may raise ``RuntimeError``; the caller decides how to
    report that. The last read happens at or after the deadline so a tag that
    appears late in the window is still honoured.
    """
    wait = RECOVERY_TAG_WAIT_SECONDS if wait_seconds is None else wait_seconds
    poll = RECOVERY_TAG_POLL_SECONDS if poll_seconds is None else poll_seconds
    sleep = time.sleep if sleep is None else sleep
    monotonic = time.monotonic if monotonic is None else monotonic
    deadline = monotonic() + wait
    while True:
        if expected in metadata.tags():
            return True
        remaining = deadline - monotonic()
        if remaining <= 0:
            return False
        sleep(min(poll, remaining))
