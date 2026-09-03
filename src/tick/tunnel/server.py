"""Iroh stream accept loop that forwards one HTTP request to box loopback."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from tick.records import ensure_private_dir, write_private_file
from tick.serve.pairing import load_secret

from .identity import derive_secret_key_bytes
from .state import TunnelInfo, write_tunnel_info

ALPN = b"tick/box-api/1"
MAX_REQUEST_BYTES = 8 * 1024 * 1024
HEADER_LIMIT = 64 * 1024
SECRET_MODE = 0o600


class TunnelError(Exception):
    """A stable tunnel refusal with a next action and no sensitive values."""

    def __init__(self, code: str, reason: str) -> None:
        self.code = code
        self.reason = reason
        super().__init__(f"{code}: {reason}")


class TunnelEndpointPort(Protocol):
    async def bind(
        self,
        *,
        bind_addr: str,
        secret_key: bytes,
        alpn: bytes,
        relay_url: str | None,
        relay_token: str | None,
    ) -> Any: ...


class IrohEndpointPort:
    """The only constructor seam around the network-capable Iroh binding."""

    async def bind(
        self,
        *,
        bind_addr: str,
        secret_key: bytes,
        alpn: bytes,
        relay_url: str | None,
        relay_token: str | None,
    ) -> Any:
        import iroh

        if relay_url is None:
            relay_mode = iroh.RelayMode.disabled()
        elif relay_token is None:
            relay_mode = iroh.RelayMode.custom_from_urls([relay_url])
        else:
            relay_map = iroh.RelayMap.empty()
            relay_map.insert(
                iroh.RelayConfig(url=relay_url, quic_port=None, auth_token=relay_token)
            )
            relay_mode = iroh.RelayMode.custom(relay_map)
        return await iroh.Endpoint.bind(
            iroh.EndpointOptions(
                preset=iroh.preset_minimal(),
                bind_addr=bind_addr,
                secret_key=secret_key,
                alpns=[alpn],
                relay_mode=relay_mode,
            )
        )


WriteChunk = Callable[[bytes], Awaitable[None]]
ForwardRequest = Callable[[bytes, WriteChunk], Awaitable[None]]
LogEvent = Callable[[str], None]


def _validated_http_request(raw: bytes) -> bytes:
    marker = raw.find(b"\r\n\r\n")
    if marker < 0 or marker + 4 > HEADER_LIMIT:
        raise TunnelError(
            "TUNNEL_REQUEST_INVALID",
            "the tunneled request headers are incomplete or too large. Retry one box request.",
        )
    header = raw[: marker + 4]
    body = raw[marker + 4 :]
    content_lengths: list[int] = []
    lines = header[:-4].split(b"\r\n")
    if not lines or len(lines[0].split(b" ")) != 3:
        raise TunnelError(
            "TUNNEL_REQUEST_INVALID",
            "the tunneled HTTP request line is invalid. Retry through a current Tick app.",
        )
    forwarded = [lines[0]]
    for line in lines[1:]:
        name, separator, value = line.partition(b":")
        if not separator:
            raise TunnelError(
                "TUNNEL_REQUEST_INVALID",
                "a tunneled HTTP header is invalid. Retry through a current Tick app.",
            )
        lowered = name.strip().lower()
        if lowered == b"content-length":
            try:
                content_lengths.append(int(value.strip()))
            except ValueError as exc:
                raise TunnelError(
                    "TUNNEL_REQUEST_INVALID",
                    "Content-Length is invalid. Retry one box request.",
                ) from exc
        if lowered != b"connection":
            forwarded.append(line)
    if (
        len(content_lengths) > 1
        or (not content_lengths and body)
        or (content_lengths and content_lengths[0] != len(body))
    ):
        raise TunnelError(
            "TUNNEL_REQUEST_INCOMPLETE",
            "the tunneled request body does not match Content-Length. Retry one box request.",
        )
    forwarded.append(b"Connection: close")
    return b"\r\n".join(forwarded) + b"\r\n\r\n" + body


async def forward_to_loopback(request: bytes, write_chunk: WriteChunk) -> None:
    """Stream loopback response bytes into Iroh as they arrive.

    A fresh TCP connection and `Connection: close` delimit each response. This
    preserves the box API's NDJSON timing without teaching the tunnel routes.
    """
    reader, writer = await asyncio.open_connection("127.0.0.1", 7433)
    try:
        writer.write(request)
        await writer.drain()
        writer.write_eof()
        while chunk := await reader.read(64 * 1024):
            await write_chunk(chunk)
    finally:
        writer.close()
        await writer.wait_closed()


class TunnelServer:
    """Fail closed before accepting a connection whose current path is relayed."""

    def __init__(self, *, forward: ForwardRequest, log_event: LogEvent) -> None:
        self._forward = forward
        self._log_event = log_event

    async def serve(self, endpoint: Any) -> None:
        while incoming := await endpoint.accept_next():
            asyncio.create_task(self._accept(incoming))

    async def _accept(self, incoming: Any) -> None:
        remote = await incoming.remote_addr()
        if remote.is_relay():
            await incoming.refuse()
            self._log_event("TUNNEL_RELAYED_REFUSED")
            return
        connection = await (await incoming.accept()).connect()
        try:
            while True:
                stream = await connection.accept_bi()
                asyncio.create_task(self._stream(stream))
        except Exception:  # connection closure ends this peer's accept loop
            return

    async def _stream(self, stream: Any) -> None:
        try:
            raw = await stream.recv().read_to_end(MAX_REQUEST_BYTES + 1)
            if len(raw) > MAX_REQUEST_BYTES:
                raise TunnelError(
                    "TUNNEL_REQUEST_TOO_LARGE",
                    "the tunneled request exceeds 8 MiB. Send a smaller box request.",
                )
            sender = stream.send()
            await self._forward(_validated_http_request(raw), sender.write_all)
            await sender.finish()
        except TunnelError as exc:
            self._log_event(exc.code)
        except Exception:
            self._log_event("TUNNEL_STREAM_FAILED")


def _persist_secret(home: Path, secret_key: bytes) -> Path:
    path = home / "tunnel" / "secret"
    ensure_private_dir(path.parent)
    encoded = secret_key.hex() + "\n"
    if not path.exists() or path.read_text(encoding="ascii") != encoded:
        write_private_file(path, encoded)
    path.chmod(SECRET_MODE)
    return path


async def run_tunnel(
    *,
    home: Path,
    udp_port: int,
    relay_url: str | None,
    relay_token: str | None,
    endpoint_port: TunnelEndpointPort,
    forward: ForwardRequest,
    log_event: LogEvent,
    now: Callable[[], datetime],
) -> str:
    """Bind the derived identity and rebind when recovery rotates it."""
    if not 1 <= udp_port <= 65535:
        raise TunnelError(
            "TUNNEL_PORT_INVALID",
            "--udp-port must be between 1 and 65535. Choose an available UDP port and retry.",
        )
    while True:
        pairing_secret = load_secret(home)
        secret_key = derive_secret_key_bytes(pairing_secret)
        _persist_secret(home, secret_key)
        endpoint = await endpoint_port.bind(
            bind_addr=f"0.0.0.0:{udp_port}",
            secret_key=secret_key,
            alpn=ALPN,
            relay_url=relay_url,
            relay_token=relay_token,
        )
        endpoint_id = str(endpoint.id())
        started = now()
        if started.tzinfo is None or started.utcoffset() is None:
            await endpoint.close()
            raise ValueError("tunnel clock must return a timezone-aware datetime")
        write_tunnel_info(
            home,
            TunnelInfo(
                endpoint_id=endpoint_id,
                udp_port=udp_port,
                relay_url=relay_url,
                since=started.astimezone(UTC),
                direct_addresses=tuple(endpoint.bound_sockets()),
            ),
        )
        watcher = asyncio.create_task(
            _close_when_pairing_changes(endpoint, home=home, pairing_secret=pairing_secret)
        )
        try:
            await TunnelServer(forward=forward, log_event=log_event).serve(endpoint)
        finally:
            watcher.cancel()
        if load_secret(home) == pairing_secret:
            return endpoint_id


async def _close_when_pairing_changes(endpoint: Any, *, home: Path, pairing_secret: str) -> None:
    """Recovery changes identity immediately instead of leaving a stale listener."""
    while True:
        await asyncio.sleep(0.5)
        if load_secret(home) != pairing_secret:
            await endpoint.close()
            return


def stderr_event(event: str) -> None:
    logging.getLogger("tick.tunnel").error(event)
