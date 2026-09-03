from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tick.cli import app
from tick.serve.pairing import create_secret
from tick.tunnel import ALPN, MAX_REQUEST_BYTES, TunnelError, TunnelServer, run_tunnel
from tick.tunnel.identity import derive_secret_key_bytes, endpoint_id_for_pairing_secret


class FakeSend:
    def __init__(self) -> None:
        self.data = b""
        self.finished = False

    async def write_all(self, value: bytes) -> None:
        self.data += value

    async def finish(self) -> None:
        self.finished = True


class FakeRecv:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.limit: int | None = None

    async def read_to_end(self, limit: int) -> bytes:
        self.limit = limit
        return self.data


class FakeStream:
    def __init__(self, request: bytes) -> None:
        self.receiver = FakeRecv(request)
        self.sender = FakeSend()

    def recv(self) -> FakeRecv:
        return self.receiver

    def send(self) -> FakeSend:
        return self.sender


class FakeRemote:
    def __init__(self, relayed: bool) -> None:
        self.relayed = relayed

    def is_relay(self) -> bool:
        return self.relayed


class FakeIncoming:
    def __init__(self, relayed: bool) -> None:
        self.remote = FakeRemote(relayed)
        self.refused = False

    async def remote_addr(self) -> FakeRemote:
        return self.remote

    async def refuse(self) -> None:
        self.refused = True


class FakeEndpoint:
    def __init__(self, endpoint_id: str) -> None:
        self.endpoint_id = endpoint_id

    def id(self) -> str:
        return self.endpoint_id

    def bound_sockets(self) -> list[str]:
        return ["0.0.0.0:7434"]

    async def accept_next(self):
        return None

    async def close(self) -> None:
        return None


class FakePort:
    def __init__(self, endpoint_id: str) -> None:
        self.endpoint = FakeEndpoint(endpoint_id)
        self.binds: list[dict[str, object]] = []

    async def bind(self, **values):
        self.binds.append(values)
        return self.endpoint


def pairing_value() -> str:
    return base64.urlsafe_b64encode(bytes(range(32))).rstrip(b"=").decode("ascii")


def test_cli_requires_port_and_one_relay_posture() -> None:
    runner = CliRunner()
    missing = runner.invoke(app, ["tunnel", "--no-relay"])
    assert missing.exit_code == 2
    assert "--udp-port" in missing.output

    posture = runner.invoke(app, ["tunnel", "--udp-port", "7434"])
    assert posture.exit_code == 2
    assert posture.output.strip().endswith(
        "Managed boxes use --no-relay; own machines can fetch rendezvous from the account."
    )


def test_identity_derivation_is_stable_and_produces_the_iroh_public_id() -> None:
    secret = pairing_value()

    assert derive_secret_key_bytes(secret).hex() == (
        "312e80a81a53bc5e739ffb0f9ea163be9cb0bf5949a4e7ae33fe42cc69e0075d"
    )
    assert len(endpoint_id_for_pairing_secret(secret)) == 64


@pytest.mark.asyncio
async def test_run_binds_exact_port_and_persists_private_nonsecret_descriptor(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    create_secret(home)
    endpoint_id = "a" * 64
    port = FakePort(endpoint_id)

    result = await run_tunnel(
        home=home,
        udp_port=7434,
        relay_url=None,
        relay_token=None,
        endpoint_port=port,
        forward=_forward_response,
        log_event=lambda _event: None,
        now=lambda: datetime(2026, 9, 3, 12, 0, tzinfo=UTC),
    )

    assert result == endpoint_id
    assert port.binds[0]["bind_addr"] == "0.0.0.0:7434"
    assert port.binds[0]["alpn"] == ALPN
    assert len(port.binds[0]["secret_key"]) == 32
    state_path = home / "tunnel" / "endpoint.json"
    assert state_path.stat().st_mode & 0o777 == 0o600
    state = json.loads(state_path.read_text())
    assert state == {
        "endpoint_id": endpoint_id,
        "udp_port": 7434,
        "relay_url": None,
        "since": "2026-09-03T12:00:00+00:00",
        "direct_addresses": ["0.0.0.0:7434"],
    }
    assert (home / "tunnel" / "secret").stat().st_mode & 0o777 == 0o600


async def _response() -> bytes:
    return b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}"


async def _forward_response(_request: bytes, write) -> None:
    await write(await _response())


@pytest.mark.asyncio
async def test_relayed_incoming_is_refused_before_accept_and_logs_code() -> None:
    events: list[str] = []
    incoming = FakeIncoming(relayed=True)
    server = TunnelServer(forward=_forward_response, log_event=events.append)

    await server._accept(incoming)

    assert incoming.refused is True
    assert events == ["TUNNEL_RELAYED_REFUSED"]


@pytest.mark.asyncio
async def test_stream_forwards_one_sized_request_and_closes_loopback_response() -> None:
    request = b"POST /v1/status HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}"
    stream = FakeStream(request)
    forwarded: list[bytes] = []

    async def forward(value: bytes, write) -> None:
        forwarded.append(value)
        response = await _response()
        await write(response[:20])
        await write(response[20:])

    server = TunnelServer(forward=forward, log_event=lambda _event: None)
    await server._stream(stream)

    assert b"Connection: close\r\n\r\n{}" in forwarded[0]
    assert stream.receiver.limit == MAX_REQUEST_BYTES + 1
    assert stream.sender.data == await _response()
    assert stream.sender.finished is True


@pytest.mark.asyncio
async def test_stream_refuses_a_body_that_does_not_match_content_length() -> None:
    events: list[str] = []
    stream = FakeStream(b"POST / HTTP/1.1\r\nContent-Length: 4\r\n\r\n{}")
    server = TunnelServer(forward=_forward_response, log_event=events.append)

    await server._stream(stream)

    assert events == ["TUNNEL_REQUEST_INCOMPLETE"]


@pytest.mark.asyncio
async def test_stream_refuses_a_request_over_eight_mibibytes() -> None:
    events: list[str] = []
    stream = FakeStream(b"GET / HTTP/1.1\r\n\r\n" + bytes(MAX_REQUEST_BYTES))
    server = TunnelServer(forward=_forward_response, log_event=events.append)

    await server._stream(stream)

    assert events == ["TUNNEL_REQUEST_TOO_LARGE"]


@pytest.mark.asyncio
async def test_invalid_udp_port_refusal_names_the_next_action(tmp_path: Path) -> None:
    create_secret(tmp_path)
    with pytest.raises(TunnelError) as refused:
        await run_tunnel(
            home=tmp_path,
            udp_port=0,
            relay_url=None,
            relay_token=None,
            endpoint_port=FakePort("a" * 64),
            forward=_forward_response,
            log_event=lambda _event: None,
            now=lambda: datetime(2026, 9, 3, 12, 0, tzinfo=UTC),
        )
    assert refused.value.code == "TUNNEL_PORT_INVALID"
    assert refused.value.reason.endswith("Choose an available UDP port and retry.")
