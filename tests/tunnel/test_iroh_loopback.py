from __future__ import annotations

import asyncio
import os
import time

import iroh
import pytest

from tick.tunnel import ALPN


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("TICK_RUN_IROH_LOOPBACK") != "1",
    reason="set TICK_RUN_IROH_LOOPBACK=1 to bind two loopback Iroh endpoints",
)
@pytest.mark.asyncio
async def test_real_iroh_stream_stays_direct_and_carries_four_mibibytes() -> None:
    """Opt-in acceptance spike: loopback only, with Iroh relay mode disabled."""

    def options() -> iroh.EndpointOptions:
        return iroh.EndpointOptions(
            preset=iroh.preset_minimal(),
            bind_addr="127.0.0.1:0",
            alpns=[ALPN],
            relay_mode=iroh.RelayMode.disabled(),
        )

    server = await iroh.Endpoint.bind(options())
    client = await iroh.Endpoint.bind(options())
    payload = bytes(4 * 1024 * 1024)

    async def receive() -> int:
        incoming = await server.accept_next()
        assert not (await incoming.remote_addr()).is_relay()
        connection = await (await incoming.accept()).connect()
        stream = await connection.accept_bi()
        body = await stream.recv().read_to_end(len(payload) + 1)
        await stream.send().write_all(b"ok")
        await stream.send().finish()
        return len(body)

    task = asyncio.create_task(receive())
    started = time.perf_counter()
    connection = await client.connect(
        iroh.EndpointAddr(server.id(), None, server.bound_sockets()), ALPN
    )
    connected = time.perf_counter()
    stream = await connection.open_bi()
    await stream.send().write_all(payload)
    await stream.send().finish()
    assert await stream.recv().read_to_end(2) == b"ok"
    assert await task == len(payload)
    finished = time.perf_counter()
    selected = [path for path in connection.paths() if path.is_selected]
    assert selected and all(path.is_ip and not path.is_relay for path in selected)
    elapsed = finished - connected
    mib_per_second = 4 / elapsed
    print(
        f"iroh loopback connect_ms={(connected - started) * 1000:.1f} "
        f"throughput_mib_s={mib_per_second:.1f}"
    )
    await client.close()
    await server.close()
