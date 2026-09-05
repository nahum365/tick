from __future__ import annotations

import hashlib
import hmac
from datetime import UTC, datetime

import pytest

from tick.serve import handlers, recovery
from tick.serve.handlers import APIError, ServeContext
from tick.serve.pairing import create_secret, load_secret
from tick.tunnel.state import TunnelInfo, write_tunnel_info


class Metadata:
    def __init__(self, tags: frozenset[str]) -> None:
        self._tags = tags

    def tags(self) -> frozenset[str]:
        return self._tags


def context(home, metadata: Metadata) -> ServeContext:
    return ServeContext(
        home=home,
        env={},
        now=lambda: datetime(2026, 9, 3, 12, 0, tzinfo=UTC),
        pid_alive=lambda _pid: False,
        start_process=lambda _argv: 1,
        signal_process=lambda _pid: None,
        provider_status=lambda: (True, "ok"),
        loopback_status=lambda: (True, "ok"),
        tunnel_status=lambda: (True, "ok"),
        unit_fragments=lambda: (True, "ok", ("paper",)),
        codex_chat_identity=lambda model: {
            "model": model or "fixture-model",
            "codex_cli_version": "0.149.0",
        },
        chat_adapter=lambda _provider, _model, _transcript, _frame, _thread=None: (),
        setup_chat_adapter=(
            lambda _provider, _model, _transcript, _frame, _session, _thread=None: ()
        ),
        provider_login_start=lambda: {},
        provider_browser_login_start=lambda _viewport: {},
        provider_login_status=lambda _login: {},
        codex_install=lambda: {
            "code": "CODEX_INSTALLED",
            "path": "/fixture/bin/codex",
            "release": "rust-v0.0.0",
            "sha256": "f" * 64,
            "reason": "fixture installed",
        },
        broker_connect_start=lambda _server, _scheme: {},
        broker_connect_complete=lambda _connect, _url: {},
        broker_connect_status=lambda _connect: {},
        browser_ceremony_url=lambda _purpose: None,
        browser_bridge=object(),
        broker_profile_operation=lambda _action, _body: {},
        commons_client=lambda: object(),  # type: ignore[return-value]
        metadata=metadata,
    )


def test_tunnel_introspection_proves_the_endpoint_id(tmp_path) -> None:
    home = tmp_path / "home"
    create_secret(home)
    serve_context = context(home, Metadata(frozenset()))
    endpoint_id = "12" * 32
    write_tunnel_info(
        home,
        TunnelInfo(
            endpoint_id=endpoint_id,
            udp_port=7434,
            relay_url=None,
            since=datetime(2026, 9, 3, 12, 0, tzinfo=UTC),
            direct_addresses=("0.0.0.0:7434",),
        ),
    )

    payload = handlers.tunnel(serve_context)

    expected = hmac.new(
        load_secret(home).encode("ascii"),
        bytes.fromhex(endpoint_id),
        hashlib.sha256,
    ).hexdigest()
    assert payload == {
        "endpoint_id": endpoint_id,
        "proof": expected,
        "direct_addresses": ["0.0.0.0:7434"],
        "relay_url": None,
        "udp_port": 7434,
    }


def test_recovery_refuses_until_exact_metadata_tag_is_visible(tmp_path, monkeypatch) -> None:
    create_secret(tmp_path)
    monkeypatch.setattr(recovery, "RECOVERY_TAG_WAIT_SECONDS", 0.0)
    serve_context = context(tmp_path, Metadata(frozenset()))

    with pytest.raises(APIError) as refused:
        handlers.pair_recover(serve_context, {"nonce": "a" * 32})

    assert refused.value.code == "recovery_tag_missing"
    assert refused.value.reason.endswith("tag this droplet from the app, and retry.")


class LateMetadata:
    """Digital Ocean metadata that shows the tag only after several reads."""

    def __init__(self, tag: str, visible_on_read: int) -> None:
        self._tag = tag
        self._visible_on_read = visible_on_read
        self.reads = 0

    def tags(self) -> frozenset[str]:
        self.reads += 1
        return frozenset({self._tag}) if self.reads >= self._visible_on_read else frozenset()


def test_recovery_keeps_reading_metadata_until_the_tag_propagates(tmp_path, monkeypatch) -> None:
    """The app tags the droplet and asks at once; metadata lags the tags API."""
    _path, old_secret = create_secret(tmp_path)
    nonce = "c" * 32
    metadata = LateMetadata(f"tick-recover-{nonce}", visible_on_read=3)
    slept: list[float] = []
    monkeypatch.setattr(recovery.time, "sleep", slept.append)
    serve_context = context(tmp_path, metadata)

    status, payload = handlers.pair_recover(serve_context, {"nonce": nonce})

    assert status == 200
    assert payload["secret"] != old_secret
    assert metadata.reads == 3
    assert slept == [recovery.RECOVERY_TAG_POLL_SECONDS] * 2


def test_wait_for_recovery_tag_reads_once_more_at_the_deadline() -> None:
    metadata = LateMetadata("tick-recover-" + "d" * 32, visible_on_read=2)
    clock = iter([0.0, 0.0, 10.0])

    assert recovery.wait_for_recovery_tag(
        metadata,
        "tick-recover-" + "d" * 32,
        wait_seconds=5.0,
        poll_seconds=2.0,
        sleep=lambda _s: None,
        monotonic=lambda: next(clock),
    )
    assert metadata.reads == 2


def test_wait_for_recovery_tag_gives_up_after_the_window() -> None:
    metadata = Metadata(frozenset())
    ticks = iter([0.0, 0.0, 2.0, 4.0, 6.0])

    assert not recovery.wait_for_recovery_tag(
        metadata,
        "tick-recover-" + "e" * 32,
        wait_seconds=5.0,
        poll_seconds=2.0,
        sleep=lambda _s: None,
        monotonic=lambda: next(ticks),
    )


def test_wait_for_recovery_tag_stays_under_the_tunnel_idle_window() -> None:
    assert recovery.RECOVERY_TAG_WAIT_SECONDS < 30.0


def test_recovery_rotates_secret_records_it_and_returns_the_new_derived_id(tmp_path) -> None:
    _path, old_secret = create_secret(tmp_path)
    nonce = "b" * 32
    serve_context = context(tmp_path, Metadata(frozenset({f"tick-recover-{nonce}"})))

    status, payload = handlers.pair_recover(serve_context, {"nonce": nonce})

    assert status == 200
    assert payload["secret"] != old_secret
    assert load_secret(serve_context.home) == payload["secret"]
    assert len(payload["endpoint_id"]) == 64
    records = (serve_context.home / "pairing" / "records.jsonl").read_text()
    assert "pairing_secret_recovered" in records
    assert payload["secret"] not in records
