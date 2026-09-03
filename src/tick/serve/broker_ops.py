"""Broker profile operations shared by authenticated app routes."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tick.auth import FileTokenStorage, LoopbackAuthorization, build_oauth_provider
from tick.broker import (
    MCPSession,
    diff_profile,
    has_confirmation_note,
    load_profile,
    profile_path,
    propose_profile,
    prove_profile,
    save_profile,
    streamable_http_session,
    verify_session_profile,
)
from tick.records import DataSource, Ledger, RecordKind, write_private_file

__all__ = ["BrokerOperations"]


class BrokerOperations:
    """Open one bounded session per explicit app operation, with no retry."""

    def __init__(self, *, home: Path, timeout_seconds: float) -> None:
        self.home = home
        self.timeout = timeout_seconds

    def _session(self, server: str) -> tuple[MCPSession, LoopbackAuthorization]:
        storage = FileTokenStorage(self.home)
        if not storage.connected():
            raise ValueError(
                "no broker grant is present. Connect from the app or run `tick connect "
                "robinhood`, then retry."
            )
        loopback = LoopbackAuthorization(
            port=0,
            timeout_seconds=self.timeout,
            open_browser=False,
            announce=lambda _line: None,
        )
        loopback.__enter__()
        provider = build_oauth_provider(server_url=server, storage=storage, loopback=loopback)
        session = MCPSession(
            streamable_http_session(server, provider), timeout_seconds=self.timeout
        )
        try:
            session.open()
        except Exception:
            loopback.__exit__(None, None, None)
            raise
        return session, loopback

    def propose(self, body: Mapping[str, Any]) -> Mapping[str, Any]:
        account = body.get("account")
        server = body.get("server_url")
        if not isinstance(account, str) or not account.strip() or not isinstance(server, str):
            raise ValueError("account and server_url are required. Name both and retry.")
        session, loopback = self._session(server)
        try:
            proposal = propose_profile(
                session.list_tools(),
                server=server,
                account_id=account,
                proposed_at=datetime.now(UTC),
            )
            path = write_private_file(
                profile_path(self.home).with_name("proposal.json"),
                proposal.model_dump_json(indent=2),
            )
            return {"proposal": proposal.model_dump(mode="json"), "path": str(path)}
        finally:
            session.close()
            loopback.__exit__(None, None, None)

    def prove(self, body: Mapping[str, Any]) -> Mapping[str, Any]:
        probe = body.get("probe")
        if not isinstance(probe, dict):
            raise ValueError(
                "probe must be an object of user-supplied values. Correct it and retry."
            )
        profile = load_profile(self.home)
        if profile is None:
            raise ValueError("no broker profile exists. Propose and confirm one, then retry.")
        session, loopback = self._session(profile.server)
        at = datetime.now(UTC)
        try:
            verified = verify_session_profile(
                profile,
                session,
                server=profile.server,
                account_id=profile.account_id,
                confirmation_recorded=has_confirmation_note(self.home, profile.profile_hash),
            )
            proven, outcomes = prove_profile(profile, verified, probe_values=probe, at=at)
            save_profile(self.home, proven)
            Ledger(profile_path(self.home).with_name("records.jsonl"), clock=lambda: at).append(
                RecordKind.NOTE,
                {
                    "event": "profile_proven",
                    "profile_hash": proven.profile_hash,
                    "inventory_hash": verified.inventory_hash,
                    "outcome": {
                        name: result.model_dump(mode="json") for name, result in outcomes.items()
                    },
                    "via": "api",
                    "at": at,
                },
                source=DataSource.RUNTIME,
            )
            return {
                "profile_hash": proven.profile_hash,
                "outcome": {
                    name: result.model_dump(mode="json") for name, result in outcomes.items()
                },
            }
        finally:
            session.close()
            loopback.__exit__(None, None, None)

    def diff(self) -> Mapping[str, Any]:
        profile = load_profile(self.home)
        if profile is None:
            return {"state": "none", "diff": []}
        session, loopback = self._session(profile.server)
        try:
            from tick.broker import contract_for

            differences = diff_profile(
                profile, tuple(contract_for(tool) for tool in session.list_tools())
            )
            return {
                "state": "confirmed" if not differences else "drifted",
                "diff": [item.model_dump(mode="json") for item in differences],
            }
        finally:
            session.close()
            loopback.__exit__(None, None, None)
