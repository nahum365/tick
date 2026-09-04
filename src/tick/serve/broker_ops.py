"""Broker profile operations shared by authenticated app routes."""

from __future__ import annotations

import json
import os
import secrets
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from tick.agents import Provider, availability, client_for
from tick.auth import FileTokenStorage, LoopbackAuthorization, build_oauth_provider
from tick.broker import (
    Category,
    MCPSession,
    ModelCategorizer,
    ProfileState,
    build_profile,
    confirm_profile,
    diff_profile,
    edit_proposal,
    has_confirmation_note,
    load_profile,
    load_proposal,
    propose_profile,
    prove_profile,
    save_profile,
    save_proposal,
    streamable_http_session,
    verify_session_profile,
)
from tick.broker.toolmap import dig
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
            redirect_uri_override=None,
            on_callback=None,
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
        server = body.get("server_url")
        if not isinstance(server, str):
            raise ValueError("server_url is required. Name the connected broker server and retry.")
        categorizer = self._categorizer(body)
        session, loopback = self._session(server)
        try:
            proposal = propose_profile(
                session.list_tools(),
                server=server,
                account_id=None,
                proposed_at=datetime.now(UTC),
                categorizer=categorizer,
            )
            path = save_proposal(self.home, proposal)
            counts = {
                "mapped": sum(
                    row.category is not None and row.category.callable
                    for row in proposal.tools.values()
                ),
                "denied": sum(
                    row.category is not None and row.category.denied
                    for row in proposal.tools.values()
                ),
                "unmapped": sum(row.category is None for row in proposal.tools.values()),
            }
            Ledger(self.home / "broker" / "records.jsonl").append(
                RecordKind.NOTE,
                {
                    "event": "broker_profile_proposed",
                    "categorizer_version": proposal.categorizer_version,
                    "tools_total": len(proposal.tools),
                    **counts,
                },
                source=DataSource.RUNTIME,
            )
            return {
                "state": "done",
                "proposal": proposal.model_dump(mode="json"),
                "path": str(path),
            }
        finally:
            session.close()
            loopback.__exit__(None, None, None)

    def _categorizer(self, body: Mapping[str, Any]) -> ModelCategorizer | None:
        named = body.get("provider") or os.environ.get("TICK_PROFILE_PROVIDER")
        model = body.get("model") or os.environ.get("TICK_PROFILE_MODEL", "")
        if named is None:
            connected = [provider for provider in Provider if availability(provider)[0]]
            if len(connected) != 1:
                return None
            provider = connected[0]
        elif isinstance(named, str):
            provider = Provider(named.lower())
        else:
            raise ValueError("provider must name a connected provider. Correct it and retry.")
        if not isinstance(model, str):
            raise ValueError("model must be a provider model id or omitted. Correct it and retry.")
        return ModelCategorizer(client=client_for(provider), model=model)

    def edit(self, tool_name: str, body: Mapping[str, Any]) -> Mapping[str, Any]:
        proposal = edit_proposal(
            load_proposal(self.home),
            tool_name,
            body,
            who="api",
            at=datetime.now(UTC),
        )
        save_proposal(self.home, proposal)
        return {"tool": proposal.tools[tool_name].model_dump(mode="json")}

    def accounts(self) -> Mapping[str, Any]:
        proposal = load_proposal(self.home)
        profile = load_profile(self.home)
        if profile is None:
            raise ValueError(
                "read.accounts is not confirmed. Finalize that tool first, then read accounts."
            )
        mapping = profile.mapping_for(Category.READ_ACCOUNTS)
        session, loopback = self._session(profile.server)
        try:
            verified = verify_session_profile(
                profile,
                session,
                server=profile.server,
                account_id=profile.account_id,
                confirmation_recorded=has_confirmation_note(self.home, profile.profile_hash),
            )
            mapping = verified.mapping_for(Category.READ_ACCOUNTS, require_proof=False)
            arguments = mapping.render({})
            Draft202012Validator(mapping.contract.input_schema).validate(arguments)
            payload = session.call_tool(mapping.contract.name, arguments)
        finally:
            session.close()
            loopback.__exit__(None, None, None)
        candidate = dig(payload, mapping.result.get("items", ""))
        if isinstance(candidate, Mapping):
            rows = [candidate]
        elif isinstance(candidate, list):
            rows = candidate
        else:
            raise ValueError(
                "the confirmed accounts path returned no list. Fix the mapping and try again."
            )
        stored = self._read_account_refs()
        by_number = {
            value["account_number"]: ref
            for ref, value in stored.items()
            if isinstance(value, Mapping) and isinstance(value.get("account_number"), str)
        }
        public: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            number = dig(row, mapping.result.get("account", ""))
            eligible = dig(row, mapping.result.get("eligible", ""))
            kind = dig(row, mapping.result.get("kind", ""))
            if not isinstance(number, str) or not number.strip():
                continue
            if not isinstance(eligible, bool) or not isinstance(kind, str) or not kind.strip():
                continue
            ref = by_number.get(number) or "acct_" + secrets.token_urlsafe(12)
            stored[ref] = {"account_number": number, "eligible": eligible, "kind": kind}
            public.append(
                {
                    "account_number_masked": "••••" + number[-4:],
                    "account_ref": ref,
                    "eligible": eligible,
                    "kind": kind,
                }
            )
        self._write_account_refs(stored)
        eligible = [row for row in public if row["eligible"]]
        if not eligible:
            raise ValueError(
                "Robinhood reports no account accessible to this agent. Review account access "
                "at Robinhood, then read accounts again."
            )
        selected = None
        if len(eligible) == 1:
            selected = self._select_account(eligible[0]["account_ref"], proposal, profile)
        return {"accounts": public, "selected": selected}

    def select_account(self, body: Mapping[str, Any]) -> Mapping[str, Any]:
        if set(body) != {"account_ref"} or not isinstance(body.get("account_ref"), str):
            raise ValueError("account_ref must name one eligible masked account. Choose one.")
        return self._select_account(
            str(body["account_ref"]), load_proposal(self.home), load_profile(self.home)
        )

    def _select_account(self, ref: str, proposal, profile) -> Mapping[str, Any]:
        if profile is None:
            raise ValueError("read.accounts is not confirmed. Finalize it before choosing.")
        record = self._read_account_refs().get(ref)
        if not isinstance(record, Mapping) or record.get("eligible") is not True:
            raise ValueError(
                "that account reference is missing or ineligible. Read accounts and choose an "
                "eligible row."
            )
        number = record.get("account_number")
        if not isinstance(number, str) or not number:
            raise ValueError(
                "that account reference has no local account value. Read accounts again."
            )
        proposal = proposal.model_copy(update={"account_id": number})
        save_proposal(self.home, proposal)
        selected = build_profile(
            server=profile.server,
            account_id=number,
            tools=profile.tools,
            inventory_hash=profile.inventory_hash,
            data_class=profile.data_class,
            sanction=profile.sanction,
            profile_format_version=profile.profile_format_version,
            canonicalizer_version=profile.canonicalizer_version,
            category_registry_version=profile.category_registry_version,
            state=ProfileState.CONFIRMED,
            observed_inventory_hash=profile.observed_inventory_hash,
            drift=profile.drift,
        )
        at = datetime.now(UTC)
        confirm_profile(self.home, selected, actor="box-api", at=at)
        Ledger(self.home / "broker" / "records.jsonl", clock=lambda: at).append(
            RecordKind.NOTE,
            {"event": "broker_account_selected", "account_ref": ref},
            source=DataSource.RUNTIME,
        )
        return {"account_ref": ref}

    def _account_refs_path(self) -> Path:
        return self.home / "broker" / "account-refs.json"

    def _read_account_refs(self) -> dict[str, Any]:
        try:
            value = json.loads(self._account_refs_path().read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        if not isinstance(value, dict):
            raise ValueError("the local account reference map is unreadable. Read accounts again.")
        return value

    def _write_account_refs(self, value: Mapping[str, Any]) -> None:
        write_private_file(self._account_refs_path(), json.dumps(value, sort_keys=True, indent=2))

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
            Ledger(self.home / "broker" / "records.jsonl", clock=lambda: at).append(
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
