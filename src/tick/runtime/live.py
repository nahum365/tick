"""Local and session-bound gates before a real broker operation is armed.

Readiness is a dependency graph, not a profile-wide drift bit. Placing an
order needs exact, confirmed and proven contracts for place, quote, positions,
balances, and order listing (reconciliation). A changed optional history tool
does not disable those independent capabilities; a changed dependency does.

The local pass checks the grant, profile, append-only confirmation evidence,
and stored proof before a socket opens. The session pass receives only a
``VerifiedSessionProfile`` produced from the complete live inventory and
checks each dependency's current per-tool state. The adapter repeats the
authorization check at its own call boundary, so persistence failure cannot
turn drift back into permission.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from tick.auth import FileTokenStorage
from tick.broker import (
    BrokerError,
    Category,
    Profile,
    ToolState,
    VerifiedSessionProfile,
    has_confirmation_note,
    load_profile,
    profile_path,
    save_profile,
)
from tick.broker.profile import ProfileTool, ProofResult, build_profile
from tick.records import DataSource, Ledger, RecordKind

from .modes import ApprovalMode

__all__ = [
    "LIVE_CAPABILITIES",
    "LiveReadiness",
    "check_live_ready",
    "check_local_live_ready",
    "local_actor",
    "record_first_live_place_proof",
]

LIVE_CAPABILITIES: tuple[Category, ...] = (
    Category.ORDER_PLACE,
    Category.READ_QUOTE,
    Category.READ_POSITIONS,
    Category.READ_BALANCES,
    Category.READ_ORDERS,
)


@dataclass(frozen=True, slots=True)
class LiveReadiness:
    """Whether live dependencies are usable and every actionable reason if not."""

    ready: bool
    missing: tuple[str, ...]
    account_id: str | None
    profile: Profile | None = None
    verified: VerifiedSessionProfile | None = None

    def __post_init__(self) -> None:
        if self.ready and self.missing:
            raise ValueError("a ready live configuration has nothing missing")
        if self.ready and self.profile is None:
            raise ValueError("a ready live configuration has a broker profile")
        if not self.ready and not self.missing:
            raise ValueError("a live configuration that is not ready must say what is missing")


def check_live_ready(
    home: str | os.PathLike[str],
    verified: VerifiedSessionProfile,
    *,
    approval_mode: ApprovalMode,
) -> LiveReadiness:
    """Evaluate live dependencies against a required fresh session binding."""
    return _check_live_ready(home, verified, approval_mode)


def check_local_live_ready(
    home: str | os.PathLike[str], *, approval_mode: ApprovalMode
) -> LiveReadiness:
    """Check local prerequisites before opening the exact profile server."""
    return _check_live_ready(home, None, approval_mode)


def _check_live_ready(
    home: str | os.PathLike[str],
    verified: VerifiedSessionProfile | None,
    approval_mode: ApprovalMode,
) -> LiveReadiness:
    approval_mode = ApprovalMode(approval_mode)
    root = Path(home)
    missing: list[str] = []
    storage = FileTokenStorage(root)
    if not storage.connected():
        missing.append(
            f"this machine holds no Robinhood grant ({storage.directory} is empty). "
            "Run: tick connect robinhood"
        )
    try:
        profile = load_profile(root)
    except BrokerError as exc:
        missing.append(
            f"the broker profile at {profile_path(root)} could not be read ({exc}). "
            "Run: tick broker propose --account <your account id>, then tick broker confirm"
        )
        profile = None
    if profile is None:
        missing.append(
            f"no broker profile has been confirmed on this machine ({profile_path(root)} "
            "does not exist). Run: tick broker propose --account <your account id>, "
            "then tick broker confirm"
        )
    else:
        try:
            recorded = has_confirmation_note(root, profile.profile_hash)
        except BrokerError as exc:
            recorded = False
            missing.append(
                f"the broker confirmation ledger could not be verified ({exc}). Start a "
                "successor ledger and confirm the profile again."
            )
        if not recorded:
            missing.append(
                f"profile {profile.profile_hash} has no append-only profile_confirmed note. "
                "Run: tick broker confirm"
            )
        for category in LIVE_CAPABILITIES:
            try:
                mapping = profile.mapping_for(category)
            except BrokerError:
                missing.append(
                    f"live order placement depends on {category.value}, but no tool is "
                    "confirmed for it. Run: tick broker propose, then confirm that tool."
                )
                continue
            proof_required = (
                category is not Category.ORDER_PLACE or approval_mode is ApprovalMode.STANDING
            )
            if proof_required and not mapping.proved:
                missing.append(
                    f"{mapping.contract.name} ({category.value}) is not proven for its "
                    "exact contract and mapping. Run: tick broker prove with the required "
                    "user-supplied probe inputs."
                )
            if verified is not None:
                state = verified.states.get(mapping.contract.name, ToolState.UNMAPPED)
                if state is not ToolState.CONFIRMED:
                    difference = next(
                        (
                            item.sentence()
                            for item in profile.drift
                            if item.tool == mapping.contract.name
                        ),
                        f"{mapping.contract.name}: {state.value}",
                    )
                    missing.append(
                        f"live order placement depends on {category.value}, but {difference}. "
                        "No broker call is permitted; run `tick broker status`, then "
                        "reconfirm that exact tool."
                    )
        if verified is not None and not verified.confirmation_recorded:
            missing.append(
                f"the session is bound to profile {profile.profile_hash}, but its "
                "profile_confirmed note is missing. Confirm it again before live use."
            )
        try:
            preflight = profile.mapping_for(Category.ORDER_PREFLIGHT)
        except BrokerError:
            preflight = None
        if preflight is not None and not preflight.proved:
            missing.append(
                f"{preflight.contract.name} (order.preflight) is offered but not proven. "
                "Run: tick broker prove with the required user-supplied probe inputs."
            )
    if missing:
        return LiveReadiness(
            ready=False,
            missing=tuple(dict.fromkeys(missing)),
            account_id=None,
            profile=profile,
            verified=verified,
        )
    assert profile is not None
    return LiveReadiness(
        ready=True,
        missing=(),
        account_id=profile.account_id,
        profile=profile,
        verified=verified,
    )


def local_actor(env: Mapping[str, str]) -> str:
    """The local terminal actor, or the honest word ``unknown``."""
    for key in ("TICK_ACTOR", "USER", "LOGNAME", "USERNAME"):
        value = env.get(key)
        if value and value.strip():
            return value.strip()
    return "unknown"


def record_first_live_place_proof(
    home: str | os.PathLike[str],
    verified: VerifiedSessionProfile,
    *,
    at: datetime,
    outcome: str,
) -> dict[str, str] | None:
    """Bind the first called live placement outcome to its exact approved mapping."""
    profile = verified.profile
    mapping = profile.mapping_for(Category.ORDER_PLACE)
    if mapping.proved:
        return None
    proof = ProofResult(
        success=True,
        resolved=("broker_order_outcome",),
        unresolved={},
        detail=(
            "the first approved live order reached the exact place contract and returned "
            f"a terminal {outcome} outcome"
        ),
    )
    values = mapping.model_dump(mode="python")
    values.update(
        {
            "proved_contract_hash": mapping.contract.contract_hash,
            "proved_mapping_hash": mapping.mapping_hash,
            "proved_at": at,
            "proof": proof,
        }
    )
    tools = dict(profile.tools)
    tools[mapping.contract.name] = ProfileTool.model_validate(values)
    updated = build_profile(
        server=profile.server,
        account_id=profile.account_id,
        tools=tools,
        inventory_hash=profile.inventory_hash,
        data_class=profile.data_class,
        sanction=profile.sanction,
        profile_format_version=profile.profile_format_version,
        canonicalizer_version=profile.canonicalizer_version,
        category_registry_version=profile.category_registry_version,
        state=profile.state,
        observed_inventory_hash=profile.observed_inventory_hash,
        drift=profile.drift,
    )
    save_profile(home, updated)
    evidence = {
        "proved_by": "first_live_fill",
        "tool": mapping.contract.name,
        "contract_hash": mapping.contract.contract_hash,
        "mapping_hash": mapping.mapping_hash,
        "outcome": outcome,
    }
    Ledger(profile_path(home).with_name("records.jsonl"), clock=lambda: at).append(
        RecordKind.NOTE,
        {"event": "profile_place_proven", **evidence, "at": at},
        source=DataSource.RUNTIME,
    )
    return evidence
