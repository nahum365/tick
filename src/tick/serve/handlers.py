"""One function per box route, delegating to the same local modules as the CLI."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from tick import __version__
from tick.agents import ModelAgentError, Provider, availability, is_model_agent
from tick.auth import FileTokenStorage
from tick.broker import (
    BrokerError,
    Category,
    ProfileState,
    ProfileTool,
    confirm_profile,
    load_profile,
    load_proposal,
)
from tick.broker.profile import (
    CANONICALIZER_VERSION,
    CATEGORY_REGISTRY_VERSION,
    PROFILE_FORMAT_VERSION,
    build_profile,
    mapping_hash,
)
from tick.chat import ChatError, ChatSession, ChatTurn, stream_turn
from tick.commons import (
    ClaimBody,
    CommonsClient,
    CommonsClientError,
    canonical_hash,
    contributor_id,
    generate_key,
    load_key,
)
from tick.commons.models import ScreenCriterion
from tick.interview import AgentKind, InterviewError, InterviewSession
from tick.records import (
    DataSource,
    Ledger,
    RecordError,
    RecordKind,
    export_evidence,
    normalize_payload,
    read,
    write_private_file,
)
from tick.runtime import (
    AgentRun,
    ApprovalError,
    ApprovalMode,
    ApprovalQueue,
    Mode,
    RunLease,
    TickRuntimeError,
    acknowledge_demotion,
    boot_id,
    check_local_live_ready,
    create_launch_ticket,
    joined_agent_status,
    launch_lock,
    load_run_lease,
    run_doctor,
    run_state,
    save_run_lease,
    state_summary,
)
from tick.spec import SpecError

from .browser_bridge import BrowserBridge, BrowserBridgeError, Viewport
from .codex_install import default_fetch, install_codex
from .pairing import rotate_secret
from .recovery import DigitalOceanMetadata, MetadataPort, recovery_tag

__all__ = [
    "APIError",
    "ServeContext",
    "agents",
    "agent_document",
    "agent_instructions",
    "agent_approval_mode",
    "approval_decide",
    "approvals",
    "broker_confirm",
    "broker_account_select",
    "broker_accounts",
    "broker_connect_complete",
    "broker_connect_start",
    "broker_connect_status",
    "broker_disconnect",
    "broker_profile_diff",
    "broker_proposal_edit",
    "broker_propose",
    "broker_prove",
    "broker_profile",
    "browser_close",
    "browser_frames",
    "browser_input",
    "browser_session_start",
    "chat_create",
    "chat_delete",
    "chat_get",
    "chat_list",
    "chat_turn",
    "commons_keygen",
    "commons_credits",
    "commons_graph",
    "commons_opt_in",
    "commons_pass",
    "commons_screen",
    "commons_status",
    "doctor",
    "doctor_ack_demotion",
    "draft_get",
    "drafts",
    "health",
    "interview_accept",
    "interview_adopt",
    "interview_answer",
    "interview_start",
    "launch",
    "ledger",
    "ledger_export",
    "ledger_new",
    "notifications",
    "pair_recover",
    "pair_rotate",
    "purge",
    "provider_login_start",
    "provider_browser_login_start",
    "provider_login_status",
    "status",
    "stop",
    "tunnel",
]

_INTERVIEW_ENV_LOCK = threading.Lock()


class APIError(Exception):
    """A stable JSON error and HTTP status with an actionable sentence."""

    def __init__(self, status: int, code: str, reason: str):
        self.status = status
        self.code = code
        self.reason = reason
        super().__init__(f"{code}: {reason}")


def _mutation_body(body: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """Separate chat confirmation evidence from the route's action arguments."""
    values = dict(body)
    via = values.pop("via", "api")
    transcript_hash = values.pop("transcript_hash", None)
    if via not in {"api", "chat"}:
        raise APIError(
            400,
            "mutation_provenance_invalid",
            "via must be api or chat. Send the channel that produced this confirmation.",
        )
    if via == "chat":
        if (
            not isinstance(transcript_hash, str)
            or len(transcript_hash) != 64
            or any(character not in "0123456789abcdef" for character in transcript_hash)
        ):
            raise APIError(
                400,
                "chat_transcript_hash_required",
                "a chat confirmation needs its 64-character transcript_hash. "
                "Refresh the proposal and confirm it again.",
            )
        return values, {"via": "chat", "transcript_hash": transcript_hash}
    if transcript_hash is not None:
        raise APIError(
            400,
            "mutation_provenance_invalid",
            "transcript_hash is only valid with via: chat. Remove it or identify the chat.",
        )
    return values, {"via": "api"}


def record_broker_outcome(home: Path, *, state: str, reason: str | None, tools: int | None) -> None:
    """Record how a broker connection ended when no API request carried the outcome.

    Runs on the connect worker thread, so it must not raise: a failed audit write
    would otherwise vanish with the thread and leave the ledger at "started".
    """
    payload: dict[str, Any] = {
        "event": f"broker_connection_{state}",
        "via": "loopback",
        "at": _aware(datetime.now(UTC)),
    }
    if reason is not None:
        payload["reason"] = reason
    if tools is not None:
        payload["tools_discovered"] = tools
    Ledger(home / "broker" / "records.jsonl", clock=lambda: datetime.now(UTC)).append(
        RecordKind.NOTE, payload, source=DataSource.RUNTIME
    )


def _record_api_mutation(context: ServeContext, domain: str, event: str) -> None:
    """Audit a box mutation without retaining request bodies or credential material."""
    Ledger(context.home / domain / "records.jsonl", clock=context.now).append(
        RecordKind.NOTE,
        {"event": event, "via": "api", "at": _aware(context.now())},
        source=DataSource.RUNTIME,
    )


@dataclass(frozen=True, slots=True)
class ServeContext:
    """Required process seams keep route tests away from real subprocesses and signals."""

    home: Path
    env: Mapping[str, str]
    now: Callable[[], datetime]
    pid_alive: Callable[[int], bool]
    start_process: Callable[[Sequence[str]], int]
    signal_process: Callable[[int], None]
    provider_status: Callable[[], tuple[bool, str]]
    loopback_status: Callable[[], tuple[bool, str]]
    tunnel_status: Callable[[], tuple[bool, str]]
    unit_fragments: Callable[[], tuple[bool, str, Sequence[str]]]
    chat_adapter: Callable[
        [Provider, str | None, tuple[ChatTurn, ...], str], Iterable[Mapping[str, Any]]
    ]
    provider_login_start: Callable[[], Mapping[str, str]]
    provider_browser_login_start: Callable[[Viewport], Mapping[str, str]]
    provider_login_status: Callable[[str], Mapping[str, str]]
    codex_install: Callable[[], Mapping[str, str]]
    broker_connect_start: Callable[[str | None, str | None], Mapping[str, Any]]
    broker_connect_complete: Callable[[str, str], Mapping[str, Any]]
    broker_connect_status: Callable[[str], Mapping[str, Any]]
    browser_ceremony_url: Callable[[str], str | None]
    browser_bridge: BrowserBridge
    broker_profile_operation: Callable[[str, Mapping[str, Any]], Mapping[str, Any]]
    commons_client: Callable[[], CommonsClient]
    metadata: MetadataPort


def default_context(home: Path, env: Mapping[str, str]) -> ServeContext:
    """Production dependencies; tests construct ``ServeContext`` with inert fakes."""

    def alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        return True

    def start(argv: Sequence[str]) -> int:
        process = subprocess.Popen(list(argv), start_new_session=True)  # noqa: S603
        return process.pid

    def send(pid: int) -> None:
        os.kill(pid, signal.SIGTERM)

    from tick.tunnel import tunnel_status

    from .doctor import codex_login_status, loopback_status, systemd_unit_fragments

    port = int(env.get("TICK_SERVE_PORT", "7433"))

    context: ServeContext
    login_manager = None
    browser_login_manager = None
    connect_manager = None
    browser_bridge = BrowserBridge.for_environment(home=home)

    def start_login() -> Mapping[str, str]:
        nonlocal login_manager
        from .provider_login import ProviderLoginManager

        login_manager = ProviderLoginManager.for_environment(now=lambda: datetime.now(UTC))
        return login_manager.start()

    def login_status(login_id: str) -> Mapping[str, str]:
        from .provider_login import ProviderLoginError

        if browser_login_manager is not None:
            try:
                return browser_login_manager.status(login_id)
            except ProviderLoginError as exc:
                if exc.code != "CODEX_LOGIN_NOT_FOUND":
                    raise
        if login_manager is None:
            raise ProviderLoginError(
                "CODEX_LOGIN_NOT_FOUND",
                f"login {login_id} is not active. Start device login again.",
            )
        return login_manager.status(login_id)

    def start_browser_login(viewport: Viewport) -> Mapping[str, str]:
        nonlocal browser_login_manager
        from .provider_login import ProviderBrowserLoginManager

        browser_login_manager = ProviderBrowserLoginManager.for_environment(bridge=browser_bridge)
        return browser_login_manager.start(viewport)

    def start_connect(server_url: str | None, redirect_scheme: str | None) -> Mapping[str, Any]:
        nonlocal connect_manager
        from .broker_connect import BrokerConnectManager

        connect_manager = BrokerConnectManager.for_environment(
            home=home,
            callback_received=lambda: browser_bridge.close_active(
                purpose="broker_connect", reason="callback_received"
            ),
            on_finished=lambda state, reason, tools: record_broker_outcome(
                home, state=state, reason=reason, tools=tools
            ),
        )
        return connect_manager.start(server_url, redirect_scheme)

    def complete_connect(connect_id: str, redirect_url: str) -> Mapping[str, Any]:
        from .broker_connect import BrokerConnectError

        if connect_manager is None:
            raise BrokerConnectError(
                "BROKER_CONNECT_NOT_FOUND",
                f"connection {connect_id} is not active. Start the broker connection again.",
            )
        return connect_manager.complete(connect_id, redirect_url)

    def connect_status(connect_id: str) -> Mapping[str, Any]:
        from .broker_connect import BrokerConnectError

        if connect_manager is None:
            raise BrokerConnectError(
                "BROKER_CONNECT_NOT_FOUND",
                f"connection {connect_id} is not active. Start the broker connection again.",
            )
        return connect_manager.status(connect_id)

    def ceremony_url(purpose: str) -> str | None:
        if purpose == "broker_connect":
            return (
                connect_manager.active_authorization_url() if connect_manager is not None else None
            )
        if purpose == "provider_login":
            return (
                browser_login_manager.active_authorization_url()
                if browser_login_manager is not None
                else None
            )
        return None

    def profile_operation(action: str, body: Mapping[str, Any]) -> Mapping[str, Any]:
        from .broker_ops import BrokerOperations

        operations = BrokerOperations(home=home, timeout_seconds=300.0)
        if action == "propose":
            return operations.propose(body)
        if action == "prove":
            return operations.prove(body)
        if action == "diff":
            return operations.diff()
        if action == "accounts":
            return operations.accounts()
        if action == "account":
            return operations.select_account(body)
        if action.startswith("edit:"):
            return operations.edit(action.removeprefix("edit:"), body)
        raise ValueError(f"unknown broker profile operation {action}")

    def build_commons_client() -> CommonsClient:
        url = env.get("COMMONS_URL")
        if not url:
            raise ValueError(
                "COMMONS_URL is not set. Set the chosen commons service on the box and retry."
            )
        return CommonsClient(url, load_key(home))

    def run_chat(
        provider: Provider,
        model: str | None,
        transcript: tuple[ChatTurn, ...],
        frame: str,
    ) -> Iterable[Mapping[str, Any]]:
        wire = tuple(turn.model_dump(mode="json") for turn in transcript)
        if provider is Provider.CODEX:
            from tick.agents.codex_client import CodexChatClient

            tick_command = shutil.which("tick") or sys.argv[0]
            client = CodexChatClient.for_environment(tick_command=tick_command)
            return client.turn(wire, frame)
        if model is None:
            raise ChatError(
                "CHAT_MODEL_REQUIRED",
                "anthropic chat needs the model the user chose. Create the chat with model.",
            )
        from tick.agents import AnthropicChatClient
        from tick.mcpbox import BoxTools, chat_tool_definitions

        client = AnthropicChatClient.for_environment(max_steps=8, max_tokens=2048)
        tools = BoxTools(context)
        return client.turn(
            model=model,
            transcript=wire,
            frame=frame,
            tools=chat_tool_definitions(),
            call_tool=tools.call,
        )

    context = ServeContext(
        home=home,
        env=env,
        now=lambda: datetime.now(UTC),
        pid_alive=alive,
        start_process=start,
        signal_process=send,
        provider_status=codex_login_status,
        loopback_status=lambda: loopback_status(home, port),
        tunnel_status=lambda: tunnel_status(home),
        unit_fragments=systemd_unit_fragments,
        chat_adapter=run_chat,
        provider_login_start=start_login,
        provider_browser_login_start=start_browser_login,
        provider_login_status=login_status,
        codex_install=lambda: install_codex(home, fetch=default_fetch),
        broker_connect_start=start_connect,
        broker_connect_complete=complete_connect,
        broker_connect_status=connect_status,
        browser_ceremony_url=ceremony_url,
        browser_bridge=browser_bridge,
        broker_profile_operation=profile_operation,
        commons_client=build_commons_client,
        metadata=DigitalOceanMetadata(),
    )
    return context


def health(context: ServeContext) -> dict[str, Any]:
    del context
    return {"tick": __version__}


def tunnel(context: ServeContext) -> dict[str, Any]:
    """Prove that local tunnel state belongs to this pairing capability."""
    from tick.serve.pairing import load_secret
    from tick.tunnel import load_tunnel_info

    try:
        info = load_tunnel_info(context.home)
        endpoint_bytes = bytes.fromhex(info.endpoint_id)
        proof = hmac.new(load_secret(context.home).encode("ascii"), endpoint_bytes, hashlib.sha256)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise APIError(
            503,
            "tunnel_unavailable",
            f"the tunnel identity is unavailable ({exc}). Restart `tick tunnel` and retry.",
        ) from exc
    return {
        "endpoint_id": info.endpoint_id,
        "proof": proof.hexdigest(),
        "direct_addresses": list(info.direct_addresses),
        "relay_url": info.relay_url,
        "udp_port": info.udp_port,
    }


def status(context: ServeContext) -> dict[str, Any]:
    """Join leases, ledgers, state, providers, and broker profile without caching."""
    current_boot = boot_id()
    profile, profile_error = _profile(context)
    result_agents: list[dict[str, Any]] = []
    ledger_ok = profile_error is None
    for agent_id in AgentRun.list_ids(context.home):
        try:
            agent = AgentRun.load(context.home, agent_id)
            summary = joined_agent_status(
                agent, pid_alive=context.pid_alive, current_boot_id=current_boot
            )
            verification = agent.verify_ledger()
            ledger_ok = ledger_ok and verification.ok
            result_agents.append(
                {
                    "id": agent_id,
                    "name": summary["name"],
                    "kind": summary["kind"],
                    # Compatibility for the first app fixtures; current_mode is authority.
                    "mode": summary["current_mode"] or Mode.PAPER.value,
                    "approval": summary["approval"],
                    "last_tick": summary["last_tick"],
                    "stopped": summary["stopped"],
                    "drifted": bool(profile and profile.state is ProfileState.DRIFTED),
                    "run_state": summary["run_state"],
                    "current_mode": summary["current_mode"],
                    "previous_run_mode": summary["previous_run_mode"],
                    "transition": summary["transition"],
                    "attention_required": summary["attention_required"],
                    "last_contact": summary["last_contact"],
                }
            )
        except (TickRuntimeError, SpecError, RecordError):
            ledger_ok = False
            result_agents.append(
                {
                    "id": agent_id,
                    "name": agent_id,
                    "kind": "unknown",
                    "mode": Mode.PAPER.value,
                    "approval": "unknown",
                    "last_tick": None,
                    "stopped": False,
                    "drifted": False,
                    "run_state": "unknown",
                    "current_mode": None,
                    "previous_run_mode": None,
                    "transition": None,
                    "attention_required": True,
                    "last_contact": None,
                }
            )
    provider = {
        name: {"available": availability(Provider(name))[0]}
        for name in (Provider.CODEX.value, Provider.ANTHROPIC.value)
    }
    return {
        "version": __version__,
        "box_time": _aware(context.now()).isoformat(),
        "agents": result_agents,
        "provider": provider,
        "broker": {
            "profile_state": profile.state.value if profile is not None else "none",
            "profile_hash": profile.profile_hash if profile is not None else None,
            "server_host": urlparse(profile.server).hostname if profile is not None else None,
            "sanction": profile.sanction if profile is not None else None,
            # The wizard's broker step is finished only when something the runtime
            # can call has proved; a profile holding just the accounts read is not.
            "tools_confirmed": (
                sum(1 for tool in profile.tools.values() if tool.confirmed_at is not None)
                if profile is not None
                else 0
            ),
            "tools_proved": (
                sum(
                    1
                    for tool in profile.tools.values()
                    if tool.proof is not None and tool.proof.success
                )
                if profile is not None
                else 0
            ),
        },
        "ledger_ok": ledger_ok,
    }


def doctor(context: ServeContext) -> dict[str, Any]:
    """Return the CLI checklist as stable JSON for the control center."""
    return run_doctor(
        context.home,
        now=_aware(context.now()),
        provider_status=context.provider_status,
        loopback_status=context.loopback_status,
        tunnel_status=context.tunnel_status,
        unit_fragments=context.unit_fragments,
        pid_alive=context.pid_alive,
    ).json()


def doctor_ack_demotion(
    context: ServeContext, body: Mapping[str, Any]
) -> tuple[int, dict[str, Any]]:
    if set(body) != {"run_id"} or not isinstance(body.get("run_id"), str):
        raise APIError(
            400,
            "demotion_ack_invalid",
            "the body must name one run_id from doctor. Refresh doctor and retry.",
        )
    report = run_doctor(
        context.home,
        now=_aware(context.now()),
        provider_status=context.provider_status,
        loopback_status=context.loopback_status,
        tunnel_status=context.tunnel_status,
        unit_fragments=context.unit_fragments,
        pid_alive=context.pid_alive,
    )
    observations = next(
        (
            list(check.detail.get("observations", []))
            for check in report.checks
            if check.name == "reboot demotion" and check.detail is not None
        ),
        [],
    )
    try:
        path = acknowledge_demotion(context.home, str(body["run_id"]), observations)
    except ValueError as exc:
        raise APIError(409, "demotion_not_observed", str(exc)) from exc
    _record_api_mutation(context, "doctor", "reboot_demotion_acknowledged")
    return 200, {"acknowledged": body["run_id"], "path": str(path)}


def chat_create(context: ServeContext, body: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
    if not set(body) <= {"provider", "model"} or "provider" not in body:
        raise APIError(
            400,
            "chat_create_invalid",
            "the body needs provider, plus model only for anthropic. Correct it and retry.",
        )
    try:
        session = ChatSession.create(
            context.home,
            provider=Provider(str(body["provider"])),
            model=str(body["model"]) if body.get("model") is not None else None,
            at=_aware(context.now()),
        )
    except (ChatError, ValueError) as exc:
        raise _chat_error(exc) from exc
    return 201, session.metadata


def chat_list(context: ServeContext) -> dict[str, Any]:
    return {"chats": ChatSession.list(context.home)}


def chat_get(context: ServeContext, session_id: str) -> dict[str, Any]:
    session = ChatSession(context.home, session_id)
    try:
        return {
            "chat": session.metadata,
            "transcript": [turn.model_dump(mode="json") for turn in session.turns()],
        }
    except ChatError as exc:
        raise _chat_error(exc) from exc


def chat_turn(
    context: ServeContext, session_id: str, body: Mapping[str, Any]
) -> tuple[dict[str, Any], ...]:
    if set(body) != {"text"} or not isinstance(body.get("text"), str):
        raise APIError(
            400,
            "chat_turn_invalid",
            "the body must contain one text message. Correct it and send the turn again.",
        )
    session = ChatSession(context.home, session_id)
    try:
        metadata = session.metadata
        provider = Provider(metadata["provider"])
        model = metadata.get("model")
        return stream_turn(
            session,
            str(body["text"]),
            at=_aware(context.now()),
            adapter=lambda transcript, frame: context.chat_adapter(
                provider, model, transcript, frame
            ),
        )
    except (ChatError, ValueError) as exc:
        raise _chat_error(exc) from exc


def chat_delete(context: ServeContext, session_id: str) -> tuple[int, dict[str, Any]]:
    try:
        ChatSession(context.home, session_id).delete()
    except ChatError as exc:
        raise _chat_error(exc) from exc
    return 200, {"deleted": session_id}


def provider_login_start(context: ServeContext) -> tuple[int, dict[str, Any]]:
    from .provider_login import ProviderLoginError

    try:
        result = dict(context.provider_login_start())
    except ProviderLoginError as exc:
        raise APIError(409, exc.code.lower(), exc.reason) from exc
    _record_api_mutation(context, "provider", "codex_device_login_started")
    return 202, result


def provider_codex_install(context: ServeContext) -> tuple[int, dict[str, Any]]:
    """Install the pinned Codex CLI on the box; the login flow itself is unchanged."""
    from .codex_install import CodexInstallError

    try:
        result = dict(context.codex_install())
    except CodexInstallError as exc:
        raise APIError(409, exc.code.lower(), exc.reason) from exc
    _record_api_mutation(context, "provider", "codex_installed")
    return 200, result


def provider_login_status(context: ServeContext, login_id: str) -> dict[str, Any]:
    from .provider_login import ProviderLoginError

    try:
        return dict(context.provider_login_status(login_id))
    except ProviderLoginError as exc:
        raise APIError(404, exc.code.lower(), exc.reason) from exc


def agents(context: ServeContext) -> dict[str, Any]:
    """List agents from ``state_summary``, the CLI's existing source of truth."""
    values: list[dict[str, Any]] = []
    for agent_id in AgentRun.list_ids(context.home):
        try:
            summary = state_summary(AgentRun.load(context.home, agent_id))
        except (TickRuntimeError, SpecError, RecordError) as exc:
            values.append({"id": agent_id, "unavailable": str(exc)})
            continue
        values.append({"id": agent_id, **{k: v for k, v in summary.items() if k != "agent_id"}})
    return {"agents": values}


def agent_document(context: ServeContext, agent_id: str) -> dict[str, Any]:
    agent = _agent(context, agent_id)
    return {
        "id": agent.agent_id,
        "document": agent.spec.model_dump(mode="json"),
        "instructions": agent.instructions() if is_model_agent(agent.spec) else None,
        "approval_mode": agent.state.approval.value,
        "state": agent.state.model_dump(mode="json"),
    }


def agent_instructions(
    context: ServeContext, agent_id: str, body: Mapping[str, Any]
) -> tuple[int, dict[str, Any]]:
    body, provenance = _mutation_body(body)
    if set(body) != {"instructions"} or not isinstance(body.get("instructions"), str):
        raise APIError(
            400,
            "instructions_invalid",
            "the body must contain the complete instructions text. Correct it and retry.",
        )
    agent = _agent(context, agent_id)
    if not is_model_agent(agent.spec):
        raise APIError(
            409,
            "instructions_not_applicable",
            "a rule agent runs its document and has no instructions file. Edit a model-driven "
            "agent instead; this agent was unchanged.",
        )
    instructions = str(body["instructions"])
    if not instructions.strip():
        raise APIError(
            400,
            "instructions_empty",
            "instructions cannot be empty. Write the model-driven agent's complete instructions.",
        )
    write_private_file(agent.instructions_path, instructions)
    digest = hashlib.sha256(instructions.encode("utf-8")).hexdigest()
    agent.ledger(clock=context.now).append(
        RecordKind.NOTE,
        {
            "event": "instructions_changed",
            "instructions_sha256": digest,
            **provenance,
            "at": _aware(context.now()),
        },
        source=DataSource.RUNTIME,
    )
    return 200, {"agent_id": agent.agent_id, "instructions_sha256": digest}


def agent_approval_mode(
    context: ServeContext, agent_id: str, body: Mapping[str, Any]
) -> tuple[int, dict[str, Any]]:
    body, provenance = _mutation_body(body)
    if set(body) != {"mode"} or body.get("mode") not in {"each", "standing"}:
        raise APIError(
            400,
            "approval_mode_invalid",
            "the body must be {mode: each|standing}. Choose one and retry.",
        )
    agent = _agent(context, agent_id)
    before = agent.state.approval
    after = ApprovalMode(str(body["mode"]))
    agent.save_state(agent.state.with_approval(after))
    agent.ledger(clock=context.now).append(
        RecordKind.NOTE,
        {
            "event": "approval_mode_changed",
            "from": before.value,
            "to": after.value,
            **provenance,
            "at": _aware(context.now()),
        },
        source=DataSource.RUNTIME,
    )
    return 200, {"agent_id": agent.agent_id, "approval_mode": after.value}


def drafts(context: ServeContext) -> dict[str, Any]:
    root = context.home / "drafts"
    values = []
    if root.is_dir():
        for path in sorted(root.iterdir()):
            if not path.is_dir():
                continue
            try:
                state = InterviewSession(context.home, path.name).state
            except InterviewError:
                continue
            values.append(state.model_dump(mode="json"))
    return {"drafts": values}


def draft_get(context: ServeContext, draft_id: str) -> dict[str, Any]:
    session = InterviewSession(context.home, draft_id)
    try:
        return {
            "draft": session.state.model_dump(mode="json"),
            "question": session.ask(),
            "transcript": session.transcript_path.read_text(encoding="utf-8").splitlines(),
        }
    except (InterviewError, OSError) as exc:
        raise _interview_error(exc) from exc


def ledger(context: ServeContext, agent_id: str, after: int) -> dict[str, Any]:
    """Return verified rows, mapping persisted ``ts`` to app-contract ``at``."""
    if after < 0:
        raise APIError(400, "after_invalid", "after must be a non-negative sequence number.")
    agent = _agent(context, agent_id)
    verification = agent.verify_ledger()
    if not verification.ok:
        raise APIError(
            409,
            "ledger_quarantined",
            f"{verification}. Start a successor ledger on the box before reading this chain.",
        )
    rows = []
    for record in read(agent.ledger_path):
        if record.seq <= after:
            continue
        wire = record.model_dump(mode="json")
        wire["at"] = wire.pop("ts")
        rows.append(wire)
    return {"records": rows, "verified": True, "next_after": rows[-1]["seq"] if rows else after}


def ledger_export(context: ServeContext, agent_id: str) -> dict[str, Any]:
    agent = _agent(context, agent_id)
    destination = context.home / "evidence" / f"{agent.agent_id}.jsonl"
    try:
        export_evidence(agent.ledger_path, destination)
        rows = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
    except (OSError, ValueError, RecordError) as exc:
        raise APIError(
            409,
            "evidence_export_failed",
            f"the redacted export could not be written ({exc}). Inspect the ledger and retry.",
        ) from exc
    return {"agent_id": agent.agent_id, "for_evidence": True, "records": rows}


def ledger_new(
    context: ServeContext, agent_id: str, body: Mapping[str, Any]
) -> tuple[int, dict[str, Any]]:
    if (
        set(body) != {"reason"}
        or not isinstance(body.get("reason"), str)
        or not body["reason"].strip()
    ):
        raise APIError(
            400,
            "successor_reason_required",
            "name why a successor ledger is needed, then retry.",
        )
    agent = _agent(context, agent_id)
    try:
        path, first = agent.start_successor_ledger(clock=context.now)
        note = agent.ledger(clock=context.now).append(
            RecordKind.NOTE,
            {
                "event": "successor_requested",
                "reason": str(body["reason"]),
                "via": "api",
                "at": _aware(context.now()),
            },
            source=DataSource.RUNTIME,
        )
    except (TickRuntimeError, RecordError, OSError) as exc:
        raise APIError(
            409,
            "successor_refused",
            f"the successor ledger was not opened ({exc}). Inspect the chain on the box.",
        ) from exc
    return 201, {"path": str(path), "first_seq": first.seq, "request_seq": note.seq}


def notifications(context: ServeContext, after: int) -> dict[str, Any]:
    if after < 0:
        raise APIError(400, "after_invalid", "after must be non-negative. Correct it and retry.")
    lines: list[dict[str, Any]] = []
    cursor = 0
    for agent_id in AgentRun.list_ids(context.home):
        agent = _agent(context, agent_id)
        for row in read(agent.ledger_path):
            cursor += 1
            if cursor <= after:
                continue
            sentence = row.payload.get("notification") or row.payload.get("reason")
            if isinstance(sentence, str) and sentence.strip():
                lines.append(
                    {
                        "cursor": cursor,
                        "agent_id": agent_id,
                        "record_seq": row.seq,
                        "kind": row.kind.value,
                        "line": sentence,
                        "at": row.ts.isoformat(),
                    }
                )
    return {"notifications": lines, "next_after": cursor}


def commons_status(context: ServeContext) -> dict[str, Any]:
    marker = context.home / "commons" / "opt-in.json"
    key = context.home / "commons" / "key"
    return {
        "opted_in": marker.exists(),
        "key_present": key.exists(),
        "service_configured": bool(context.env.get("COMMONS_URL")),
    }


def commons_keygen(context: ServeContext) -> tuple[int, dict[str, Any]]:
    try:
        key = generate_key(context.home)
    except (FileExistsError, OSError, ValueError) as exc:
        raise APIError(
            409,
            "commons_key_refused",
            f"{exc}. Keep the existing identity or create a new box.",
        ) from exc
    _record_api_mutation(context, "commons", "commons_key_generated")
    return 201, {"contributor_id": contributor_id(key)}


def commons_opt_in(context: ServeContext, body: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
    body, provenance = _mutation_body(body)
    if set(body) != {"confirm"} or body.get("confirm") is not True:
        raise APIError(
            400,
            "commons_confirmation_required",
            "show the public-claim disclosure, then send {confirm: true} if the user opts in.",
        )
    try:
        try:
            key = load_key(context.home)
        except FileNotFoundError:
            key = generate_key(context.home)
        marker = {
            "enabled_at": _aware(context.now()),
            "contributor_id": contributor_id(key),
            "outward_schema_hash": canonical_hash(ClaimBody.model_json_schema()),
            **provenance,
        }
        path = write_private_file(
            context.home / "commons" / "opt-in.json",
            json.dumps(normalize_payload(marker), sort_keys=True) + "\n",
        )
        Ledger(context.home / "commons" / "records.jsonl", clock=context.now).append(
            RecordKind.NOTE,
            {"event": "commons_opted_in", **provenance, "at": _aware(context.now())},
            source=DataSource.RUNTIME,
        )
    except (OSError, ValueError) as exc:
        raise APIError(
            409,
            "commons_opt_in_refused",
            f"the opt-in could not be recorded ({exc}). Work locally or retry.",
        ) from exc
    return 200, {"opted_in": True, "path": str(path)}


def commons_pass(context: ServeContext, ticker: str) -> dict[str, Any]:
    if not ticker.strip():
        raise APIError(400, "ticker_required", "name a public ticker such as XYZ, then retry.")
    try:
        result = context.commons_client().pass_for(ticker)
        return {"pass": result.model_dump(mode="json")}
    except CommonsClientError as exc:
        raise APIError(
            409, exc.code.lower(), f"{exc.reason}. Work locally or retry later."
        ) from exc
    except (OSError, ValueError, RuntimeError) as exc:
        raise APIError(
            409,
            "commons_pass_refused",
            f"the public pass is unavailable ({exc}). Work locally or retry later.",
        ) from exc


def commons_graph(
    context: ServeContext,
    ticker: str,
    depth: int,
    observed_before: str | None,
) -> dict[str, Any]:
    """Read only public claim edges; no runtime or broker object enters this path."""
    if not ticker.strip():
        raise APIError(400, "ticker_required", "name a public ticker such as XYZ, then retry.")
    if depth not in {1, 2}:
        raise APIError(400, "depth_invalid", "depth must be 1 or 2. Choose one and retry.")
    observed = _observed_before(observed_before)
    try:
        result = context.commons_client().graph_for(ticker, depth, observed)
        return result.model_dump(mode="json")
    except CommonsClientError as exc:
        raise APIError(
            409, exc.code.lower(), f"{exc.reason}. Work locally or retry later."
        ) from exc
    except (OSError, ValueError, RuntimeError) as exc:
        raise APIError(
            409,
            "commons_graph_refused",
            f"the public graph is unavailable ({exc}). Work locally or retry later.",
        ) from exc


def commons_screen(
    context: ServeContext,
    encoded_criteria: Sequence[str],
    observed_before: str | None,
) -> dict[str, Any]:
    """Validate explicit criteria before constructing the commons transport."""
    if not encoded_criteria:
        raise APIError(
            400,
            "screen_criteria_required",
            "add at least one predicate criterion; an empty screen cannot list everything.",
        )
    criteria: list[ScreenCriterion] = []
    for encoded in encoded_criteria:
        try:
            decoded = json.loads(encoded)
            criteria.append(ScreenCriterion.model_validate(decoded))
        except (json.JSONDecodeError, ValueError) as exc:
            raise APIError(
                400,
                "screen_criterion_invalid",
                f"each criterion must match the closed predicate comparison shape ({exc}). "
                "Correct the criterion and retry.",
            ) from exc
    observed = _observed_before(observed_before)
    try:
        result = context.commons_client().screen(tuple(criteria), observed)
        return result.model_dump(mode="json")
    except CommonsClientError as exc:
        raise APIError(
            409, exc.code.lower(), f"{exc.reason}. Work locally or retry later."
        ) from exc
    except (OSError, ValueError, RuntimeError) as exc:
        raise APIError(
            409,
            "commons_screen_refused",
            f"the public screen is unavailable ({exc}). Work locally or retry later.",
        ) from exc


def commons_credits(context: ServeContext, observed_before: str | None) -> dict[str, Any]:
    """Read credits for the box public key, never for an agent or brokerage account."""
    observed = _observed_before(observed_before)
    try:
        result = context.commons_client().credits(observed)
        return result.model_dump(mode="json", by_alias=True)
    except CommonsClientError as exc:
        raise APIError(
            409, exc.code.lower(), f"{exc.reason}. Work locally or retry later."
        ) from exc
    except (OSError, ValueError, RuntimeError) as exc:
        raise APIError(
            409,
            "commons_credits_refused",
            f"credits are unavailable ({exc}). Work locally or retry later.",
        ) from exc


def purge(context: ServeContext, body: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
    box_name = context.env.get("TICK_BOX_NAME")
    if not box_name:
        raise APIError(
            409,
            "box_name_missing",
            "TICK_BOX_NAME is not set on this box. Set it locally before requesting a purge.",
        )
    if set(body) != {"confirm"} or body.get("confirm") != box_name:
        raise APIError(
            400,
            "purge_confirmation_mismatch",
            "confirm must exactly match this box name. Read the displayed name and retry.",
        )
    for agent_id in AgentRun.list_ids(context.home):
        agent = _agent(context, agent_id)
        agent.request_stop(reason="purge requested from the box API", at=_aware(context.now()))
        agent.ledger(clock=context.now).append(
            RecordKind.STOP,
            {"event": "purge_requested", "via": "api", "at": _aware(context.now())},
            source=DataSource.RUNTIME,
        )
        lease = load_run_lease(context.home, agent_id)
        if (
            lease is not None
            and run_state(lease, pid_alive=context.pid_alive, current_boot_id=boot_id())
            == "running"
        ):
            try:
                context.signal_process(lease.pid)
            except OSError:
                pass
    shutil.rmtree(context.home)
    return 200, {
        "purged": box_name,
        "reason": (
            "Tick state was deleted after STOP was set for every agent. Destroy the droplet "
            "in your Digital Ocean account to remove the box and stop charges."
        ),
    }


def approvals(context: ServeContext) -> dict[str, Any]:
    values: list[dict[str, Any]] = []
    for agent_id in AgentRun.list_ids(context.home):
        queue = ApprovalQueue.system(context.home, agent_id)
        for request in queue.pending():
            payload = request.model_dump(mode="json")
            payload["expires_at"] = payload["deadline_wall"]
            remaining = (request.deadline_wall - _aware(context.now())).total_seconds()
            payload["remaining_seconds"] = max(0, int(remaining))
            values.append(payload)
    return {"approvals": values}


def approval_decide(
    context: ServeContext, approval_id: str, body: Mapping[str, Any]
) -> tuple[int, dict[str, Any]]:
    body, provenance = _mutation_body(body)
    decision = body.get("decision")
    if set(body) != {"decision"} or decision not in {"approve", "decline"}:
        raise APIError(
            400,
            "approval_decision_invalid",
            "the body must be exactly {decision: approve|decline}. Refresh the "
            "approval and try again.",
        )
    agent_id, queue = _queue_holding(context, approval_id)
    try:
        resolution = queue.decide(
            approval_id,
            approve=decision == "approve",
            decided_via=provenance["via"],
        )
    except ApprovalError as exc:
        raise APIError(exc.status, exc.code, exc.reason) from exc
    if provenance["via"] == "chat":
        _agent(context, agent_id).ledger(clock=context.now).append(
            RecordKind.NOTE,
            {
                "event": "approval_decided_from_chat",
                "approval_id": approval_id,
                "outcome": resolution.outcome.value,
                **provenance,
                "at": _aware(context.now()),
            },
            source=DataSource.RUNTIME,
        )
    return 200, {"resolution": resolution.model_dump(mode="json")}


def stop(
    context: ServeContext, agent_id: str, body: Mapping[str, Any]
) -> tuple[int, dict[str, Any]]:
    """Set and fsync STOP first; every fallible notification follows it."""
    body, provenance = _mutation_body(body)
    if body:
        raise APIError(400, "stop_body_invalid", "stop accepts only chat provenance. Retry it.")
    agent = _agent(context, agent_id)
    existed = agent.stop_requested()
    path = agent.request_stop(reason="stopped from the box API", at=_aware(context.now()))
    for request in ApprovalQueue.system(context.home, agent_id).pending():
        ApprovalQueue.system(context.home, agent_id).abort_by_stop(request.approval_id)
    lease = load_run_lease(context.home, agent_id)
    observed = run_state(lease, pid_alive=context.pid_alive, current_boot_id=boot_id())
    signal_status = "not_running"
    if lease is not None and observed == "running":
        try:
            context.signal_process(lease.pid)
            signal_status = "sent"
        except OSError:
            signal_status = "failed"
    record_status = "existing" if existed else "recorded"
    if not existed:
        try:
            agent.ledger(clock=context.now).append(
                RecordKind.STOP,
                {
                    "event": "stop_requested",
                    "reason": agent.stop_reason(),
                    "already_set": False,
                    **provenance,
                    "at": _aware(context.now()),
                },
                source=DataSource.RUNTIME,
            )
        except (OSError, RecordError):
            record_status = "failed"
    known_stopped = observed == "stopped"
    return 200, {
        "status": "stopped" if known_stopped else "stop_requested",
        "reason": agent.stop_reason(),
        "stop_path": str(path),
        "record_status": record_status,
        "signal_status": signal_status,
    }


def launch(
    context: ServeContext, agent_id: str, body: Mapping[str, Any]
) -> tuple[int, dict[str, Any]]:
    """Start exactly one run; live authority exists only in argv plus its ticket."""
    body, provenance = _mutation_body(body)
    allowed = {"live", "standing_ok", "idempotency_key"}
    if not set(body) <= allowed or set(body) < {"live", "standing_ok"}:
        raise APIError(
            400,
            "launch_invalid",
            "the launch body needs live and standing_ok booleans. Correct it and try again.",
        )
    live = body["live"]
    standing_ok = body["standing_ok"]
    key = body.get("idempotency_key")
    if not isinstance(live, bool) or not isinstance(standing_ok, bool):
        raise APIError(
            400, "launch_invalid", "live and standing_ok must be booleans. Correct the request."
        )
    if key is not None and (not isinstance(key, str) or not key.strip()):
        raise APIError(
            400, "idempotency_key_invalid", "idempotency_key must be a non-empty string."
        )
    agent = _agent(context, agent_id)
    if live and agent.state.approval is ApprovalMode.STANDING and not standing_ok:
        raise APIError(
            409,
            "standing_acknowledgement_required",
            "this agent has standing approval, so live requires standing_ok: true. "
            "Nothing was started; acknowledge it or launch paper.",
        )
    with launch_lock(context.home, agent_id):
        idempotent = _idempotent_launch(context, agent_id, key)
        if idempotent is not None:
            return 200, idempotent
        if agent.stop_requested():
            raise APIError(
                409,
                "agent_stopped",
                f"agent {agent_id} is stopped: {agent.stop_reason()}. Nothing was started; "
                f"remove {agent.stop_path} on the box only when it should run again.",
            )
        existing = load_run_lease(context.home, agent_id)
        if run_state(existing, pid_alive=context.pid_alive, current_boot_id=boot_id()) == "running":
            raise APIError(
                409,
                "already_running",
                f"agent {agent_id} already has a running process. Stop it before "
                "launching another.",
            )
        if live:
            readiness = check_local_live_ready(context.home, approval_mode=agent.state.approval)
            if not readiness.ready:
                raise APIError(
                    409,
                    "live_not_ready",
                    " ".join(readiness.missing)
                    + " Nothing was started; correct readiness and launch live again.",
                )
            if readiness.profile is not None and readiness.profile.state is ProfileState.DRIFTED:
                raise APIError(
                    409,
                    "profile_drifted",
                    "the broker profile is drifted. Nothing was started; run `tick broker status`, "
                    "then reconfirm each changed dependency before launching live.",
                )
        run_id = secrets.token_hex(12)
        current_boot = boot_id()
        ticket_path: Path | None = None
        fallback = False
        if live:
            try:
                ticket_path, fallback = create_launch_ticket(
                    context.home,
                    agent_id=agent_id,
                    run_id=run_id,
                    approval_mode=agent.state.approval,
                    standing_ok=standing_ok,
                    created_at=_aware(context.now()),
                    env=context.env,
                    current_boot_id=current_boot,
                )
            except (OSError, ValueError) as exc:
                raise APIError(
                    409,
                    "live_ticket_failed",
                    f"the one-use live ticket could not be created ({exc}). Nothing was started; "
                    "fix the runtime directory and try again.",
                ) from exc
        argv = _launch_argv(
            context,
            agent_id=agent_id,
            run_id=run_id,
            live=live,
            standing_ok=standing_ok,
            ticket_path=ticket_path,
        )
        try:
            pid = context.start_process(argv)
        except OSError as exc:
            if ticket_path is not None:
                try:
                    ticket_path.unlink()
                except FileNotFoundError:
                    pass
            raise APIError(
                500,
                "launch_failed",
                f"the run process could not start ({exc}). Nothing was launched; "
                "try again on the box.",
            ) from exc
        lease = RunLease(
            agent_id=agent_id,
            run_id=run_id,
            boot_id=current_boot,
            pid=pid,
            mode=Mode.LIVE if live else Mode.PAPER,
            approval=agent.state.approval,
            launch_source="api",
            started_at=_aware(context.now()),
            previous_run_id=existing.run_id if existing else None,
            previous_run_mode=existing.mode if existing else None,
            previous_run_boot_id=existing.boot_id if existing else None,
        )
        save_run_lease(context.home, lease)
        agent.ledger(clock=context.now).append(
            RecordKind.NOTE,
            {
                "event": "launch_requested",
                "run_id": run_id,
                "mode": "live" if live else "paper",
                **provenance,
                "at": _aware(context.now()),
            },
            source=DataSource.RUNTIME,
        )
        response = {
            "run_id": run_id,
            "status": "launched",
            "mode": "live" if live else "paper",
            "ticket_fallback": fallback,
        }
        _remember_launch(context, agent_id, key, response)
        return 202, response


def interview_start(context: ServeContext, body: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
    body, provenance = _mutation_body(body)
    if not set(body) <= {"provider", "kind", "model"} or set(body) < {"provider", "kind"}:
        raise APIError(
            400,
            "interview_start_invalid",
            "the body needs provider and kind, plus model only for anthropic. "
            "Correct it and retry.",
        )
    provider = body.get("provider")
    kind = body.get("kind")
    model = body.get("model")
    if provider not in {"codex", "anthropic"} or kind not in {"rule", "model"}:
        raise APIError(
            400,
            "interview_start_invalid",
            "provider must be codex or anthropic and kind must be rule or model. "
            "Correct the request.",
        )
    if provider == "anthropic" and (not isinstance(model, str) or not model.strip()):
        raise APIError(
            400,
            "interview_model_required",
            "anthropic requires the model you chose. Add model and start the interview again.",
        )
    if provider == "codex" and model is not None:
        raise APIError(
            400,
            "interview_model_forbidden",
            "codex resolves and reports its own model id. Send model: null and start the "
            "interview again.",
        )
    try:
        session = InterviewSession.create(
            context.home, provider=Provider(provider), kind=AgentKind(kind)
        )
    except (InterviewError, ValueError) as exc:
        raise _interview_error(exc) from exc
    if model is not None:
        write_private_file(session.directory / "api-model", str(model) + "\n")
    Ledger(session.directory / "records.jsonl", clock=context.now).append(
        RecordKind.NOTE,
        {"event": "interview_started", **provenance, "at": _aware(context.now())},
        source=DataSource.RUNTIME,
    )
    return 201, {"id": session.draft_id, "question": session.ask()}


def interview_answer(
    context: ServeContext, draft_id: str, body: Mapping[str, Any]
) -> tuple[int, dict[str, Any]]:
    body, provenance = _mutation_body(body)
    if set(body) != {"answer"} or not isinstance(body.get("answer"), str):
        raise APIError(400, "interview_answer_invalid", "the body must contain one text answer.")
    session = InterviewSession(context.home, draft_id)
    try:
        reply = _with_interview_model(session, lambda: session.answer(str(body["answer"])))
    except InterviewError as exc:
        raise _interview_error(exc) from exc
    Ledger(session.directory / "records.jsonl", clock=context.now).append(
        RecordKind.NOTE,
        {"event": "interview_answered", **provenance, "at": _aware(context.now())},
        source=DataSource.RUNTIME,
    )
    return 200, {"id": draft_id, "message": reply, "question": session.ask()}


def interview_accept(
    context: ServeContext, draft_id: str, body: Mapping[str, Any]
) -> tuple[int, dict[str, Any]]:
    body, provenance = _mutation_body(body)
    if body:
        raise APIError(
            400, "interview_accept_invalid", "accept takes only chat provenance. Retry it."
        )
    session = InterviewSession(context.home, draft_id)
    try:
        reply = session.accept()
    except InterviewError as exc:
        raise _interview_error(exc) from exc
    Ledger(session.directory / "records.jsonl", clock=context.now).append(
        RecordKind.NOTE,
        {"event": "interview_suggestion_accepted", **provenance, "at": _aware(context.now())},
        source=DataSource.RUNTIME,
    )
    return 200, {"id": draft_id, "message": reply, "question": session.ask()}


def interview_adopt(
    context: ServeContext, draft_id: str, body: Mapping[str, Any]
) -> tuple[int, dict[str, Any]]:
    body, provenance = _mutation_body(body)
    if not set(body) <= {"name", "max_cancels", "approval"} or set(body) < {"name", "max_cancels"}:
        raise APIError(
            400,
            "interview_adopt_invalid",
            "adopt needs name and max_cancels; provide them and retry.",
        )
    name = body.get("name")
    max_cancels = body.get("max_cancels")
    if not isinstance(name, str) or not name.strip() or not isinstance(max_cancels, int):
        raise APIError(
            400, "interview_adopt_invalid", "name must be text and max_cancels an integer."
        )
    session = InterviewSession(context.home, draft_id)
    try:
        draft = session.completed_draft
        spec = draft.to_agent().model_copy(update={"name": name.strip()})
        selected = draft.approval if "approval" not in body else ApprovalMode(body["approval"])
        run = AgentRun.create(
            context.home,
            spec,
            max_cancels_per_session=max_cancels,
            approval=selected,
            created_at=_aware(context.now()),
            instructions=draft.instructions,
        )
        run.ledger(clock=context.now).append(
            RecordKind.NOTE,
            {
                "event": "adopted",
                "draft_id": draft_id,
                "provenance": draft.provenance,
                "transcript_sha256": draft.transcript_sha256,
                "provider": draft.provider,
                "model_reported": draft.model_reported,
                **provenance,
            },
            source=DataSource.RUNTIME,
        )
    except (InterviewError, TickRuntimeError, ValueError) as exc:
        raise _interview_error(exc) from exc
    return 201, {"agent_id": run.agent_id, "name": spec.name}


def broker_profile(context: ServeContext) -> dict[str, Any]:
    profile, error = _profile(context)
    if error is not None:
        raise APIError(409, "broker_profile_invalid", error)
    try:
        proposal = load_proposal(context.home)
    except BrokerError:
        proposal = None
    profile_payload = profile.model_dump(mode="json") if profile is not None else None
    proposal_payload = proposal.model_dump(mode="json") if proposal is not None else None
    # Account identifiers are broker credentials for routing purposes.  The box
    # retains the exact value, while every phone-facing representation is masked.
    for payload in (profile_payload, proposal_payload):
        if payload is None:
            continue
        account_id = payload.pop("account_id", None)
        if isinstance(account_id, str):
            payload["account_id_masked"] = f"••••{account_id[-4:]}"
    return {"profile": profile_payload, "proposal": proposal_payload}


def _browser_viewport(body: Mapping[str, Any]) -> Viewport:
    viewport = body.get("viewport")
    if not isinstance(viewport, Mapping) or set(viewport) != {"width", "height"}:
        raise APIError(
            400,
            "BROWSER_VIEWPORT_INVALID",
            "viewport must contain integer width and height. Send the phone's visible frame "
            "size and retry.",
        )
    try:
        return Viewport(width=viewport["width"], height=viewport["height"])
    except (BrowserBridgeError, TypeError) as exc:
        reason = exc.reason if isinstance(exc, BrowserBridgeError) else str(exc)
        raise APIError(400, "BROWSER_VIEWPORT_INVALID", reason) from exc


def _browser_error(exc: BrowserBridgeError, *, status: int = 409) -> APIError:
    return APIError(status, exc.code, exc.reason)


def browser_session_start(
    context: ServeContext, body: Mapping[str, Any]
) -> tuple[int, dict[str, Any]]:
    if set(body) != {"url", "viewport", "purpose"} or not isinstance(body.get("url"), str):
        raise APIError(
            400,
            "BROWSER_SESSION_INVALID",
            "send url, viewport, and purpose from the active ceremony. Start that ceremony "
            "again if they are unavailable.",
        )
    purpose = body.get("purpose")
    if purpose not in {"broker_connect", "provider_login"}:
        raise APIError(
            400,
            "BROWSER_PURPOSE_INVALID",
            "purpose must be broker_connect or provider_login. Start that ceremony and retry.",
        )
    url = str(body["url"])
    active_url = context.browser_ceremony_url(str(purpose))
    requested_host = urlparse(url).hostname
    active_host = urlparse(active_url).hostname if active_url is not None else None
    if requested_host is None or requested_host != active_host:
        raise APIError(
            409,
            "BROWSER_URL_NOT_A_CEREMONY",
            "the URL host was not produced by this box's active ceremony. Start the provider "
            "or broker ceremony again and open the URL it returns.",
        )
    try:
        opened = context.browser_bridge.open(url, _browser_viewport(body), str(purpose))
    except BrowserBridgeError as exc:
        raise _browser_error(exc) from exc
    return 201, opened


def browser_frames(context: ServeContext, session_id: str) -> Iterable[dict[str, Any]]:
    if not context.browser_bridge.knows(session_id):
        raise APIError(
            404,
            "BROWSER_SESSION_NOT_FOUND",
            "that browser session is not active. Start the ceremony again if you still need "
            "its browser.",
        )

    def generate() -> Iterable[dict[str, Any]]:
        for t_ms, jpeg, origin in context.browser_bridge.frames(session_id):
            yield {
                "t": t_ms,
                "jpeg": base64.b64encode(jpeg).decode("ascii"),
                "origin": origin,
                "done": False,
            }
        yield {"done": True, "reason": context.browser_bridge.close_reason(session_id)}

    return generate()


def browser_input(
    context: ServeContext, session_id: str, body: Mapping[str, Any]
) -> tuple[int, dict[str, Any]]:
    if set(body) != {"events"} or not isinstance(body.get("events"), list):
        raise APIError(
            400,
            "BROWSER_EVENTS_REQUIRED",
            "events must be a list of browser inputs. Send the gesture again.",
        )
    try:
        context.browser_bridge.input(session_id, body["events"])
    except BrowserBridgeError as exc:
        raise _browser_error(
            exc, status=404 if exc.code == "BROWSER_SESSION_NOT_FOUND" else 400
        ) from exc
    return 202, {"accepted": len(body["events"])}


def browser_close(context: ServeContext, session_id: str) -> tuple[int, dict[str, Any]]:
    try:
        context.browser_bridge.close(session_id, "user_closed")
    except BrowserBridgeError as exc:
        raise _browser_error(exc, status=404) from exc
    return 200, {"session_id": session_id, "reason": "user_closed"}


def provider_browser_login_start(
    context: ServeContext, body: Mapping[str, Any]
) -> tuple[int, dict[str, Any]]:
    if set(body) != {"viewport"}:
        raise APIError(
            400,
            "BROWSER_VIEWPORT_INVALID",
            "browser login requires the phone's viewport. Send its visible frame size and retry.",
        )
    from .provider_login import ProviderLoginError

    try:
        result = dict(context.provider_browser_login_start(_browser_viewport(body)))
    except ProviderLoginError as exc:
        raise APIError(409, exc.code, exc.reason) from exc
    _record_api_mutation(context, "provider", "provider_browser_login_started")
    return 202, result


def broker_connect_start(
    context: ServeContext, body: Mapping[str, Any]
) -> tuple[int, dict[str, Any]]:
    if not set(body) <= {"server_url", "redirect_scheme"} or (
        "server_url" in body and not isinstance(body["server_url"], str)
    ):
        raise APIError(
            400,
            "broker_connect_invalid",
            "server_url, when supplied, must be text. Correct it and start again.",
        )
    scheme = body.get("redirect_scheme")
    if scheme is not None and not (
        isinstance(scheme, str) and re.fullmatch(r"[a-z][a-z0-9+.-]{1,31}", scheme)
    ):
        raise APIError(
            400,
            "broker_connect_invalid",
            "redirect_scheme, when supplied, must be a lowercase URL scheme like tick. "
            "Correct it and start again.",
        )
    from .broker_connect import BrokerConnectError

    try:
        result = dict(context.broker_connect_start(body.get("server_url"), scheme))
    except BrokerConnectError as exc:
        raise APIError(409, exc.code.lower(), exc.reason) from exc
    _record_api_mutation(context, "broker", "broker_connection_started")
    return 202, result


def broker_connect_complete(
    context: ServeContext, connect_id: str, body: Mapping[str, Any]
) -> tuple[int, dict[str, Any]]:
    if set(body) != {"redirect_url"} or not isinstance(body.get("redirect_url"), str):
        raise APIError(
            400,
            "broker_redirect_invalid",
            "post the complete redirect_url intercepted by the phone, then retry.",
        )
    from .broker_connect import BrokerConnectError

    try:
        result = dict(context.broker_connect_complete(connect_id, str(body["redirect_url"])))
    except BrokerConnectError as exc:
        raise APIError(409, exc.code.lower(), exc.reason) from exc
    _record_api_mutation(context, "broker", "broker_connection_completed")
    return 202, result


def broker_connect_status(context: ServeContext, connect_id: str) -> dict[str, Any]:
    from .broker_connect import BrokerConnectError

    try:
        return dict(context.broker_connect_status(connect_id))
    except BrokerConnectError as exc:
        raise APIError(404, exc.code.lower(), exc.reason) from exc


def _broker_profile_operation(
    context: ServeContext, action: str, body: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        return dict(context.broker_profile_operation(action, body))
    except (BrokerError, ModelAgentError, ValueError, OSError) as exc:
        raise APIError(
            409,
            f"broker_{action.split(':', 1)[0]}_refused",
            f"{exc} Nothing gained authority; correct it and retry.",
        ) from exc


def broker_propose(context: ServeContext, body: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
    result = _broker_profile_operation(context, "propose", body)
    return 202, result


def broker_proposal_edit(
    context: ServeContext, tool: str, body: Mapping[str, Any]
) -> tuple[int, dict[str, Any]]:
    return 200, _broker_profile_operation(context, f"edit:{tool}", body)


def broker_accounts(context: ServeContext) -> tuple[int, dict[str, Any]]:
    return 200, _broker_profile_operation(context, "accounts", {})


def broker_account_select(
    context: ServeContext, body: Mapping[str, Any]
) -> tuple[int, dict[str, Any]]:
    return 200, _broker_profile_operation(context, "account", body)


def broker_prove(context: ServeContext, body: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
    return 200, _broker_profile_operation(context, "prove", body)


def broker_profile_diff(context: ServeContext) -> dict[str, Any]:
    return _broker_profile_operation(context, "diff", {})


def broker_disconnect(context: ServeContext) -> tuple[int, dict[str, Any]]:
    storage = FileTokenStorage(context.home)
    removed = storage.forget()
    Ledger(context.home / "broker" / "records.jsonl", clock=context.now).append(
        RecordKind.NOTE,
        {
            "event": "broker_disconnected",
            "removed": [path.name for path in removed],
            "via": "api",
            "at": _aware(context.now()),
        },
        source=DataSource.RUNTIME,
    )
    return 200, {
        "removed": [path.name for path in removed],
        "reason": (
            "the local grant copy was removed. Revoke it at Robinhood too; the app can "
            "still use paper agents."
        ),
    }


def broker_confirm(context: ServeContext, body: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
    body, provenance = _mutation_body(body)
    allowed = {"tool", "confirm", "fixed"}
    if (
        not set(body) <= allowed
        or set(body) < {"tool", "confirm"}
        or body.get("confirm") is not True
    ):
        raise APIError(
            400,
            "broker_confirm_invalid",
            "confirm one tool with {tool, confirm: true, fixed?: {...}}; denied tools are "
            "never confirmable.",
        )
    tool = body.get("tool")
    if not isinstance(tool, str) or not tool.strip():
        raise APIError(400, "broker_confirm_invalid", "tool must name one proposed tool.")
    fixed = body.get("fixed", {})
    if not isinstance(fixed, dict) or any(not isinstance(name, str) for name in fixed):
        raise APIError(
            400,
            "broker_confirm_invalid",
            "fixed must be an object keyed by exact broker argument names. Correct it and retry.",
        )
    try:
        proposal = load_proposal(context.home)
    except (OSError, ValueError) as exc:
        raise APIError(
            409,
            "broker_profile_missing",
            f"the reviewed broker proposal is unavailable ({exc}). Propose and review one first.",
        ) from exc
    proposed = proposal.tools.get(tool)
    if proposed is None:
        raise APIError(
            404, "broker_tool_not_found", f"tool {tool!r} is not in the proposal. Refresh it."
        )
    if proposed.category is None:
        raise APIError(
            409,
            "broker_tool_unmapped",
            f"tool {tool!r} has no deterministic category. Review it on the box first.",
        )
    if proposed.category.denied:
        raise APIError(
            403,
            "broker_tool_denied",
            f"tool {tool!r} is deterministically denied and cannot be confirmed. "
            "Choose a callable tool.",
        )
    if proposal.account_id is None and proposed.category is not Category.READ_ACCOUNTS:
        raise APIError(
            409,
            "broker_account_required",
            "Finalize read.accounts and choose an eligible masked account before finalizing "
            "another tool.",
        )
    existing = load_profile(context.home)
    confirmed: dict[str, ProfileTool] = {}
    for name, candidate in proposal.tools.items():
        category = candidate.category
        if category is not None and category.denied:
            confirmed[name] = ProfileTool(
                category=category,
                contract=candidate.contract,
                arguments={},
                result={},
                confirmed_contract_hash=None,
                mapping_hash=mapping_hash(category, {}, {}),
                confirmed_at=None,
                confirmed_by=None,
                categorizer_version=proposal.categorizer_version,
                proved_contract_hash=None,
                proved_mapping_hash=None,
                proved_at=None,
                proof=None,
            )
            continue
        prior = existing.tools.get(name) if existing is not None else None
        if name != tool:
            if (
                prior is not None
                and category is prior.category
                and candidate.contract.contract_hash == prior.contract.contract_hash
            ):
                confirmed[name] = prior
            continue
        assert category is not None
        arguments = dict(candidate.arguments) | dict(fixed)
        try:
            confirmed[name] = ProfileTool(
                category=category,
                contract=candidate.contract,
                arguments=arguments,
                result=candidate.result,
                confirmed_contract_hash=candidate.contract.contract_hash,
                mapping_hash=mapping_hash(category, arguments, candidate.result),
                confirmed_at=_aware(context.now()),
                confirmed_by="api",
                categorizer_version=proposal.categorizer_version,
                proved_contract_hash=None,
                proved_mapping_hash=None,
                proved_at=None,
                proof=None,
            )
        except ValueError as exc:
            raise APIError(
                409,
                "broker_mapping_incomplete",
                f"tool {tool!r} cannot be confirmed ({exc}). Supply every fixed value and retry.",
            ) from exc
    try:
        profile = build_profile(
            server=proposal.server,
            account_id=proposal.account_id,
            tools=confirmed,
            inventory_hash=proposal.inventory_hash,
            data_class="display_only",
            sanction=proposal.sanction,
            profile_format_version=PROFILE_FORMAT_VERSION,
            canonicalizer_version=CANONICALIZER_VERSION,
            category_registry_version=CATEGORY_REGISTRY_VERSION,
            state=ProfileState.CONFIRMED,
            observed_inventory_hash=proposal.inventory_hash,
            drift=(),
        )
        confirm_profile(
            context.home,
            profile,
            actor="box-api",
            at=_aware(context.now()),
            via=provenance["via"],
            transcript_hash=provenance.get("transcript_hash"),
        )
    except (OSError, BrokerError, RecordError, ValueError) as exc:
        raise APIError(
            409,
            "broker_confirm_failed",
            f"the confirmation could not be recorded ({exc}). Nothing gained "
            "authority; retry on the box.",
        ) from exc
    return 200, {"tool": tool, "confirmed": True, "profile_hash": profile.profile_hash}


def pair_rotate(context: ServeContext) -> tuple[int, dict[str, Any]]:
    path, secret = rotate_secret(context.home)
    _record_api_mutation(context, "pairing", "pairing_secret_rotated")
    return 200, {"secret": secret, "path": str(path)}


def pair_recover(context: ServeContext, body: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
    """Rotate only after the droplet itself observes the app's ownership tag."""
    if set(body) != {"nonce"} or not isinstance(body.get("nonce"), str):
        raise APIError(
            400,
            "recovery_nonce_invalid",
            "the body must contain one recovery nonce. Start recovery again from the app.",
        )
    try:
        expected = recovery_tag(str(body["nonce"]))
        tags = context.metadata.tags()
    except (ValueError, RuntimeError) as exc:
        raise APIError(409, "recovery_proof_unavailable", str(exc)) from exc
    if expected not in tags:
        raise APIError(
            403,
            "recovery_tag_missing",
            "the droplet does not carry this recovery tag. Sign in to Digital Ocean, tag "
            "this droplet from the app, and retry.",
        )
    path, secret = rotate_secret(context.home)
    from tick.tunnel import endpoint_id_for_pairing_secret

    endpoint_id = endpoint_id_for_pairing_secret(secret)
    _record_api_mutation(context, "pairing", "pairing_secret_recovered")
    return 200, {"secret": secret, "path": str(path), "endpoint_id": endpoint_id}


def _agent(context: ServeContext, agent_id: str) -> AgentRun:
    try:
        return AgentRun.load(context.home, agent_id)
    except (TickRuntimeError, SpecError) as exc:
        raise APIError(404, "agent_not_found", str(exc)) from exc


def _profile(context: ServeContext):
    try:
        return load_profile(context.home), None
    except BrokerError as exc:
        return None, str(exc)


def _queue_holding(context: ServeContext, approval_id: str) -> tuple[str, ApprovalQueue]:
    for agent_id in AgentRun.list_ids(context.home):
        queue = ApprovalQueue.system(context.home, agent_id)
        try:
            queue.get(approval_id)
        except ApprovalError as exc:
            if exc.code == "approval_not_found":
                continue
            raise APIError(exc.status, exc.code, exc.reason) from exc
        return agent_id, queue
    raise APIError(
        404, "approval_not_found", f"approval {approval_id} does not exist. Refresh the list."
    )


def _launch_argv(
    context: ServeContext,
    *,
    agent_id: str,
    run_id: str,
    live: bool,
    standing_ok: bool,
    ticket_path: Path | None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "tick",
        "run",
        agent_id,
        "--run-id",
        run_id,
        "--launch-source",
        "api",
    ]
    if live:
        assert ticket_path is not None
        command.extend(["--live", "--launch-ticket", str(ticket_path)])
        if standing_ok:
            command.append("--live-standing-ok")
    if AgentRun.load(context.home, agent_id).state.approval is ApprovalMode.EACH:
        command.extend(["--approval-window", "300s"])
    supervisor = context.env.get("TICK_SUPERVISOR")
    if supervisor == "systemd":
        unit = f"tick-live-{agent_id}-{run_id}" if live else f"tick-agent-{agent_id}-{run_id}"
        return [
            "systemd-run",
            "--collect",
            "-p",
            "Restart=no",
            "--unit",
            unit,
            *command,
        ]
    return command


def _idempotency_path(context: ServeContext, agent_id: str) -> Path:
    return context.home / "agents" / agent_id / "launch-requests.json"


def _idempotent_launch(context: ServeContext, agent_id: str, key: Any) -> dict[str, Any] | None:
    if key is None:
        return None
    try:
        payload = json.loads(_idempotency_path(context, agent_id).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = payload.get(key)
    return dict(value) if isinstance(value, dict) else None


def _remember_launch(
    context: ServeContext, agent_id: str, key: Any, response: Mapping[str, Any]
) -> None:
    if key is None:
        return
    path = _idempotency_path(context, agent_id)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    payload[str(key)] = dict(response)
    write_private_file(path, json.dumps(normalize_payload(payload), sort_keys=True) + "\n")


def _with_interview_model(session: InterviewSession, call: Callable[[], str]) -> str:
    path = session.directory / "api-model"
    try:
        model = path.read_text(encoding="utf-8").strip()
    except OSError:
        return call()
    with _INTERVIEW_ENV_LOCK:
        previous = os.environ.get("TICK_INTERVIEW_MODEL")
        os.environ["TICK_INTERVIEW_MODEL"] = model
        try:
            return call()
        finally:
            if previous is None:
                os.environ.pop("TICK_INTERVIEW_MODEL", None)
            else:
                os.environ["TICK_INTERVIEW_MODEL"] = previous


def _interview_error(exc: Exception) -> APIError:
    if isinstance(exc, InterviewError):
        return APIError(409, exc.code.lower(), exc.reason)
    return APIError(400, "interview_invalid", f"{exc} Correct the interview request and retry.")


def _chat_error(exc: Exception) -> APIError:
    if isinstance(exc, ChatError):
        status = 404 if exc.code == "CHAT_NOT_FOUND" else 409
        return APIError(status, exc.code.lower(), exc.reason)
    return APIError(400, "chat_invalid", f"{exc} Correct the chat request and retry.")


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("the server clock must return a timezone-aware datetime")
    return value


def _observed_before(value: str | None) -> datetime | None:
    """Decode an optional release cutoff without treating a naive clock as evidence."""
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise APIError(
            400,
            "observed_before_invalid",
            "observed_before must be an ISO datetime with a timezone. Correct it and retry.",
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise APIError(
            400,
            "observed_before_invalid",
            "observed_before must include Z or an explicit offset. Correct it and retry.",
        )
    return parsed
