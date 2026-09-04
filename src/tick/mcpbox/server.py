"""MCP reads box state; every action-shaped tool records only a proposal."""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from tick.chat import SetupChatSession, SetupScope
from tick.interview import SLOTS, InterviewError, InterviewSession
from tick.records import DataSource, Ledger, RecordKind
from tick.runtime import AgentRun
from tick.serve import handlers
from tick.serve.handlers import ServeContext

__all__ = ["BoxTools", "build_server", "run_stdio"]


class BoxTools:
    """Tool functions kept callable without starting an MCP transport in tests."""

    def __init__(self, context: ServeContext, *, setup_session_id: str | None) -> None:
        self.context = context
        self.setup_session_id = setup_session_id

    def status(self) -> dict[str, Any]:
        return handlers.status(self.context)

    def agents(self) -> dict[str, Any]:
        return handlers.agents(self.context)

    def agent_document(self, agent_id: str) -> dict[str, Any]:
        run = AgentRun.load(self.context.home, agent_id)
        return {
            "agent": run.spec.model_dump(mode="json"),
            "instructions": run.instructions() if run.instructions_path.exists() else None,
            "approval": run.state.approval.value,
        }

    def ledger(self, agent_id: str, after: int) -> dict[str, Any]:
        return handlers.ledger(self.context, agent_id, after)

    def approvals(self) -> dict[str, Any]:
        return handlers.approvals(self.context)

    def broker_profile(self) -> dict[str, Any]:
        return handlers.broker_profile(self.context)

    def doctor(self) -> dict[str, Any]:
        return handlers.doctor(self.context)

    def commons_pass(self, ticker: str) -> dict[str, Any]:
        return handlers.commons_pass(self.context, ticker)

    def broker_inventory(self) -> dict[str, Any]:
        self._require_scope(SetupScope.BROKER_PROFILE)
        return dict(self.context.broker_profile_operation("inventory", {}))

    def broker_draft(self) -> dict[str, Any]:
        setup = self._require_scope(SetupScope.BROKER_PROFILE)
        return {**handlers.broker_profile(self.context), "valid": setup.state.valid}

    def propose_broker_profile(self, document: dict[str, Any]) -> dict[str, Any]:
        setup = self._require_scope(SetupScope.BROKER_PROFILE)
        try:
            result = dict(
                self.context.broker_profile_operation("propose_document", {"document": document})
            )
        except Exception as exc:  # noqa: BLE001 - deterministic refusal becomes a tool result
            verdict = {
                "code": "BROKER_DOCUMENT_INVALID",
                "reason": (
                    f"{exc} No tool gained authority; correct the document and propose it again."
                ),
                "evidence": ["checked"],
            }
            setup.save(
                document=document,
                valid=False,
                verdict=verdict,
                at=self.context.now(),
            )
            return {"valid": False, **verdict}
        proposal = dict(result["proposal"])
        verdict = {
            "code": "BROKER_DOCUMENT_VALID",
            "reason": (
                "The document matches the broker proposal schema. Warnings remain advisory; "
                "the person can finalize tools individually."
            ),
            "warnings": result.get("warnings", {}),
            "denied": result.get("denied", []),
            "evidence": ["checked"],
        }
        setup.save(document=proposal, valid=True, verdict=verdict, at=self.context.now())
        return {"valid": True, "document": proposal, **verdict}

    def prove_broker_draft(self, probe: dict[str, Any]) -> dict[str, Any]:
        setup = self._require_scope(SetupScope.BROKER_PROFILE)
        try:
            result = dict(self.context.broker_profile_operation("prove", {"probe": probe}))
        except Exception as exc:  # noqa: BLE001 - proof refusal must become conversation
            verdict = {
                "code": "BROKER_PROOF_REFUSED",
                "reason": f"{exc} Fix the named mapping or probe value, then prove it again.",
                "evidence": ["checked"],
            }
            setup.save(
                document=setup.state.document,
                valid=False,
                verdict=verdict,
                at=self.context.now(),
            )
            return {"valid": False, **verdict}
        outcomes = result.get("outcome")
        passed = (
            isinstance(outcomes, dict)
            and bool(outcomes)
            and all(
                isinstance(value, dict) and value.get("success") is True
                for value in outcomes.values()
            )
        )
        verdict = {
            "code": "BROKER_PROOF_VALID" if passed else "BROKER_PROOF_FAILED",
            "reason": (
                "Every finalized mapping proved; the person can continue reviewing."
                if passed
                else "At least one finalized mapping did not prove. Fix the named mapping or "
                "probe value, then prove it again."
            ),
            "outcome": outcomes if isinstance(outcomes, dict) else {},
            "evidence": ["checked"],
        }
        setup.save(
            document=setup.state.document,
            valid=passed,
            verdict=verdict,
            at=self.context.now(),
        )
        return {"valid": passed, **verdict}

    def broker_accounts(self) -> dict[str, Any]:
        self._require_scope(SetupScope.BROKER_PROFILE)
        result = dict(self.context.broker_profile_operation("accounts", {}))
        return {**result, "evidence": ["display_only"]}

    def interview_script(self) -> dict[str, Any]:
        self._require_scope(SetupScope.AGENT_DRAFT)
        return {
            "slots": [
                {"name": slot.name, "question": slot.question, "type": slot.type} for slot in SLOTS
            ],
            "evidence": ["checked"],
        }

    def agent_draft(self) -> dict[str, Any]:
        setup = self._require_scope(SetupScope.AGENT_DRAFT)
        return {
            "document": setup.state.document,
            "valid": setup.state.valid,
            "open_questions": setup.state.verdict.get("open_questions", []),
            "verdict": setup.state.verdict,
            "evidence": ["checked"],
        }

    def propose_agent_draft(self, document: dict[str, Any]) -> dict[str, Any]:
        setup = self._require_scope(SetupScope.AGENT_DRAFT)
        chat = setup.chat
        candidate = {"draft_id": chat.session_id, **document}
        try:
            InterviewSession.from_chat_document(
                self.context.home,
                draft_id=chat.session_id,
                provider=chat.metadata["provider"],
                document=document,
                transcript=chat.transcript_path.read_bytes(),
            )
        except (InterviewError, OSError, ValueError) as exc:
            code = exc.code if isinstance(exc, InterviewError) else "DRAFT_DOCUMENT_INVALID"
            reason = exc.reason if isinstance(exc, InterviewError) else str(exc)
            verdict = {
                "code": code,
                "reason": f"{reason} The person can answer the missing questions and try again.",
                "open_questions": _open_questions(document),
                "evidence": ["checked"],
            }
            setup.save(
                document=candidate,
                valid=False,
                verdict=verdict,
                at=self.context.now(),
            )
            return {"valid": False, "document": candidate, **verdict}
        verdict = {
            "code": "AGENT_DOCUMENT_VALID",
            "reason": "Every meaning-bearing field has provenance. The person can adopt the draft.",
            "open_questions": [],
            "evidence": ["checked"],
        }
        setup.save(document=candidate, valid=True, verdict=verdict, at=self.context.now())
        return {"valid": True, "document": candidate, **verdict}

    def _require_scope(self, expected: SetupScope) -> SetupChatSession:
        if self.setup_session_id is None:
            raise ValueError(
                "this setup tool has no setup session. Open it through /v1/setup/chat."
            )
        setup = SetupChatSession(self.context.home, self.setup_session_id)
        if setup.state.scope is not expected:
            raise ValueError(
                f"this chat is scoped to {setup.state.scope.value}, not {expected.value}. "
                "Use a tool exposed by this setup chat."
            )
        return setup

    def proposal(
        self,
        action: str,
        *,
        agent_id: str | None,
        arguments: dict[str, Any],
        transcript_hash: str,
    ) -> dict[str, Any]:
        """Append proposal evidence and make no requested state change."""
        now = self.context.now()
        proposal_id = secrets.token_hex(12)
        wire_arguments = ({"agent_id": agent_id} if agent_id is not None else {}) | arguments
        payload = {
            "event": "proposal",
            "proposal_id": proposal_id,
            "action": action,
            "arguments": wire_arguments,
            "via": "chat",
            "transcript_hash": transcript_hash,
            "at": now,
        }
        path = (
            AgentRun.load(self.context.home, agent_id).ledger_path
            if agent_id is not None
            else self.context.home / "chat" / "proposals.jsonl"
        )
        Ledger(path, clock=lambda: now).append(RecordKind.NOTE, payload, source=DataSource.RUNTIME)
        return {
            "proposal_id": proposal_id,
            "action": action,
            "arguments": wire_arguments,
            "transcript_hash": transcript_hash,
            "executed": False,
        }

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Dispatch the closed Anthropic tool loop through these same functions."""
        if self.setup_session_id is not None:
            setup = self._require_scope(
                SetupChatSession(self.context.home, self.setup_session_id).state.scope
            )
            scoped = {
                SetupScope.BROKER_PROFILE: {
                    "broker_inventory": self.broker_inventory,
                    "broker_draft": self.broker_draft,
                    "propose_broker_profile": self.propose_broker_profile,
                    "prove_broker_draft": self.prove_broker_draft,
                    "broker_accounts": self.broker_accounts,
                },
                SetupScope.AGENT_DRAFT: {
                    "interview_script": self.interview_script,
                    "agent_draft": self.agent_draft,
                    "propose_agent_draft": self.propose_agent_draft,
                    "commons_pass": self.commons_pass,
                },
            }[setup.state.scope]
            function = scoped.get(name)
            if function is None:
                raise ValueError(
                    f"tool {name!r} is outside this {setup.state.scope.value} setup chat. "
                    "Use one of its advertised tools."
                )
            return function(**arguments)
        reads = {
            "status": self.status,
            "agents": self.agents,
            "agent_document": self.agent_document,
            "ledger": self.ledger,
            "approvals": self.approvals,
            "broker_profile": self.broker_profile,
            "commons_pass": self.commons_pass,
            "doctor": self.doctor,
        }
        if name in reads:
            return reads[name](**arguments)
        actions = {
            "propose_launch": "launch",
            "propose_stop": "stop",
            "propose_approval_decision": "approval_decision",
            "propose_adopt_draft": "adopt_draft",
            "propose_instructions_change": "instructions_change",
            "start_interview": "start_interview",
            "interview_answer": "interview_answer",
        }
        action = actions.get(name)
        if action is None:
            raise ValueError(f"unknown Tick box tool {name!r}; refresh the tool definitions.")
        values = dict(arguments)
        transcript_hash = str(values.pop("transcript_hash"))
        agent_id = values.pop("agent_id", None)
        return self.proposal(
            action,
            agent_id=str(agent_id) if agent_id is not None else None,
            arguments=values,
            transcript_hash=transcript_hash,
        )


def build_server(tools: BoxTools) -> MCPServer:
    server = MCPServer(name="tick-box", version="1")
    if tools.setup_session_id is not None:
        scope = SetupChatSession(tools.context.home, tools.setup_session_id).state.scope
        functions = (
            (
                tools.broker_inventory,
                tools.broker_draft,
                tools.propose_broker_profile,
                tools.prove_broker_draft,
                tools.broker_accounts,
            )
            if scope is SetupScope.BROKER_PROFILE
            else (
                tools.interview_script,
                tools.agent_draft,
                tools.propose_agent_draft,
                tools.commons_pass,
            )
        )
        for function in functions:
            server.tool()(function)
        return server
    server.tool()(tools.status)
    server.tool()(tools.agents)
    server.tool()(tools.agent_document)
    server.tool()(tools.ledger)
    server.tool()(tools.approvals)
    server.tool()(tools.broker_profile)
    server.tool()(tools.doctor)
    server.tool()(tools.commons_pass)

    def propose_launch(
        agent_id: str, live: bool, standing_ok: bool, transcript_hash: str
    ) -> dict[str, Any]:
        return tools.proposal(
            "launch",
            agent_id=agent_id,
            arguments={"live": live, "standing_ok": standing_ok},
            transcript_hash=transcript_hash,
        )

    def propose_stop(agent_id: str, transcript_hash: str) -> dict[str, Any]:
        return tools.proposal(
            "stop", agent_id=agent_id, arguments={}, transcript_hash=transcript_hash
        )

    def propose_approval_decision(
        agent_id: str, approval_id: str, decision: str, transcript_hash: str
    ) -> dict[str, Any]:
        return tools.proposal(
            "approval_decision",
            agent_id=agent_id,
            arguments={"approval_id": approval_id, "decision": decision},
            transcript_hash=transcript_hash,
        )

    def propose_adopt_draft(
        draft_id: str,
        name: str,
        max_cancels: int,
        approval: str,
        transcript_hash: str,
    ) -> dict[str, Any]:
        return tools.proposal(
            "adopt_draft",
            agent_id=None,
            arguments={
                "draft_id": draft_id,
                "name": name,
                "max_cancels": max_cancels,
                "approval": approval,
            },
            transcript_hash=transcript_hash,
        )

    def propose_instructions_change(
        agent_id: str, instructions: str, transcript_hash: str
    ) -> dict[str, Any]:
        return tools.proposal(
            "instructions_change",
            agent_id=agent_id,
            arguments={"instructions": instructions},
            transcript_hash=transcript_hash,
        )

    def start_interview(
        provider: str, kind: str, model: str | None, transcript_hash: str
    ) -> dict[str, Any]:
        return tools.proposal(
            "start_interview",
            agent_id=None,
            arguments={"provider": provider, "kind": kind, "model": model},
            transcript_hash=transcript_hash,
        )

    def interview_answer(draft_id: str, answer: str, transcript_hash: str) -> dict[str, Any]:
        return tools.proposal(
            "interview_answer",
            agent_id=None,
            arguments={"draft_id": draft_id, "answer": answer},
            transcript_hash=transcript_hash,
        )

    for function in (
        propose_launch,
        propose_stop,
        propose_approval_decision,
        propose_adopt_draft,
        propose_instructions_change,
        start_interview,
        interview_answer,
    ):
        server.tool()(function)
    return server


def run_stdio(home: Path, context: ServeContext, *, setup_session_id: str | None) -> None:
    del home
    build_server(BoxTools(context, setup_session_id=setup_session_id)).run("stdio")


def _open_questions(document: dict[str, Any]) -> list[str]:
    provenance = document.get("provenance")
    if not isinstance(provenance, dict):
        return ["provenance"]
    spec = document.get("spec")
    if not isinstance(spec, dict):
        return ["spec"]
    from tick.agents import ModelAgentSpec
    from tick.interview import meaning_bearing_fields
    from tick.spec import StrategySpec

    try:
        parsed = (
            ModelAgentSpec.model_validate(spec)
            if spec.get("kind") == "model_agent"
            else StrategySpec.model_validate(spec)
        )
    except ValueError:
        return ["spec"]
    return sorted(meaning_bearing_fields(parsed) - provenance.keys())
