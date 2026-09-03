"""MCP reads box state; every action-shaped tool records only a proposal."""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from tick.records import DataSource, Ledger, RecordKind
from tick.runtime import AgentRun
from tick.serve import handlers
from tick.serve.handlers import ServeContext

__all__ = ["BoxTools", "build_server", "run_stdio"]


class BoxTools:
    """Tool functions kept callable without starting an MCP transport in tests."""

    def __init__(self, context: ServeContext) -> None:
        self.context = context

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


def run_stdio(home: Path, context: ServeContext) -> None:
    del home
    build_server(BoxTools(context)).run("stdio")
