"""Closed structural tool descriptions shared by provider chat adapters."""

from __future__ import annotations

from typing import Any

from tick.agents import ModelAgentSpec
from tick.broker.profile_model import proposal_document_schema
from tick.chat.setup import SetupScope
from tick.runtime import ApprovalMode
from tick.spec import StrategySpec

__all__ = ["chat_tool_definitions", "setup_tool_definitions"]


def chat_tool_definitions() -> tuple[dict[str, Any], ...]:
    """Return schemas only; none contains strategy text or inferred parameters."""
    reads = {
        "status": {},
        "agents": {},
        "agent_document": {"agent_id": {"type": "string"}},
        "ledger": {
            "agent_id": {"type": "string"},
            "after": {"type": "integer", "minimum": 0},
        },
        "approvals": {},
        "broker_profile": {},
        "commons_pass": {"ticker": {"type": "string"}},
        "doctor": {},
    }
    proposals = {
        "propose_launch": {
            "agent_id": {"type": "string"},
            "live": {"type": "boolean"},
            "standing_ok": {"type": "boolean"},
            "transcript_hash": {"type": "string"},
        },
        "propose_stop": {
            "agent_id": {"type": "string"},
            "transcript_hash": {"type": "string"},
        },
        "propose_approval_decision": {
            "agent_id": {"type": "string"},
            "approval_id": {"type": "string"},
            "decision": {"type": "string", "enum": ["approve", "decline"]},
            "transcript_hash": {"type": "string"},
        },
        "propose_adopt_draft": {
            "draft_id": {"type": "string"},
            "name": {"type": "string"},
            "max_cancels": {"type": "integer", "minimum": 0},
            "approval": {"type": "string", "enum": ["each", "standing"]},
            "transcript_hash": {"type": "string"},
        },
        "propose_instructions_change": {
            "agent_id": {"type": "string"},
            "instructions": {"type": "string"},
            "transcript_hash": {"type": "string"},
        },
        "start_interview": {
            "provider": {"type": "string", "enum": ["codex", "anthropic"]},
            "kind": {"type": "string", "enum": ["rule", "model"]},
            "model": {"type": ["string", "null"]},
            "transcript_hash": {"type": "string"},
        },
        "interview_answer": {
            "draft_id": {"type": "string"},
            "answer": {"type": "string"},
            "transcript_hash": {"type": "string"},
        },
    }
    values = []
    for name, properties in {**reads, **proposals}.items():
        values.append(
            {
                "name": name,
                "description": (
                    "Read box state."
                    if name in reads
                    else "Record a proposal for separate user confirmation; execute nothing."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": properties,
                    "required": list(properties),
                    "additionalProperties": False,
                },
            }
        )
    return tuple(values)


def setup_tool_definitions(scope: SetupScope) -> tuple[dict[str, Any], ...]:
    """Return only the tools that can contribute to one setup document."""
    selected = SetupScope(scope)
    if selected is SetupScope.BROKER_PROFILE:
        schemas = {
            "broker_inventory": _object({}),
            "broker_draft": _object({}),
            "propose_broker_profile": _object(
                {"document": proposal_document_schema()}, required=("document",)
            ),
            "prove_broker_draft": _object(
                {
                    "probe": {
                        "type": "object",
                        "additionalProperties": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "integer"},
                                {"type": "boolean"},
                            ]
                        },
                    }
                },
                required=("probe",),
            ),
            "broker_accounts": _object({}),
        }
    else:
        document = _object(
            {
                "spec": {
                    "anyOf": [
                        StrategySpec.model_json_schema(),
                        ModelAgentSpec.model_json_schema(),
                    ]
                },
                "instructions": {"type": ["string", "null"]},
                "approval": {"type": "string", "enum": [item.value for item in ApprovalMode]},
                "provenance": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
                "model_reported": {"type": "string"},
            },
            required=("spec", "instructions", "approval", "provenance", "model_reported"),
        )
        schemas = {
            "interview_script": _object({}),
            "agent_draft": _object({}),
            "propose_agent_draft": _object({"document": document}, required=("document",)),
            "commons_pass": _object({"ticker": {"type": "string"}}, required=("ticker",)),
        }
    return tuple(
        {
            "name": name,
            "description": (
                "Read deterministic setup state."
                if not name.startswith(("propose_", "prove_"))
                else "Check and record setup state without finalizing or adopting it."
            ),
            "input_schema": schema,
        }
        for name, schema in schemas.items()
    )


def _object(properties: dict[str, Any], *, required: tuple[str, ...] = ()) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }
