"""Closed structural tool descriptions shared by provider chat adapters."""

from __future__ import annotations

from typing import Any

__all__ = ["chat_tool_definitions"]


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
