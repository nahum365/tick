"""Render an interview draft from its document, never the model's prose."""

from __future__ import annotations

from tick.agents import ModelAgentSpec
from tick.compile.explain import explain as explain_rules

from .draft import Draft

__all__ = ["explain"]


def explain(draft: Draft) -> tuple[str, ...]:
    """Plain facts a person can compare with the document before adoption."""
    spec = draft.spec
    lines = [
        f"Draft {draft.draft_id}: {len(spec.universe)} symbol(s), cadence {spec.cadence.kind}.",
        (
            "Cage: maximum position "
            f"{spec.cage.max_position_pct}%, maximum positions {spec.cage.max_positions}, "
            f"maximum order ${spec.cage.max_order_notional}, maximum daily drawdown "
            f"{spec.cage.max_daily_drawdown_pct}%, session {spec.cage.allowed_session}."
        ),
        f"Approval: {draft.approval.value}.",
    ]
    if isinstance(spec, ModelAgentSpec):
        lines.insert(
            1,
            f"Model-driven agent: {spec.model} through {spec.provider}; instructions are present.",
        )
    else:
        for item in explain_rules(spec):
            lines.append(f"{item.rule_id}: {item.what_it_does}")
            lines.extend(f"It cannot know {blind}" for blind in item.what_it_cannot_know)
    return tuple(lines)
