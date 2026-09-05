"""Model-driven agents — judgment from a model, authority from the runtime.

    from tick.agents import ModelAgent, ModelAgentSpec, load_agent_spec_file

    spec = load_agent_spec_file("my-agent.json")        # either kind
    agent = ModelAgent(spec, client=client, instructions=instructions)

A model agent's document is a universe, a cadence, a cage and a model id
(`spec.py`). Its *instructions* are a file the user writes, under
`TICK_HOME/agents/<agent id>/instructions.md`, and the agent refuses to run
without one. Per tick it is handed this tick's snapshot and answers with order
intents in a strict schema; every intent then goes through the same
`apply_cage` and the same broker port a rule agent's intents go through.

The audit constraints this package carries, each with a test:

- **Bring your own model.** The key or the login is the user's. Two adapters
  are shipped (`providers.py` is the closed list): `anthropic_client.py`, the
  API on the user's key, and `codex_client.py`, the Codex CLI on the user's
  own login. Each names no endpoint of ours, and Tick operates none. Which one
  an agent uses is pinned in its document as `provider`.
- **Tick authors no strategies.** The prompt is the user's instructions plus
  the snapshot, and `ModelRequest` has no field for text of Tick's own. There
  is no default instruction set, no example, no shortlist and no heuristic
  anywhere in this package; a missing instructions file refuses.
- **Long only, and only inside the user's universe.** A sell larger than the
  position, and a symbol the user's own document did not name, are refused
  whole and recorded — never truncated, never quietly dropped.
- **Accurate naming.** These are model-driven agents and the model id is shown
  and recorded. A deterministic spec agent is a rule agent and is never called
  anything else.
"""

from __future__ import annotations

from .anthropic_client import (
    API_KEY_ENV,
    AnthropicChatClient,
    AnthropicModelClient,
    read_model_reply,
)
from .client import ModelClient, ModelReply, ModelRequest, StructuredReply, intents_of
from .codex_client import CODEX_BINARY, CodexModelClient
from .errors import (
    InstructionsMissing,
    MissingApiKey,
    ModelAgentError,
    ModelReplyError,
    ProviderUnavailable,
    ThreadLost,
)
from .model_agent import (
    MAX_OUTPUT_TOKENS,
    MODEL_SOURCE_PREFIX,
    PROMPT_JOIN,
    ModelAgent,
    model_id_of,
)
from .providers import (
    PROVIDERS,
    Provider,
    ProviderInfo,
    ProviderShape,
    availability,
    client_for,
)
from .schema import (
    EMIT_TOOL_NAME,
    MAX_INTENTS,
    TOOL_NAMES,
    emit_tool,
    intents_schema,
    tool_definitions,
)
from .snapshot import build_snapshot, snapshot_json
from .spec import (
    MODEL_AGENT_KIND,
    AgentSpec,
    ModelAgentSpec,
    agent_spec_id,
    dump_agent_spec,
    is_model_agent,
    load_agent_spec_file,
    loads_agent_spec,
    parse_agent_spec,
)

__all__ = [
    "API_KEY_ENV",
    "CODEX_BINARY",
    "EMIT_TOOL_NAME",
    "MAX_INTENTS",
    "MAX_OUTPUT_TOKENS",
    "MODEL_AGENT_KIND",
    "MODEL_SOURCE_PREFIX",
    "PROMPT_JOIN",
    "TOOL_NAMES",
    "AgentSpec",
    "AnthropicModelClient",
    "AnthropicChatClient",
    "CodexModelClient",
    "InstructionsMissing",
    "MissingApiKey",
    "ModelAgent",
    "ModelAgentError",
    "ModelAgentSpec",
    "ModelClient",
    "ModelReply",
    "ModelReplyError",
    "ModelRequest",
    "StructuredReply",
    "PROVIDERS",
    "Provider",
    "ProviderInfo",
    "ProviderShape",
    "ProviderUnavailable",
    "ThreadLost",
    "agent_spec_id",
    "availability",
    "build_snapshot",
    "client_for",
    "dump_agent_spec",
    "emit_tool",
    "intents_of",
    "intents_schema",
    "is_model_agent",
    "load_agent_spec_file",
    "loads_agent_spec",
    "model_id_of",
    "parse_agent_spec",
    "read_model_reply",
    "snapshot_json",
    "tool_definitions",
]
