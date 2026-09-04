"""Brokers — the only place an order becomes real, and the simulation that isn't.

    from tick.broker import PaperBroker

    broker = PaperBroker(market, starting_cash=Decimal("10000.00"), max_cancels=5)
    result = broker.place(intent)

`BrokerPort` is the protocol the runtime speaks. `PaperBroker` implements it as
a local simulation: it computes fills on the user's machine from the same
market data the engine read, and it sends nothing anywhere.
`ProfileBroker` implements it over a broker MCP through a freshly verified
profile rather than through any assumption about tool names. It reads and
trades ONE account and drops every row for another account before it leaves
the adapter. `RobinhoodMCPBroker` is a one-release deprecated import name for
that same verified boundary.

Long-only is enforced here as well as in the engine, on purpose. The engine
refuses a sell larger than the position when it sizes the order; the broker
refuses it again when it places one. An intent can reach a broker from more
places than the evaluator — a model agent, a replay, a future manual command —
and "the caller already checked" is not a property a brokerage boundary should
depend on.
"""

from __future__ import annotations

from .errors import BrokerError, BrokerUnavailable, CapabilityUnmapped, ToolResultUnreadable
from .mcp_session import MCPSession, SessionOpener, streamable_http_session
from .paper import AVG_COST_QUANTUM, PaperBroker
from .port import (
    BrokerOrder,
    BrokerPort,
    Cancelled,
    CancelResult,
    Fill,
    OrderOutcomeUnknown,
    OrderState,
    PlaceResult,
    RejectCode,
    Rejected,
)
from .profile import (
    HOST_ALLOWLIST,
    Category,
    DriftDifference,
    Profile,
    ProfileProposal,
    ProfileState,
    ProfileTool,
    ProposalEdit,
    ProposalReply,
    ProposalReplyTool,
    ToolContract,
    ToolState,
    VerifiedSessionProfile,
    build_profile,
    confirm_profile,
    contract_for,
    diff_profile,
    edit_proposal,
    has_confirmation_note,
    inventory_hash,
    load_profile,
    load_proposal,
    mapping_hash,
    profile_path,
    proposal_path,
    propose_profile,
    prove_profile,
    prove_proposal,
    sanction_for,
    save_profile,
    save_proposal,
    verify_session_profile,
)
from .profile_broker import ProfileBroker
from .profile_model import MODEL_PROPOSAL_TOOL, ModelCategorizer, check_proposal
from .robinhood import RobinhoodMCPBroker
from .toolmap import (
    Capability,
    CapabilityMapping,
    DiscoveredTool,
    Proposal,
    ToolMap,
    load_tool_map,
    propose,
    save_tool_map,
    toolmap_path,
)

__all__ = [
    "AVG_COST_QUANTUM",
    "BrokerError",
    "BrokerOrder",
    "BrokerPort",
    "BrokerUnavailable",
    "Category",
    "Capability",
    "CapabilityMapping",
    "CapabilityUnmapped",
    "DiscoveredTool",
    "DriftDifference",
    "HOST_ALLOWLIST",
    "MCPSession",
    "Proposal",
    "RobinhoodMCPBroker",
    "SessionOpener",
    "ToolMap",
    "ToolResultUnreadable",
    "load_tool_map",
    "propose",
    "save_tool_map",
    "streamable_http_session",
    "toolmap_path",
    "CancelResult",
    "Cancelled",
    "Fill",
    "OrderOutcomeUnknown",
    "OrderState",
    "PaperBroker",
    "Profile",
    "ProfileBroker",
    "ProfileProposal",
    "ProposalEdit",
    "ProposalReply",
    "ProposalReplyTool",
    "ProfileState",
    "ProfileTool",
    "PlaceResult",
    "RejectCode",
    "Rejected",
    "ToolContract",
    "ToolState",
    "VerifiedSessionProfile",
    "build_profile",
    "confirm_profile",
    "contract_for",
    "diff_profile",
    "edit_proposal",
    "has_confirmation_note",
    "inventory_hash",
    "load_profile",
    "load_proposal",
    "mapping_hash",
    "profile_path",
    "proposal_path",
    "propose_profile",
    "MODEL_PROPOSAL_TOOL",
    "ModelCategorizer",
    "check_proposal",
    "prove_profile",
    "prove_proposal",
    "sanction_for",
    "save_profile",
    "save_proposal",
    "verify_session_profile",
]
