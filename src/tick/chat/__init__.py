"""Private, persistent conversations on the user's box."""

from .session import CHAT_FRAME, ChatError, ChatSession, ChatTurn, stream_turn
from .setup import SETUP_FRAMES, SetupChatSession, SetupScope, SetupState
from .setup_loop import MAX_SETUP_MODEL_TURNS, SetupLoopDecision, run_setup_loop

__all__ = [
    "CHAT_FRAME",
    "SETUP_FRAMES",
    "ChatError",
    "ChatSession",
    "ChatTurn",
    "MAX_SETUP_MODEL_TURNS",
    "SetupChatSession",
    "SetupLoopDecision",
    "SetupScope",
    "SetupState",
    "stream_turn",
    "run_setup_loop",
]
