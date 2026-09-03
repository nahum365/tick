"""Tick — the always-on agent runtime for Robinhood Agentic accounts.

The runtime lives on the USER's machine (laptop or a VPS in their own cloud
account). It authenticates to Robinhood's Trading MCP with the user's own
OAuth grant, executes the user's strategies — deterministic specs, or a model
caged inside deterministic limits — against the user's Agentic account, and
keeps an immutable forward-only record of everything it did.

Invariants live in CLAUDE.md. The short version: credentials never leave the
user's box; paper first; the spec decides, the model compiles and explains;
no number is ever fabricated.
"""

__version__ = "0.0.1"
