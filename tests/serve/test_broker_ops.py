"""Account discovery stays on the box and binds only broker-declared eligibility."""

from __future__ import annotations

import stat
from datetime import UTC, datetime

import pytest

from tick.broker import (
    Category,
    DiscoveredTool,
    ProfileProposal,
    ProfileState,
    ProfileTool,
    ProposalReplyTool,
    confirm_profile,
    contract_for,
    inventory_hash,
    load_profile,
    save_proposal,
)
from tick.broker.profile import (
    CANONICALIZER_VERSION,
    CATEGORY_REGISTRY_VERSION,
    PROFILE_FORMAT_VERSION,
    ProposedTool,
    build_profile,
    mapping_hash,
)
from tick.serve.broker_ops import BrokerOperations

AT = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)
SERVER = "https://agent.robinhood.com/mcp/trading"


def accounts_tool() -> DiscoveredTool:
    row = {
        "type": "object",
        "properties": {
            "account_number": {"type": "string"},
            "agentic_allowed": {"type": "boolean"},
            "brokerage_account_type": {"type": "string"},
        },
        "required": ["account_number", "agentic_allowed", "brokerage_account_type"],
    }
    return DiscoveredTool(
        name="get_accounts",
        title=None,
        description="List accounts and the explicit eligibility field.",
        input_schema={"type": "object", "additionalProperties": False},
        output_schema={
            "type": "object",
            "properties": {
                "data": {
                    "type": "object",
                    "properties": {"accounts": {"type": "array", "items": row}},
                }
            },
        },
        annotations={"readOnlyHint": True},
        execution=None,
    )


class Session:
    def __init__(self, discovered, rows) -> None:
        self.discovered = discovered
        self.rows = rows
        self.calls = []

    def list_tools(self):
        return [self.discovered]

    def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return {"data": {"accounts": self.rows}}

    def close(self):
        return None


class Loopback:
    def __exit__(self, *_args):
        return None


class Operations(BrokerOperations):
    def __init__(self, *, home, session) -> None:
        super().__init__(home=home, timeout_seconds=1.0)
        self.fake_session = session

    def _session(self, server):
        assert server == SERVER
        return self.fake_session, Loopback()


def configured(home, rows):
    discovered = accounts_tool()
    contract = contract_for(discovered)
    original = ProposalReplyTool(
        name=discovered.name,
        category=Category.READ_ACCOUNTS,
        arguments={},
        result={
            "items": "data.accounts",
            "account": "account_number",
            "eligible": "agentic_allowed",
            "kind": "brokerage_account_type",
        },
        reason="This lists accounts with broker-declared eligibility.",
    )
    proposed = ProposedTool(
        contract=contract,
        category=original.category,
        arguments=original.arguments,
        result=original.result,
        reason=original.reason,
        warnings=(),
        original=original,
        edits=(),
    )
    proposal = ProfileProposal(
        server=SERVER,
        account_id=None,
        sanction="official",
        inventory_hash=inventory_hash([contract]),
        tools={discovered.name: proposed},
        categorizer_version="model-v1:provider-model-fixture",
        proposed_at=AT,
    )
    save_proposal(home, proposal)
    mapping = ProfileTool(
        category=Category.READ_ACCOUNTS,
        contract=contract,
        arguments={},
        result=original.result,
        confirmed_contract_hash=contract.contract_hash,
        mapping_hash=mapping_hash(Category.READ_ACCOUNTS, {}, original.result),
        confirmed_at=AT,
        confirmed_by="api",
        categorizer_version=proposal.categorizer_version,
        proved_contract_hash=None,
        proved_mapping_hash=None,
        proved_at=None,
        proof=None,
    )
    profile = build_profile(
        server=SERVER,
        account_id=None,
        tools={discovered.name: mapping},
        inventory_hash=proposal.inventory_hash,
        data_class="display_only",
        sanction="official",
        profile_format_version=PROFILE_FORMAT_VERSION,
        canonicalizer_version=CANONICALIZER_VERSION,
        category_registry_version=CATEGORY_REGISTRY_VERSION,
        state=ProfileState.CONFIRMED,
        observed_inventory_hash=proposal.inventory_hash,
        drift=(),
    )
    confirm_profile(home, profile, actor="box-api", at=AT)
    return Operations(home=home, session=Session(discovered, rows))


def row(number: str, eligible: bool, kind: str = "individual"):
    return {
        "account_number": number,
        "agentic_allowed": eligible,
        "brokerage_account_type": kind,
    }


def test_one_eligible_account_is_selected_without_sending_its_number_to_the_phone(tmp_path):
    number = "account-placeholder-1234"
    operations = configured(tmp_path, [row(number, True), row("other-placeholder-5678", False)])
    result = operations.accounts()
    assert result["accounts"][0]["account_number_masked"] == "••••1234"
    assert result["selected"] == {"account_ref": result["accounts"][0]["account_ref"]}
    assert number not in str(result)
    assert load_profile(tmp_path).account_id == number
    refs = tmp_path / "broker" / "account-refs.json"
    assert stat.S_IMODE(refs.stat().st_mode) == 0o600
    assert number not in (tmp_path / "broker" / "records.jsonl").read_text()


def test_several_eligible_accounts_wait_for_the_persons_opaque_choice(tmp_path):
    operations = configured(
        tmp_path,
        [row("account-placeholder-1234", True), row("account-placeholder-5678", True)],
    )
    listed = operations.accounts()
    assert listed["selected"] is None
    chosen = operations.select_account({"account_ref": listed["accounts"][1]["account_ref"]})
    assert chosen == {"account_ref": listed["accounts"][1]["account_ref"]}
    assert load_profile(tmp_path).account_id == "account-placeholder-5678"


def test_no_eligible_account_refuses_with_the_robinhood_correction(tmp_path):
    operations = configured(tmp_path, [row("account-placeholder-1234", False)])
    with pytest.raises(ValueError) as caught:
        operations.accounts()
    assert str(caught.value) == (
        "Robinhood reports no account accessible to this agent. Review account access at "
        "Robinhood, then read accounts again."
    )


def test_propose_records_its_note_on_the_real_ledger(tmp_path):
    """Live 2026-09-04: the proposal saved, then the ledger note raised (no clock)
    and the person saw internal_error after a minute of waiting."""
    from tick.records import read

    class NoProviderOperations(Operations):
        def _categorizer(self, _body):
            return None  # deterministic fallback; the ledger path is what is under test

    ops = NoProviderOperations(home=tmp_path, session=Session(accounts_tool(), []))
    result = ops.propose({"server_url": SERVER})
    assert result["state"] == "done"
    rows = list(read(tmp_path / "broker" / "records.jsonl"))
    assert rows[-1].payload["event"] == "broker_profile_proposed"
    assert rows[-1].payload["tools_total"] == 1
