"""One record: its hash, its chain position, and the line it is written as.

The property under test is that a `Record` cannot exist in a state that lies.
Its hash is checked on construction — from a file or from Python — so every
place downstream that holds a `Record` is holding something that verified.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from tick.records import (
    GENESIS_PREV_HASH,
    DataSource,
    LedgerCorrupt,
    PayloadError,
    Record,
    RecordKind,
    decode_line,
    encode_line,
    record_hash,
)
from tick.spec import canonical_encode, sha256_hex

TS = datetime(2026, 9, 1, 13, 30, tzinfo=UTC)


def genesis(**overrides) -> Record:
    fields = {
        "seq": 1,
        "ts": TS,
        "kind": RecordKind.NOTE,
        "payload": {"text": "placeholder", "source": DataSource.FIXTURE.value},
        "prev_hash": GENESIS_PREV_HASH,
    }
    fields.update(overrides)
    return Record.chained(**fields)


def test_the_genesis_constant_is_pinned():
    """Changing it invalidates every ledger ever written; it may never drift silently."""
    assert GENESIS_PREV_HASH == sha256_hex(b"tick.records.chain.v1.genesis")
    assert GENESIS_PREV_HASH == ("be09c05a18bf24597788e3345c9afa6fff1cfa17edd6b01f3a5f0a268f8392ab")


def test_the_hash_is_sha256_over_the_canonical_body():
    record = genesis()
    expected = sha256_hex(
        canonical_encode(
            {
                "seq": 1,
                "ts": "2026-09-01T13:30:00+00:00",
                "kind": "note",
                "payload": {"source": "fixture", "text": "placeholder"},
                "prev_hash": GENESIS_PREV_HASH,
            }
        )
    )
    assert record.hash == expected
    assert (
        record_hash(
            seq=record.seq,
            ts=record.ts,
            kind=record.kind,
            payload=record.payload,
            prev_hash=record.prev_hash,
        )
        == record.hash
    )


def test_a_record_whose_hash_does_not_match_its_contents_cannot_be_built():
    record = genesis()
    with pytest.raises(ValidationError, match="does not match its contents"):
        Record(
            seq=record.seq,
            ts=record.ts,
            kind=record.kind,
            payload=record.payload | {"text": "something else"},
            prev_hash=record.prev_hash,
            hash=record.hash,
        )


@pytest.mark.parametrize(
    "change",
    [
        pytest.param({"seq": 2, "prev_hash": "b" * 64}, id="seq"),
        pytest.param({"ts": datetime(2026, 9, 1, 13, 31, tzinfo=UTC)}, id="ts"),
        pytest.param({"kind": RecordKind.STOP}, id="kind"),
        pytest.param({"payload": {"text": "other", "source": "fixture"}}, id="payload"),
        pytest.param({"prev_hash": "a" * 64, "seq": 2}, id="prev_hash"),
    ],
)
def test_changing_anything_at_all_changes_the_hash(change):
    assert genesis(**change).hash != genesis().hash


def test_only_the_first_record_may_chain_to_genesis():
    with pytest.raises(ValidationError, match="only record 1 may do"):
        genesis(seq=2)
    with pytest.raises(ValidationError, match="fragment of another ledger"):
        genesis(prev_hash="c" * 64)


def test_a_record_must_name_where_its_data_came_from():
    with pytest.raises(ValidationError, match="every payload names where its data came from"):
        genesis(payload={"text": "placeholder"})
    with pytest.raises(ValidationError, match="is not a data source"):
        genesis(payload={"text": "placeholder", "source": "somewhere"})


def test_the_source_is_readable_off_the_record():
    record = genesis(payload={"text": "x", "source": "robinhood"})
    assert record.source is DataSource.ROBINHOOD
    assert record.source.derived_from_robinhood is True


@pytest.mark.parametrize("source", [DataSource.FIXTURE, DataSource.PAPER])
def test_only_robinhood_sourced_rows_are_marked_as_robinhood_derived(source):
    """The flag a later export tool reads to decide what it must refuse to ship."""
    assert source.derived_from_robinhood is False


def test_the_kinds_are_a_closed_vocabulary():
    assert {kind.value for kind in RecordKind} == {
        "decision",
        "order",
        "fill",
        "rejected",
        "refusal",
        "mode_change",
        "stop",
        "note",
        "retired",
    }


def test_a_decimal_in_a_payload_is_stored_exactly_and_reads_back_exactly():
    record = genesis(payload={"price": Decimal("184.20"), "source": "paper"})
    assert record.payload["price"] == "184.20"
    assert Decimal(record.payload["price"]) == Decimal("184.20")
    assert '"price":"184.20"' in record.as_line()


def test_a_line_round_trips_through_encode_and_decode():
    record = genesis()
    assert decode_line(encode_line(record), at=1) == record


def test_a_line_must_be_the_canonical_encoding_of_what_it_contains():
    """Re-indenting or reordering a line is a rewrite, even when it decodes the same."""
    record = genesis()
    document = json.loads(record.as_line())
    for rewritten in (
        json.dumps(document, indent=2),
        json.dumps(dict(reversed(list(document.items())))),
        json.dumps(document, separators=(", ", ": ")),
    ):
        with pytest.raises(LedgerCorrupt, match="not in canonical form"):
            decode_line(rewritten, at=1)


def test_a_duplicate_key_does_not_slip_past_as_the_last_value():
    record = genesis()
    doubled = record.as_line().replace('"seq":1', '"seq":9,"seq":1', 1)
    with pytest.raises(LedgerCorrupt):
        decode_line(doubled, at=1)


def test_a_bare_decimal_number_in_a_line_is_refused_rather_than_rounded():
    line = '{"hash":"a","kind":"note","payload":{"p":12.5},"prev_hash":"b","seq":1,"ts":"x"}'
    with pytest.raises(LedgerCorrupt, match="recorded as strings"):
        decode_line(line, at=1)


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        pytest.param("not json at all", "not JSON", id="garbage"),
        pytest.param("[1,2,3]", "JSON object", id="array"),
        pytest.param('{"seq":1}', "missing", id="missing-fields"),
    ],
)
def test_a_line_that_is_not_a_record_says_so_and_names_its_position(line, expected):
    with pytest.raises(LedgerCorrupt, match=expected) as exc:
        decode_line(line, at=4)
    assert exc.value.at == 4


def test_a_record_refuses_a_payload_that_is_not_an_object():
    with pytest.raises(PayloadError):
        Record.chained(
            seq=1,
            ts=TS,
            kind=RecordKind.NOTE,
            payload=["not", "an", "object"],
            prev_hash=GENESIS_PREV_HASH,
        )


def test_a_record_is_frozen():
    record = genesis()
    with pytest.raises(ValidationError):
        record.seq = 2


def test_a_broken_line_names_its_position_once_not_twice():
    """`LedgerCorrupt` prefixes the position; the validator's message already had it.

    The prefix says "line 1", not "record 1": the position is something the
    reader can count in the file, while a `seq` read out of a line nobody has
    verified yet is only what that line claims.
    """
    record = genesis()
    tampered = record.as_line().replace('"text":"placeholder"', '"text":"changed"')
    with pytest.raises(LedgerCorrupt) as exc:
        decode_line(tampered, at=1)
    message = str(exc.value)
    assert message.count("line 1") == 1
    assert message.startswith("line 1: hash ")
    assert "does not match its contents" in message


def test_a_rewritten_line_says_so_and_does_not_blame_the_chain():
    """Not in canonical form is its own finding, and it claims nothing else.

    The byte check knows one thing: these are not the bytes Tick writes. It
    cannot tell an editor's reformatting from a torn write, and it has learned
    nothing about the chain — the record inside may hash perfectly and link
    perfectly. So the sentence names the two possible causes and disclaims the
    failure it did not observe.
    """
    record = genesis()
    reindented = json.dumps(json.loads(record.as_line()), indent=2)
    with pytest.raises(LedgerCorrupt) as exc:
        decode_line(reindented, at=4)
    message = str(exc.value)
    assert message.startswith("line 4: the line is not in canonical form")
    assert "rewritten in place or torn" in message
    assert "predecessor" not in message
