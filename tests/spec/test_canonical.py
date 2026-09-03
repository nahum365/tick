"""The immutability primitive: one spec, one id, on any machine.

The record in slice 03 chains against `spec_id`, and `tick run --live` will
refuse a spec whose id is not the one the user approved. So these tests are
about a property, not a function: reformatting a document must not move its
id, and changing anything at all must.
"""

from __future__ import annotations

import copy
import hashlib
import json
from decimal import Decimal

import pytest

from tick.spec import (
    canonical_bytes,
    canonical_dumps,
    canonical_encode,
    canonical_json,
    dump_spec,
    load_spec_file,
    loads_spec,
    parse_spec,
    sha256_hex,
    short_spec_id,
    spec_id,
)

from .conftest import read_document, valid_paths


def _reorder_keys(node):
    """Same document, every object's keys written in the opposite order."""
    if isinstance(node, dict):
        return {key: _reorder_keys(node[key]) for key in sorted(node, reverse=True)}
    if isinstance(node, list):
        return [_reorder_keys(item) for item in node]
    return node


@pytest.mark.parametrize("path", valid_paths(), ids=lambda p: p.stem)
def test_key_order_and_whitespace_do_not_change_the_id(path):
    original = load_spec_file(path)
    shuffled = json.dumps(_reorder_keys(read_document(path)), indent=7, default=str)
    compact = json.dumps(read_document(path), separators=(",", ":"), default=str)
    assert spec_id(loads_spec(shuffled)) == spec_id(original)
    assert spec_id(loads_spec(compact)) == spec_id(original)


@pytest.mark.parametrize("path", valid_paths(), ids=lambda p: p.stem)
def test_the_id_survives_a_dump_and_reload(path):
    original = load_spec_file(path)
    assert spec_id(loads_spec(dump_spec(original))) == spec_id(original)


def test_canonical_json_sorts_keys_and_carries_no_whitespace(simple_document):
    encoded = canonical_json(parse_spec(simple_document))
    assert ", " not in encoded and ": " not in encoded
    keys = ("cadence", "cage", "name", "rules", "universe", "version")
    positions = [encoded.index(f'"{key}":') for key in keys]
    assert positions == sorted(positions)


def test_canonical_json_writes_decimals_as_exact_strings(simple_document):
    encoded = canonical_json(parse_spec(simple_document))
    assert '"max_order_notional":"500.00"' in encoded
    assert '"value":"100.00"' in encoded


def test_canonical_bytes_is_the_utf8_encoding(simple_document):
    spec = parse_spec(simple_document)
    assert canonical_bytes(spec) == canonical_json(spec).encode("utf-8")


def test_the_id_is_lowercase_sha256_hex(simple_document):
    identifier = spec_id(parse_spec(simple_document))
    assert len(identifier) == 64
    assert identifier == identifier.lower()
    int(identifier, 16)


def test_short_id_is_a_prefix_of_the_id(simple_document):
    spec = parse_spec(simple_document)
    assert spec_id(spec).startswith(short_spec_id(spec))
    assert len(short_spec_id(spec)) == 12


def test_the_id_is_stable_across_repeated_encodings(simple_document):
    spec = parse_spec(simple_document)
    assert len({spec_id(spec) for _ in range(5)}) == 1


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda d: d.__setitem__("name", "Renamed"), id="name"),
        pytest.param(lambda d: d.__setitem__("version", 2), id="version"),
        pytest.param(lambda d: d["universe"].append("MSFT"), id="universe"),
        pytest.param(lambda d: d.__setitem__("cadence", {"kind": "daily_open"}), id="cadence"),
        pytest.param(lambda d: d["rules"][0].__setitem__("id", "dip2"), id="rule-id"),
        pytest.param(
            lambda d: d["rules"][0]["when"].__setitem__("op", ">"),
            id="operator",
        ),
        pytest.param(
            lambda d: d["rules"][0]["then"]["size"].__setitem__("shares", 2),
            id="size",
        ),
        pytest.param(
            lambda d: d["cage"].__setitem__("max_positions", 4),
            id="cage",
        ),
    ],
)
def test_any_change_at_all_produces_a_different_id(mutate, simple_document):
    before = spec_id(parse_spec(copy.deepcopy(simple_document)))
    mutate(simple_document)
    assert spec_id(parse_spec(simple_document)) != before


def test_a_trailing_zero_is_part_of_the_document(simple_document):
    """`1.5` and `1.50` are the same number and different documents.

    The id names what a human approved, so it follows the text, not the
    arithmetic. Normalising here would make two approvals indistinguishable.
    """
    document = copy.deepcopy(simple_document)
    document["cage"]["max_daily_drawdown_pct"] = "2.0"
    other = copy.deepcopy(simple_document)
    other["cage"]["max_daily_drawdown_pct"] = "2.00"
    assert parse_spec(document).cage.max_daily_drawdown_pct == Decimal("2.00")
    assert spec_id(parse_spec(document)) != spec_id(parse_spec(other))


@pytest.mark.parametrize("path", valid_paths(), ids=lambda p: p.stem)
def test_the_shipped_fixtures_all_have_distinct_ids(path):
    ids = {spec_id(load_spec_file(other)) for other in valid_paths()}
    assert len(ids) == len(valid_paths())
    assert spec_id(load_spec_file(path)) in ids


def test_the_spec_id_is_the_shared_primitives_applied_to_the_model(simple_document):
    """`spec_id` is `sha256_hex(canonical_encode(...))` and nothing else.

    The record chain hashes with the same two functions. Pinning the
    composition here is what stops the two from drifting into different
    notions of canonical while both docstrings keep claiming one.
    """
    spec = parse_spec(simple_document)
    payload = spec.model_dump(mode="json")
    assert canonical_json(spec) == canonical_dumps(payload)
    assert canonical_bytes(spec) == canonical_encode(payload)
    assert spec_id(spec) == sha256_hex(canonical_encode(payload))


def test_canonical_dumps_sorts_keys_at_every_depth():
    payload = {"b": 1, "a": {"z": [{"y": 2, "x": 3}], "w": None}}
    assert canonical_dumps(payload) == '{"a":{"w":null,"z":[{"x":3,"y":2}]},"b":1}'


def test_canonical_dumps_does_not_escape_non_ascii():
    assert canonical_dumps({"note": "café"}) == '{"note":"café"}'


def test_sha256_hex_is_lowercase_hex_of_the_bytes_it_was_given():
    digest = sha256_hex(b"tick")
    assert digest == hashlib.sha256(b"tick").hexdigest()
    assert digest == digest.lower()
    assert len(digest) == 64
