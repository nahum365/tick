"""What may be recorded, and what is refused rather than rounded.

The payload boundary is where invariant 5 meets serialisation. A `Decimal` that
became a float on the way into the record would put a number nobody computed
into permanent evidence, and it would do it silently — the file would still
verify, because the hash would be over the wrong number.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from pydantic import BaseModel

from tick.records import MAX_PAYLOAD_DEPTH, PayloadError, canonical_timestamp, normalize_payload
from tick.records.payload import normalize_value


class Money(BaseModel):
    amount: Decimal
    label: str


class Sloppy(BaseModel):
    ratio: float


def test_a_decimal_becomes_its_exact_string():
    assert normalize_payload({"price": Decimal("12.50")}) == {"price": "12.50"}


@pytest.mark.parametrize("literal", ["12.50", "12.5", "0.00000001", "1E+3", "-4.25", "0"])
def test_every_decimal_round_trips_through_the_payload_unchanged(literal):
    """The string in the record reconstructs the same Decimal, digit for digit.

    Exponent form and all: `str(Decimal("0.00000001"))` is `"1E-8"`, and what
    matters is that `Decimal(recorded)` is the identical number with the
    identical scale — `as_tuple()` compares both, where `==` would call
    `1.5` and `1.50` the same and miss a lost digit.
    """
    recorded = normalize_payload({"n": Decimal(literal)})["n"]
    assert recorded == str(Decimal(literal))
    assert Decimal(recorded).as_tuple() == Decimal(literal).as_tuple()


def test_a_trailing_zero_survives_because_it_is_part_of_the_number_as_written():
    assert normalize_payload({"a": Decimal("1.5")})["a"] == "1.5"
    assert normalize_payload({"a": Decimal("1.50")})["a"] == "1.50"


def test_a_binary_float_is_refused_and_the_message_says_where():
    with pytest.raises(PayloadError) as exc:
        normalize_payload({"fill": {"price": 12.5}})
    assert "payload.fill.price" in str(exc.value)
    assert "binary float" in str(exc.value)


def test_a_float_inside_a_list_is_refused_with_its_index():
    with pytest.raises(PayloadError) as exc:
        normalize_payload({"prices": [Decimal("1.00"), 2.5]})
    assert "payload.prices[1]" in str(exc.value)


def test_a_float_that_arrives_inside_a_model_is_refused_too():
    """A model is not trusted for being a model; its dump is walked like anything else."""
    with pytest.raises(PayloadError):
        normalize_payload({"thing": Sloppy(ratio=0.5)})


def test_a_pydantic_model_is_recorded_as_its_json_dump():
    payload = normalize_payload({"cost": Money(amount=Decimal("184.20"), label="XYZ")})
    assert payload == {"cost": {"amount": "184.20", "label": "XYZ"}}


def test_a_non_finite_decimal_is_refused():
    with pytest.raises(PayloadError):
        normalize_payload({"n": Decimal("NaN")})


def test_a_bool_stays_a_bool_and_does_not_become_a_number():
    payload = normalize_payload({"live": True, "count": 1})
    assert payload["live"] is True
    assert payload["count"] == 1


def test_an_aware_datetime_becomes_its_iso_form():
    moment = datetime(2026, 9, 1, 13, 30, tzinfo=UTC)
    assert normalize_payload({"at": moment})["at"] == "2026-09-01T13:30:00+00:00"


def test_a_naive_datetime_is_refused():
    with pytest.raises(PayloadError) as exc:
        canonical_timestamp(datetime(2026, 9, 1, 13, 30))
    assert "no timezone" in str(exc.value)


def test_a_non_string_key_is_refused_rather_than_renamed():
    with pytest.raises(PayloadError) as exc:
        normalize_payload({"seqs": {1: "one"}})
    assert "keys are strings" in str(exc.value)


@pytest.mark.parametrize("value", [object(), {"a", "b"}, b"bytes", complex(1, 2)])
def test_a_value_with_no_exact_json_form_is_refused(value):
    with pytest.raises(PayloadError):
        normalize_payload({"thing": value})


def test_a_payload_is_an_object_not_a_list_or_a_scalar():
    for value in ([1, 2], "text", 7):
        with pytest.raises(PayloadError):
            normalize_payload(value)


def test_a_payload_that_nests_too_deep_is_refused_rather_than_overflowing():
    deep: object = "bottom"
    for _ in range(MAX_PAYLOAD_DEPTH + 2):
        deep = {"down": deep}
    with pytest.raises(PayloadError) as exc:
        normalize_payload(deep)
    assert "nests deeper" in str(exc.value)


def test_normalizing_is_idempotent():
    """A payload read back out of a record can be recorded again unchanged."""
    once = normalize_payload({"price": Decimal("12.50"), "at": datetime(2026, 9, 1, tzinfo=UTC)})
    assert normalize_payload(once) == once


def test_normalize_value_leaves_none_alone():
    assert normalize_value(None) is None
