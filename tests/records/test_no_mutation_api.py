"""There is no way to unsay something, and this test is why it stays that way.

Invariant 4 is an absence: no delete, no edit, no rewrite, no compaction — a
record that no longer applies is superseded by a `retired` record naming it.
An absence cannot be reviewed by reading a diff, because the diff that breaks
it looks like a helpful `def prune_old_records(...)` somebody needed on a
Friday. So the absence is asserted against the package's own syntax tree: the
names it defines, and the calls it makes.

The check is deliberately blunt. A legitimate function that trips it should be
renamed, or this list should be argued with in review — which is exactly the
conversation that ought to happen before a mutation path lands in the record.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import tick.records
from tick.records import Ledger

PACKAGE = Path(tick.records.__file__).parent

#: Name stems that describe unsaying something. Matched against every function
#: and method the package defines.
MUTATION_STEMS = (
    "delete",
    "remove",
    "erase",
    "purge",
    "prune",
    "drop",
    "truncate",
    "compact",
    "rewrite",
    "edit",
    "amend",
    "modify",
    "overwrite",
    "update",
    "revise",
    "patch",
)

#: Calls that would remove or replace the file itself.
FORBIDDEN_CALLS = frozenset(
    {
        "remove",
        "unlink",
        "rmtree",
        "truncate",
        "rename",
        "replace",
        "write_text",
        "write_bytes",
    }
)


def modules() -> list[Path]:
    files = sorted(PACKAGE.rglob("*.py"))
    assert files, "no source files found for tick.records"
    return files


def tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


@pytest.mark.parametrize("path", modules(), ids=lambda p: p.name)
def test_the_package_defines_no_function_that_unsays_something(path: Path):
    defined = [
        node.name
        for node in ast.walk(tree(path))
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    ]
    offending = [name for name in defined if any(stem in name.lower() for stem in MUTATION_STEMS)]
    assert not offending, (
        f"{path.name} defines {offending}. Nothing in the record is ever deleted or "
        f"edited (invariant 4); a row that no longer applies is superseded by an "
        f"appended `retired` record naming it."
    )


@pytest.mark.parametrize("path", modules(), ids=lambda p: p.name)
def test_the_package_never_calls_anything_that_would_replace_the_file(path: Path):
    called = {
        node.func.attr
        for node in ast.walk(tree(path))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    offending = called & FORBIDDEN_CALLS
    assert not offending, (
        f"{path.name} calls {sorted(offending)}. The ledger is opened for appending "
        f"and nothing else; replacing the file wholesale is how an append-only record "
        f"quietly becomes an editable one."
    )


def test_the_public_api_offers_no_way_to_take_a_record_back():
    """What a caller can reach: append, retire, read, verify. Nothing else writes."""
    public = {name for name in dir(Ledger) if not name.startswith("_")}
    assert public == {"append", "retire", "records", "last", "verify"}
    assert not {
        name
        for name in tick.records.__all__
        if any(stem in name.lower() for stem in MUTATION_STEMS)
    }
