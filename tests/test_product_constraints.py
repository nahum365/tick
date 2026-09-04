"""Constraints that are true of the code as a whole, pinned mechanically.

These come from the 2026-08-31 compliance and terms audit (Robinhood's Customer
Agreement §29 "API & MCP", the Market Data Addendum, and Advisers Act case law).
They are properties nobody can verify by reading a diff — "no network call
anywhere in the engine" is exactly the kind of claim that stays true until the
day someone adds one import — so they are tested against the source tree.

Each test names the constraint it enforces. The scope is every package built so
far — `tick.engine`, `tick.broker`, `tick.records`, `tick.runtime` — and
`tick/cli.py`, which is the product's face and the surface a "starter strategy"
would appear on. Slices that add `auth/` and the Robinhood adapter extend the
lists here rather than exempting themselves — and the record is in scope for a
reason of its own, since it is the one file on disk holding what the user owns
and what was done with it. Nothing may carry it off the machine.

`tick.compile`, `tick.agents`, and `tick.interview` are the packages that may hold a model
client, and they are scanned by their own rules rather than exempted
(`MODEL_PACKAGES`): each may import the provider SDK the user pays for and
NOTHING else that opens a socket, neither may name an endpoint of Tick's own,
and both are held to every naming and no-strategies rule the rest of the
product is. The claim "there is no hosted LLM path" is that pair of scans, not
a sentence in a README.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tick.cli import app

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "tick"

#: Directories of documents that are records of how the code was built, not
#: surfaces the product presents: one description per merged slice, and the
#: compliance audit as it was written. The audit quotes primary sources and the
#: PR descriptions are engineering history — rewriting either to satisfy a
#: scan would destroy the thing that makes it worth keeping. Everything else
#: under `docs/`, plus the README, is a document a user reads, and the
#: no-strategies and naming rules bind there as hard as they bind `--help`.
#: The list is checked for existence, so a typo cannot silently exempt the tree.
INTERNAL_DOC_DIRS = ("docs/prs", "docs/audit")


def require_private_docs() -> None:
    """Skip in the public runtime export, which ships without `docs/`.

    scripts/publish_runtime.py copies only the box runtime; the design and
    walkthrough documents stay in the private monorepo, so tests that read
    them have nothing to check there.
    """
    if not (ROOT / "docs").is_dir():
        pytest.skip("docs/ is not part of the public runtime export")


#: The packages built so far. Local computation and local files only.
#:
#: `spec` was outside every scan in this file until slice 08, which is how its
#: symbol-format examples came to name two real securities in a validation
#: message a user reads. A package that is not in a list here is not exempt
#: from the rules — it is unchecked against them, which is worse, so the lists
#: are the whole tree by construction (`test_every_package_is_scanned`).
LOCAL_PACKAGES = ("engine", "broker", "records", "runtime", "spec")

#: Packages that may hold a model client — the compiler and the caged model
#: agent, and nothing else. The user's own provider SDK is allowed in these;
#: every other transport is not, and each has exactly ONE file that imports it.
MODEL_PACKAGES = (
    "compile",
    "agents",
    # The interview asks the user's selected adapter through `agents.client_for`;
    # it imports no SDK and persists only private drafts under TICK_HOME.
    "interview",
    # Private transcripts call only the user's selected model adapter.
    "chat",
)

#: Packages that may open a socket, and the only ones: the connect ceremony.
#: They reach Robinhood's own MCP host and the loopback address, and the scans
#: below say so name by name rather than exempting the package.
#: Packages that may open a socket, and the only ones: the connect ceremony and
#: the commons client. `commons` carries closed public-claim/pass shapes only;
#: it cannot accept private brokerage state even though it may use HTTP.
TRANSPORT_PACKAGES = (
    "auth",
    "serve",
    "commons",
    # Iroh and loopback HTTP forwarding are the direct box transport boundary.
    "tunnel",
    # The box MCP speaks stdio only; it imports the MCP SDK and opens no socket.
    "mcpbox",
)

#: The three files inside `broker/` that speak MCP: the synchronous seam, the
#: adapter over it, and the mock server the tests drive. They are named one by
#: one rather than exempting `broker/`, so the paper broker and the port stay
#: under the local-only scan they have been under since slice 02.
BROKER_TRANSPORT_MODULES = (
    "broker/__init__.py",
    "broker/mcp_session.py",
    "broker/mock_mcp.py",
)

#: Where an `http_client` may be handed to a library: the one file that opens
#: the MCP transport, and it passes a client with no base URL of its own — the
#: server URL is the transport's own argument. Nothing else may pass one.
HTTP_CLIENT_EXEMPT = ("broker/mcp_session.py",)

#: Everything the product ships. The naming, no-strategies and flat-fee scans
#: cover all of it; only the transport scan distinguishes the lists.
PRODUCT_PACKAGES = LOCAL_PACKAGES + MODEL_PACKAGES + TRANSPORT_PACKAGES

#: Top-level modules in the same scope. The CLI is the product's face, so the
#: strategy-naming and naming-accuracy scans matter most there.
LOCAL_MODULES = ("cli.py",)

#: The one SDK the compiler may reach for: the user's own model provider, on
#: the user's own key. Every other name in FORBIDDEN_IMPORTS stays forbidden
#: there too, so the compiler cannot grow a second way off the machine.
USERS_PROVIDER_SDK = "anthropic"

#: The one file that may start a process: the CLI-shaped model adapter, which
#: runs the `codex` command the user installed and logged in with. It is
#: exempted by name rather than by package so a second subprocess anywhere in
#: `agents/` or `compile/` still fails this scan.
CLI_ADAPTER_MODULES = ("agents/codex_client.py",)
CLI_ADAPTER_IMPORTS = frozenset({"subprocess"})
PROCESS_MODULES = (
    *CLI_ADAPTER_MODULES,
    "serve/doctor.py",  # doctor asks the installed provider CLI for login status
    "serve/handlers.py",
    "serve/provider_login.py",  # supervises the provider's device-login command
)

#: Ways to point a client somewhere other than where its SDK points by default.
#: A Tick-operated endpoint would be built out of one of these.
ENDPOINT_KEYWORDS = frozenset({"base_url", "api_base", "proxy", "proxies", "http_client"})

#: Anything that could open a socket, build an LLM client, or shell out. The
#: user's positions, balances and prices are computed on the user's own machine
#: and are never transmitted anywhere, so none of these belongs in this code.
FORBIDDEN_IMPORTS = frozenset(
    {
        "anthropic",
        "ftplib",
        "http",
        "httpx",
        "mcp",
        "requests",
        "smtplib",
        "socket",
        "socketserver",
        "ssl",
        "subprocess",
        "telnetlib",
        "urllib",
        "urllib3",
        "webbrowser",
        "xmlrpc",
    }
)

#: Real securities. Tick authors no strategies and names no instruments: not in
#: help text, not in docstrings, not as a default, not as an example. Product
#: code uses placeholders (XYZ, ABCD, WXY).
REAL_TICKERS = re.compile(
    r"\b(AAPL|MSFT|NVDA|TSLA|AMZN|GOOGL?|META|SPY|QQQ|VOO|VTI|BRK\.B|AMD|NFLX|COIN)\b"
)

#: Everything the connect ceremony is allowed to import. It speaks HTTP to one
#: remote host, runs a listener on the loopback interface, and hands a URL to
#: the machine's browser; anything else in FORBIDDEN_IMPORTS stays forbidden.
TRANSPORT_IMPORTS = frozenset(
    {"http", "httpx", "mcp", "socketserver", "ssl", "subprocess", "urllib", "webbrowser"}
)

#: The generally permitted literal hosts: Robinhood's own MCP and loopback.
#: Narrow link-local exceptions are named by module below.
ALLOWED_URL_HOSTS = frozenset({"agent.robinhood.com", "127.0.0.1", "{LOOPBACK_HOST}"})

# Recovery is the one named module allowed to read the droplet's link-local
# metadata service. It cannot route off-box and receives only instance tags.
MODULE_URL_HOSTS = {
    "serve/recovery.py": frozenset({"169.254.169.254"}),
    # The pinned Codex CLI release is fetched from the CLI's own GitHub releases,
    # SHA-256 verified before anything is written; no Tick host is involved.
    "serve/codex_install.py": frozenset({"github.com"}),
}

_URL_HOST = re.compile(r"https?://([^/\s'\"<>)\\]*)")


def python_files(*packages: str, modules: tuple[str, ...] = LOCAL_MODULES) -> list[Path]:
    files: list[Path] = []
    for package in packages:
        files.extend(sorted((SRC / package).rglob("*.py")))
    files.extend(SRC / module for module in modules)
    assert files, f"no source files found for {packages}"
    return files


def local_only_files() -> list[Path]:
    """The packages that compute locally, minus the broker files that speak MCP."""
    return [
        path
        for path in python_files(*LOCAL_PACKAGES)
        if path.relative_to(SRC).as_posix() not in BROKER_TRANSPORT_MODULES
    ]


def transport_files() -> list[Path]:
    """Every file that may open a socket: the connect ceremony and the MCP seam."""
    return python_files(*TRANSPORT_PACKAGES, modules=BROKER_TRANSPORT_MODULES)


def imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("path", local_only_files(), ids=lambda p: p.name)
def test_the_engine_and_broker_reach_no_network_and_no_model(path: Path):
    """Robinhood data never leaves the user's machine, and there is no hosted LLM path."""
    offending = imported_roots(path) & FORBIDDEN_IMPORTS
    assert not offending, (
        f"{path.relative_to(SRC)} imports {sorted(offending)}. The paper simulation "
        f"and the record are computed locally: account state is never transmitted, "
        f"and no LLM client is constructed against any endpoint here."
    )


@pytest.mark.parametrize("path", python_files(*PRODUCT_PACKAGES), ids=lambda p: p.name)
def test_no_real_security_is_named_in_product_code(path: Path):
    """Tick authors no strategies: no presets, no starters, no example instruments."""
    found = REAL_TICKERS.findall(path.read_text(encoding="utf-8"))
    assert not found, (
        f"{path.relative_to(SRC)} names {sorted(set(found))}. Product code uses "
        f"placeholder symbols (XYZ, ABCD); naming a real security anywhere the "
        f"product surfaces reads as a recommendation."
    )


@pytest.mark.parametrize("path", sorted(SRC.rglob("*.py")), ids=lambda p: str(p.relative_to(SRC)))
def test_a_deterministic_agent_is_never_called_an_ai_agent(path: Path):
    """Accurate naming: spec agents are rule agents; only model agents are model-driven."""
    text = path.read_text(encoding="utf-8").lower()
    assert "ai agent" not in text, (
        f"{path.relative_to(SRC)} says 'AI agent'. A deterministic spec agent is a "
        f"rule agent; a model-driven one is described as model-driven and shows its "
        f"model id."
    )


def test_every_package_the_product_ships_is_named_by_one_of_these_lists():
    """A package nobody listed is unchecked, not exempt — so the lists are total.

    Every scan above runs over a named list of packages. That is deliberate:
    the connect ceremony may open a socket and the engine may not, and a scan
    that could not tell them apart would have to be weakened to the loosest of
    them. The cost is that a package added later is silently outside every
    rule until somebody remembers this file, which is exactly what happened to
    `spec`. So the lists are checked against the tree instead of trusted.
    """
    packages = sorted(
        path.name for path in SRC.iterdir() if path.is_dir() and (path / "__init__.py").exists()
    )
    assert packages == sorted(PRODUCT_PACKAGES), (
        f"{sorted(set(packages) - set(PRODUCT_PACKAGES))} are in src/tick and in none of "
        f"LOCAL_PACKAGES, MODEL_PACKAGES or TRANSPORT_PACKAGES. Put a new package in the "
        f"list that describes what it may reach; do not leave it unscanned."
    )
    modules = sorted(path.name for path in SRC.glob("*.py"))
    assert modules == sorted({"__init__.py", *LOCAL_MODULES})


def test_no_short_side_exists_anywhere_in_the_product():
    """Long only (invariant 9): there is no vocabulary for a short in the tree.

    The refusals are tested where they live — the engine sizing a sell, the
    paper broker, the Robinhood adapter, the model agent's proposal check.
    This is the structural half of the same invariant: a short is not
    expressible, so no future caller can reach one by asking for it by name.
    Prose is not the target (`shortly before the close` is fine); these are
    the identifiers and wire values a short side would have to be spelled as.
    """
    forbidden = re.compile(
        r"\b(sell_short|short_sell|sellShort|shortSell|buy_to_cover|sell_to_open"
        r"|short_position|short_qty|short_shares|shortable|naked_short|borrow_shares)\b",
        re.IGNORECASE,
    )
    for path in sorted(SRC.rglob("*.py")):
        found = forbidden.findall(path.read_text(encoding="utf-8"))
        assert not found, (
            f"{path.relative_to(SRC)} names {sorted(set(found))}. Tick is long-only: a "
            f"sell closes a position and never opens a short, so there is no side, "
            f"field or wire value for one."
        )

    from tick.spec import Side

    assert sorted(side.value for side in Side) == ["buy", "sell"]


def test_the_product_ships_no_strategy_or_market_data_of_its_own():
    """No bundled fixtures, presets or example specs anywhere under `src/tick`."""
    data_files = [
        path
        for path in SRC.rglob("*")
        if path.is_file() and path.suffix in {".json", ".yaml", ".yml", ".csv"}
    ]
    assert data_files == [], (
        f"{[str(p.relative_to(SRC)) for p in data_files]} ship inside the package. "
        f"Test fixtures live under tests/ and are never exposed by the product."
    )


def test_the_fixture_port_has_no_default_directory():
    """A caller must always say which files to read; there is no bundled series."""
    import inspect

    from tick.engine import FixtureMarketData

    signature = inspect.signature(FixtureMarketData.from_directory)
    assert signature.parameters["directory"].default is inspect.Parameter.empty
    assert signature.parameters["now"].default is inspect.Parameter.empty


def test_the_record_is_a_local_file_with_no_transport(monkeypatch, tmp_path):
    """The ledger is written to disk on the user's machine and sent nowhere.

    The import scan above proves no transport is reachable from the package.
    This proves the positive: an append produces a file under the directory it
    was given, and nothing else — no second path, no upload, no queue.
    """
    from datetime import UTC, datetime

    from tick.records import DataSource, Ledger, RecordKind

    home = tmp_path / "tick-home"
    ledger = Ledger(home / "agents" / "one" / "records.jsonl", clock=lambda: datetime.now(UTC))
    ledger.append(RecordKind.NOTE, {"text": "placeholder"}, source=DataSource.ROBINHOOD)

    written = sorted(path.relative_to(home) for path in home.rglob("*") if path.is_file())
    assert [str(path) for path in written] == [
        "agents/one/records.jsonl",
        "agents/one/records.jsonl.lock",
    ]


def test_nothing_in_this_slice_counts_trades_or_assets():
    """Flat fees only: no code counts trades or assets for billing."""
    for path in python_files(*PRODUCT_PACKAGES):
        text = path.read_text(encoding="utf-8").lower()
        for word in ("billing", "invoice", "per_trade_fee", "commission"):
            assert word not in text, f"{path.relative_to(SRC)} mentions {word!r}"


# ----------------------------------------------------------------------
# The compiler: the user's own provider, and no endpoint of Tick's
# ----------------------------------------------------------------------


@pytest.mark.parametrize("path", python_files(*MODEL_PACKAGES, modules=()), ids=lambda p: p.name)
def test_the_compiler_reaches_no_transport_but_the_users_own_provider(path: Path):
    """Bring your own model: the provider SDK, and no other way off the machine.

    The CLI-shaped adapter is the one named exception: it may import
    `subprocess` to run the user's own `codex`, and nothing else on the list.
    """
    permitted = {USERS_PROVIDER_SDK}
    if path.relative_to(SRC).as_posix() in CLI_ADAPTER_MODULES:
        permitted |= CLI_ADAPTER_IMPORTS
    offending = imported_roots(path) & (FORBIDDEN_IMPORTS - permitted)
    assert not offending, (
        f"{path.relative_to(SRC)} imports {sorted(offending)}. The compiler may reach "
        f"the model provider the USER pays for, through that provider's own SDK, and "
        f"nothing else — no second transport, no socket, no subprocess."
    )


def test_only_the_cli_adapter_starts_a_process():
    """One file may shell out, so one file is what a reviewer has to read."""
    importers = [
        path.relative_to(SRC).as_posix()
        for path in sorted(SRC.rglob("*.py"))
        if imported_roots(path) & CLI_ADAPTER_IMPORTS
    ]
    assert sorted(importers) == sorted(PROCESS_MODULES), (
        f"{importers} import subprocess. The Codex adapter and reviewed box launcher are "
        f"the only places Tick runs a command; nothing else may."
    )


def test_only_the_adapter_imports_the_provider_sdk():
    """One file talks to a provider, so one file is what a reviewer has to read."""
    importers = [
        path.relative_to(SRC).as_posix()
        for path in python_files(*MODEL_PACKAGES, modules=())
        if USERS_PROVIDER_SDK in imported_roots(path)
    ]
    assert sorted(importers) == ["agents/anthropic_client.py", "compile/anthropic_client.py"], (
        f"{importers} import {USERS_PROVIDER_SDK}. The SDK is confined to one adapter "
        f"per package; everything else takes an injected client, which is what keeps "
        f"the key, the endpoint and the request inspectable in one place."
    )


def url_hosts(path: Path) -> list[tuple[int, str]]:
    """Every host a URL literal in `path` names, f-string templates included.

    An f-string is reconstructed with its interpolations written back as
    `{name}` placeholders, so `f"http://{LOOPBACK_HOST}:{self.port}/…"` is read
    as the host `{LOOPBACK_HOST}` and has to be named in `ALLOWED_URL_HOSTS`
    like any other. Without that, interpolating the host would be the one way
    to write a URL this scan could not see.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    joined = [node for node in ast.walk(tree) if isinstance(node, ast.JoinedStr)]
    inside = {id(child) for node in joined for child in ast.walk(node) if child is not node}

    def hosts_in(text: str, lineno: int) -> list[tuple[int, str]]:
        return [(lineno, match.split(":")[0]) for match in _URL_HOST.findall(text)]

    found: list[tuple[int, str]] = []
    for node in joined:
        template = ""
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                template += part.value
            elif isinstance(part, ast.FormattedValue):
                name = part.value.id if isinstance(part.value, ast.Name) else "?"
                template += "{" + name + "}"
        found.extend(hosts_in(template, node.lineno))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in inside
        ):
            found.extend(hosts_in(node.value, node.lineno))
    return found


@pytest.mark.parametrize("path", sorted(SRC.rglob("*.py")), ids=lambda p: str(p.relative_to(SRC)))
def test_no_url_in_the_product_names_a_host_but_robinhoods_or_the_loopback(path: Path):
    """Tick operates no data endpoint; the one link-local metadata read is named.

    The scan enumerates the hosts rather than banning URLs outright, which it
    used to: the connect ceremony has to name Robinhood's own MCP and the
    loopback address the authorization code comes back to. Those two are the
    whole list. A URL to anything else — a Tick host, a logging service, an
    analytics beacon — fails here whatever the surrounding comment says.
    """
    relative = path.relative_to(SRC).as_posix()
    allowed = ALLOWED_URL_HOSTS | MODULE_URL_HOSTS.get(relative, frozenset())
    offending = [(lineno, host) for lineno, host in url_hosts(path) if host and host not in allowed]
    assert not offending, (
        f"{relative} names {offending}. Its reviewed literal hosts are {sorted(allowed)}; "
        "add a narrow module exception with a reason or remove the endpoint."
    )


@pytest.mark.parametrize("path", sorted(SRC.rglob("*.py")), ids=lambda p: str(p.relative_to(SRC)))
def test_no_client_anywhere_is_pointed_at_an_endpoint_of_ticks_own(path: Path):
    """No hosted path: nothing redirects a client away from where its SDK points.

    An AST check rather than a grep, so a docstring that discusses `base_url`
    (this codebase has one, deliberately) does not have to be spelled around.
    """
    relative = path.relative_to(SRC).as_posix()
    permitted = {"http_client"} if relative in HTTP_CLIENT_EXEMPT else set()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg in ENDPOINT_KEYWORDS - permitted:
            raise AssertionError(
                f"{relative} line {node.lineno} passes {node.arg!r}. "
                f"Redirecting a client is how a hosted path appears."
            )


@pytest.mark.parametrize(
    "path", python_files(*TRANSPORT_PACKAGES, modules=()), ids=lambda p: p.name
)
def test_the_connect_ceremony_reaches_robinhood_and_the_loopback_and_nothing_else(path: Path):
    """The one package that may open a socket, held to a named list of ways.

    `auth/` speaks HTTP to Robinhood's authorization and MCP endpoints, listens
    on 127.0.0.1 for the redirect, and opens the user's browser. It may not
    build a model client, shell out, or reach a second transport — the token it
    handles is the credential invariant 1 is about.
    """
    offending = imported_roots(path) & (FORBIDDEN_IMPORTS - TRANSPORT_IMPORTS)
    assert not offending, (
        f"{path.relative_to(SRC)} imports {sorted(offending)}. The connect ceremony "
        f"may reach Robinhood and the loopback interface, and nothing else."
    )


def test_the_compiler_never_writes_an_api_key_anywhere():
    """Tick holds no credential (invariant 1), model keys included.

    The CLI-shaped adapter is exempt by name: `codex exec` takes its answer
    schema as a file path, so the adapter writes the intents schema into a
    temporary directory it deletes on the way out. What it writes is pinned
    in tests/agents/test_codex_client.py; it never sees a credential at all.
    """
    for path in python_files(*MODEL_PACKAGES, modules=()):
        if path.relative_to(SRC).as_posix() in CLI_ADAPTER_MODULES:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in ("write_text", "write_bytes"):
                raise AssertionError(
                    f"{path.relative_to(SRC)} line {node.lineno} writes a file. The "
                    f"compiler reads a key from the environment and writes nothing; "
                    f"the CLI is what puts a compiled spec on disk."
                )


# ----------------------------------------------------------------------
# The documentation is a surface too — and so is every line of `--help`
# ----------------------------------------------------------------------


def user_documents() -> list[str]:
    """Every markdown document a user reads, discovered rather than listed.

    The README and everything under `docs/` except the internal records. It is
    a walk rather than a tuple because the tuple was `("docs/WALKTHROUGH.md",)`
    for three slices while the PRD sat beside it naming a real security in an
    example nobody scanned.
    """
    excluded = tuple(ROOT / relative for relative in INTERNAL_DOC_DIRS)
    documents = [ROOT / "README.md"]
    documents.extend(
        path
        for path in sorted((ROOT / "docs").rglob("*.md"))
        if not any(path.is_relative_to(directory) for directory in excluded)
    )
    return [str(path.relative_to(ROOT)) for path in documents]


def test_the_excluded_document_directories_all_exist():
    """An exclusion that names nothing would exempt nothing — or everything."""
    require_private_docs()
    for relative in INTERNAL_DOC_DIRS:
        assert (ROOT / relative).is_dir(), (
            f"{relative} is excluded from the documentation scans and does not exist. "
            f"A stale exclusion hides the day a real surface is moved into it."
        )
    assert "docs/PRD.md" in user_documents()
    assert "docs/WALKTHROUGH.md" in user_documents()
    assert "README.md" in user_documents()
    assert not [name for name in user_documents() if name.startswith("docs/prs/")]


@pytest.mark.parametrize("relative", user_documents())
def test_no_user_facing_document_names_a_real_security(relative: str):
    """Tick authors no strategies — in a walkthrough as much as in `--help`."""
    path = ROOT / relative
    found = REAL_TICKERS.findall(path.read_text(encoding="utf-8"))
    assert not found, (
        f"{relative} names {sorted(set(found))}. Documentation uses placeholder "
        f"symbols (XYZ, ABCD, WXY); a walkthrough that names a real security is a "
        f"recommendation someone will read as one."
    )


@pytest.mark.parametrize("relative", user_documents())
def test_no_user_facing_document_calls_a_deterministic_agent_an_ai_agent(relative: str):
    """A spec agent is a rule agent; a model-driven one shows its model id."""
    text = (ROOT / relative).read_text(encoding="utf-8").lower()
    assert "ai agent" not in text, (
        f"{relative} says 'AI agent'. A deterministic spec agent is a rule agent, and "
        f"a model-driven one is described as model-driven."
    )


def command_paths() -> list[tuple[str, ...]]:
    """Every `--help` a person can reach, walked out of the command tree.

    Discovered, not listed. The hand-written list in `tests/runtime/test_cli.py`
    was written in slice 04 and never grew the connect and broker commands
    slice 06 added, so four help screens went unscanned — the exact failure
    mode a scan exists to prevent. Anything typer can route to is here.
    """
    import typer

    root = typer.main.get_command(app)
    found: list[tuple[str, ...]] = [()]

    def walk(command: object, prefix: tuple[str, ...]) -> None:
        for name, child in sorted(getattr(command, "commands", {}).items()):
            found.append((*prefix, name))
            walk(child, (*prefix, name))

    walk(root, ())
    return found


def help_text(path: tuple[str, ...], home: Path) -> str:
    """`tick … --help`, rendered with TICK_HOME pointed at a temporary directory."""
    result = CliRunner(env={"TICK_HOME": str(home)}).invoke(app, [*path, "--help"])
    assert result.exit_code == 0, result.output
    return result.output


def test_the_help_scan_reaches_every_command_the_cli_exposes():
    """The discovery is the point; a walk that found two screens proves nothing."""
    paths = command_paths()
    for expected in [
        (),
        ("run",),
        ("agent", "new"),
        ("connect", "robinhood"),
        ("broker", "propose"),
        ("broker", "confirm"),
        ("broker", "prove"),
    ]:
        assert expected in paths, f"`tick {' '.join(expected)}` is not in the walked tree"
    assert len(paths) >= 14


@pytest.mark.parametrize("path", command_paths(), ids=lambda p: " ".join(p) or "tick")
def test_no_help_text_anywhere_names_a_real_security(path: tuple[str, ...], tmp_path: Path):
    """`--help` is the surface a starter strategy would appear on first."""
    found = REAL_TICKERS.findall(help_text(path, tmp_path))
    assert not found, (
        f"`tick {' '.join(path)} --help` names {sorted(set(found))}. Help text uses "
        f"placeholders (XYZ, ABCD): an example naming a real security is a "
        f"recommendation with a shell prompt in front of it."
    )


@pytest.mark.parametrize("path", command_paths(), ids=lambda p: " ".join(p) or "tick")
def test_no_help_text_anywhere_calls_a_deterministic_agent_an_ai_agent(
    path: tuple[str, ...], tmp_path: Path
):
    text = help_text(path, tmp_path).lower()
    assert "ai agent" not in text, (
        f"`tick {' '.join(path)} --help` says 'AI agent'. A deterministic agent is a "
        f"rule agent; a model-driven one is described as model-driven."
    )


@pytest.mark.parametrize("path", command_paths(), ids=lambda p: " ".join(p) or "tick")
def test_no_help_text_counts_trades_or_assets_for_billing(path: tuple[str, ...], tmp_path: Path):
    """Flat fees only — nothing the CLI says implies a fee that scales."""
    text = help_text(path, tmp_path).lower()
    for word in ("per-trade fee", "per trade fee", "commission", "assets under management"):
        assert word not in text, f"`tick {' '.join(path)} --help` mentions {word!r}"


def test_the_walkthrough_says_live_was_not_exercised_against_the_real_endpoint():
    """The honest limit, stated where a person about to trade will read it."""
    require_private_docs()
    text = (ROOT / "docs" / "WALKTHROUGH.md").read_text(encoding="utf-8").lower()
    assert "exercised against robinhood's real endpoint" in text
    assert "discovered" in text and "tools/list" in text


def test_the_walkthrough_says_robinhood_can_revoke_the_connection_at_any_time():
    """§29.7(c) is the residual risk, said where a person is about to connect.

    The owner ruled on 2026-09-02 that no written §29.1 consent is sought
    (docs/PRD.md §2.2), so the walkthrough must no longer say an inquiry is
    outstanding — and must still say, before `tick connect robinhood`, that
    Robinhood can revoke the connection for any reason without notice.
    """
    require_private_docs()
    text = (ROOT / "docs" / "WALKTHROUGH.md").read_text(encoding="utf-8").lower()
    assert "§29.7(c)" in text and "revoke" in text and "without" in text and "notice" in text
    assert "has not come back" not in text
    assert "express written consent" not in text


def test_the_walkthrough_limits_broker_data_to_the_users_direct_infrastructure():
    require_private_docs()
    text = (ROOT / "docs" / "WALKTHROUGH.md").read_text(encoding="utf-8").lower()
    assert "directly from this box to this paired phone" in text
    assert "no tick-operated service can read it" in text
    assert "broker-derived rows" in text and "hashes and kinds" in text


def test_no_user_facing_document_counts_trades_or_assets_for_billing():
    """Flat fees only."""
    for relative in user_documents():
        text = (ROOT / relative).read_text(encoding="utf-8").lower()
        for word in ("per-trade fee", "per trade fee", "commission", "assets under management"):
            assert word not in text, f"{relative} mentions {word!r}"
