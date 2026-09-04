"""`tick` — the command line, and the only way a person drives the runtime.

    tick agent interview --provider codex --kind rule
    tick agent draft show <draft-id>
    tick agent adopt <draft-id> --max-cancels 3
    tick agent new "buy 5 shares of XYZ when ..." --out strategy.json
    tick agent add strategy.json --max-cancels 3
    tick agent add model-agent.json --max-cancels 3 --instructions ./my-words.md
    tick agents
    tick run <agent-id> --once --market fixture:./series
    tick run <agent-id> --live --approve each
    tick stop <agent-id>
    tick status <agent-id>
    tick ledger <agent-id> --verify --tail 5
    tick ledger new <agent-id>
    tick connect robinhood --account <agentic-account-id>
    tick broker tools
    tick broker propose
    tick broker confirm --yes-all-reads
    tick broker prove --probe symbol=XYZ

Three things about this file are product decisions rather than plumbing.

**It names no securities.** Not in help text, not in an example, not as a
default. Tick authors no strategies: a "starter strategy" or a sample universe
in `--help` is a recommendation with a shell prompt in front of it. Every
example here uses the placeholder `XYZ`, and the market fixtures under `tests/`
are test material that no command exposes.

**Deterministic agents are rule agents.** The word "AI" does not appear. A spec
agent executes a document mechanically; a model-driven agent is described as
model-driven and shows its model id, in `tick agents`, in `tick status` and in
every notification it produces.

**A model agent runs YOUR instructions.** `tick agent add` on a model-driven
document requires `--instructions <file>` and copies that file into the agent's
directory. Tick ships no default instructions, offers no starter, and completes
nothing: an agent with no words of yours refuses to be created.

**The compiler translates; it never authors.** `tick agent new` turns YOUR
words into a spec using YOUR model API key, and refuses — with the question
to answer — rather than choosing a symbol, a threshold or a limit you did not
give it. It never suggests an instrument.

**There is no real market data source yet.** `--market fixture:<path>` reads
JSON series the user points it at, and that is all there is; `--help` says so
rather than implying a feed exists.

**Live is an act, not a setting.** `--live` is required on EVERY run that
places real orders; without it a run is paper, and a live agent run without it
is recorded as going back to paper. The flip is written into the record before
anything connects, with who and when in it. Standing approval — real orders
placed without asking — needs a second flag of its own, and that
acknowledgement is recorded too.

**Connecting is a ceremony, not a flag.** `tick connect robinhood` prints, in
plain words and before any browser opens, that Robinhood's grant reads EVERY
account you have and that Tick narrows itself to the one you name. It never
prints a token. `tick broker propose` writes an untrusted proposal;
`confirm` authorizes exact tools, and `prove` records what their mapped result
paths actually returned.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import shlex
import signal
import time
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Annotated

import typer
from typer.core import TyperGroup

from tick.agents import (
    PROVIDERS,
    AgentSpec,
    ModelAgent,
    ModelAgentError,
    ModelAgentSpec,
    Provider,
    availability,
    client_for,
    load_agent_spec_file,
)
from tick.auth import (
    ROBINHOOD_MCP_URL,
    AuthError,
    FileTokenStorage,
    LoopbackAuthorization,
    build_oauth_provider,
    disclosure_text,
)
from tick.broker import (
    BrokerError,
    Category,
    DiscoveredTool,
    MCPSession,
    PaperBroker,
    ProfileBroker,
    ProfileProposal,
    ProfileState,
    ProfileTool,
    ToolState,
    confirm_profile,
    diff_profile,
    has_confirmation_note,
    load_profile,
    profile_path,
    propose_profile,
    prove_profile,
    sanction_for,
    save_profile,
    streamable_http_session,
    verify_session_profile,
)
from tick.broker.profile import (
    CANONICALIZER_VERSION,
    CATEGORIZER_VERSION,
    CATEGORY_REGISTRY_VERSION,
    PROFILE_FORMAT_VERSION,
    build_profile,
    mapping_hash,
)
from tick.commons.cli import app as commons_app
from tick.compile import (
    DEFAULT_MODEL,
    AnthropicSpecProposer,
    CompileError,
    CompileRefusal,
    CompileResult,
    compile_text,
)
from tick.engine import (
    CadenceRefused,
    EngineError,
    FixtureMarketData,
    OrderIntent,
    RuleEvaluator,
    check_cadence,
)
from tick.interview import (
    AgentKind,
    InterviewError,
    InterviewSession,
)
from tick.interview import (
    adopt as adopt_draft,
)
from tick.interview import (
    explain as explain_draft,
)
from tick.interview import (
    start as start_interview,
)
from tick.records import (
    DataSource,
    Ledger,
    RecordError,
    RecordKind,
    encode_line,
    ensure_private_dir,
    export_evidence,
    read,
    tick_home,
    utc_clock,
    write_private_file,
)
from tick.runtime import (
    AgentRun,
    ApprovalDecision,
    ApprovalMode,
    ApprovalOutcome,
    ApprovalQueue,
    ApprovalWindow,
    LaunchError,
    MarketClock,
    Mode,
    ModeNotWired,
    RunLease,
    Runner,
    Scheduler,
    TickOutcome,
    TickRuntimeError,
    acknowledge_demotion,
    agent_id_for,
    boot_id,
    check_live_ready,
    check_local_live_ready,
    consume_launch_ticket,
    create_launch_ticket,
    joined_agent_status,
    load_run_lease,
    local_actor,
    parse_approval_window,
    run_doctor,
    save_run_lease,
    state_summary,
    stop_by_signal,
)
from tick.serve import (
    PairingError,
    create_secret,
    pairing_secret_path,
    rotate_secret,
)
from tick.spec import SpecError, StrategySpec, dump_spec

__all__ = ["app", "main"]

#: The longest the run loop sleeps at a stretch, so a `tick stop` is noticed
#: within this many seconds however far off the next scheduled tick is.
STOP_POLL_SECONDS = 5.0

#: What `--market` understands today. There is exactly one scheme.
FIXTURE_SCHEME = "fixture:"

#: How long a broker session may take to establish or to answer. Long, because
#: the handshake includes a person reading a consent screen in a browser; a
#: chosen number rather than a default, and `--timeout` overrides it.
BROKER_TIMEOUT_SECONDS = 300.0

#: How many cancels the discovery commands may make. They make none; the
#: adapter still requires the limit, and a limit of zero is the honest one here.
DISCOVERY_MAX_CANCELS = 0


class _LedgerGroup(TyperGroup):
    """Lets `tick ledger <agent-id>` mean `tick ledger show <agent-id>`.

    `tick ledger new <agent-id>` is a real subcommand, so the group cannot
    simply take an argument. Anything that is not a known subcommand is handed
    to `show`, which is what a person typing an agent id expects.
    """

    def resolve_command(self, ctx, args):
        if args and args[0] not in self.commands and not args[0].startswith("-"):
            return super().resolve_command(ctx, ["show", *args])
        return super().resolve_command(ctx, args)


app = typer.Typer(
    name="tick",
    help=(
        "Tick — the always-on runtime for your agents, on your own machine.\n\n"
        "A rule agent executes a strategy spec you wrote (or compiled from your own "
        "words) mechanically, on a schedule. A model-driven agent asks a model you "
        "name, on your own API key or your own Codex login, with instructions you "
        "wrote, and every order it proposes goes through the same cage. Both run in "
        "paper mode with per-order approval by default and record every decision in "
        "an append-only file under TICK_HOME (default ~/.tick).\n\n"
        "Nothing here trades on its own authority and nothing leaves your machine."
    ),
    no_args_is_help=True,
    add_completion=False,
)
agent_app = typer.Typer(
    help="Interview, compile, inspect drafts, and add agents.",
    no_args_is_help=True,
)
draft_app = typer.Typer(
    help="Inspect an interview draft before it creates an agent.",
    no_args_is_help=True,
)
ledger_app = typer.Typer(
    cls=_LedgerGroup,
    help="Read an agent's record, or start a successor when it will not verify.",
    no_args_is_help=True,
)
connect_app = typer.Typer(
    help="Authorise Tick against a brokerage, on this machine.",
    no_args_is_help=True,
)
disconnect_app = typer.Typer(
    help="Delete this machine's copy of a brokerage grant.",
    no_args_is_help=True,
)
broker_app = typer.Typer(
    help="Read what a connected broker declares, and bind it to what Tick needs.",
    no_args_is_help=True,
)
provider_app = typer.Typer(
    help="The model providers Tick ships an adapter for, and whether this machine can reach one.",
    no_args_is_help=True,
)
pair_app = typer.Typer(
    help="Create or rotate the private credential an app uses to reach this box.",
    no_args_is_help=True,
)


@app.callback()
def _prepend_home_bin() -> None:
    """Make `TICK_HOME/bin` (where `tick provider install codex` places the CLI) visible.

    Every `shutil.which("codex")` in the runtime then finds a box-installed Codex,
    including under systemd, whose PATH omits the user's own bin directories.
    """
    home_bin = _home() / "bin"
    current = os.environ.get("PATH", "")
    if str(home_bin) not in current.split(os.pathsep):
        os.environ["PATH"] = f"{home_bin}{os.pathsep}{current}" if current else str(home_bin)


app.add_typer(provider_app, name="provider")
app.add_typer(pair_app, name="pair")
app.add_typer(agent_app, name="agent")
agent_app.add_typer(draft_app, name="draft")
app.add_typer(ledger_app, name="ledger")
app.add_typer(connect_app, name="connect")
app.add_typer(disconnect_app, name="disconnect")
app.add_typer(broker_app, name="broker")
app.add_typer(commons_app, name="commons")


# ----------------------------------------------------------------------
# Shared plumbing
# ----------------------------------------------------------------------


def _home() -> Path:
    """`TICK_HOME`, read once, from the process environment."""
    return tick_home(os.environ)


def _fail(message: str, *, code: int = 1) -> None:
    typer.secho(message, err=True, fg=typer.colors.RED)
    raise typer.Exit(code=code)


def _load(agent_id: str) -> AgentRun:
    try:
        return AgentRun.load(_home(), agent_id)
    except (TickRuntimeError, SpecError) as exc:
        _fail(str(exc))
        raise  # pragma: no cover - _fail always raises


def _moment(at: str | None) -> datetime:
    """The moment to evaluate as of: `--at`, or the wall clock."""
    if at is None:
        return datetime.now(UTC)
    try:
        parsed = datetime.fromisoformat(at)
    except ValueError:
        _fail(f"--at {at!r} is not an ISO 8601 moment (try 2026-09-01T11:00:00-04:00).")
        raise  # pragma: no cover
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        _fail(
            f"--at {at!r} has no timezone. Market logic runs on Eastern sessions, and a "
            f"moment with no offset is a time in an unstated zone."
        )
    return parsed


def _market(source: str, *, now: datetime) -> FixtureMarketData:
    """Build the market-data port `--market` names. One scheme, and it says so."""
    if not source.startswith(FIXTURE_SCHEME):
        _fail(
            f"--market {source!r} is not understood. Tick has no live market-data "
            f"source yet; the only form is fixture:<path>, reading JSON series files "
            f"you point it at."
        )
    path = Path(source[len(FIXTURE_SCHEME) :]).expanduser()
    try:
        if path.is_dir():
            return FixtureMarketData.from_directory(path, now=now)
        return FixtureMarketData.from_file(path, now=now)
    except EngineError as exc:
        _fail(str(exc))
        raise  # pragma: no cover


def _cash(amount: str) -> Decimal:
    try:
        value = Decimal(amount)
    except InvalidOperation:
        _fail(f"--paper-cash {amount!r} is not an amount (try 10000.00).")
        raise  # pragma: no cover
    if value <= 0:
        _fail(f"--paper-cash {amount} must be greater than zero.")
    return value


def _report(outcome: TickOutcome) -> None:
    if outcome.stopped:
        typer.echo(f"stopped: {outcome.halt_reason}")
        return
    if not outcome.session_open:
        typer.echo(f"the market was closed at {outcome.at.isoformat()}; nothing was read.")
        return
    typer.echo(
        f"tick {outcome.at.isoformat()}: {len(outcome.fills)} filled, "
        f"{len(outcome.not_placed)} not placed, {outcome.records} records written."
    )


# ----------------------------------------------------------------------
# agent interview / draft / adopt / new (compile) / add / agents
# ----------------------------------------------------------------------


@agent_app.command("interview")
def agent_interview(
    provider: Annotated[
        Provider,
        typer.Option(
            "--provider",
            help=(
                "The connected provider that extracts your answers on this machine. "
                "Tick stores no provider credential."
            ),
        ),
    ],
    kind: Annotated[
        AgentKind | None,
        typer.Option(
            "--kind",
            help="Choose 'rule' or 'model' now, or leave it for the interview to ask.",
        ),
    ] = None,
    transcript: Annotated[
        Path | None,
        typer.Option(
            "--transcript",
            help=(
                "Replay one answer per line instead of reading the terminal. A line "
                "containing only 'accept' accepts a pending provider suggestion."
            ),
        ),
    ] = None,
) -> None:
    """Build a draft from your answers; create nothing until `agent adopt`.

    Every field is asked without a default. The provider extracts typed values
    from your own words under the current field schema. A suggestion is first
    discouraged, then shown only after you ask again, and still needs a separate
    `accept` turn before it can enter the draft.
    """
    try:
        draft_id = start_interview(provider, kind)
        session = InterviewSession(_home(), draft_id)
        typer.echo(f"draft {draft_id}")
        if transcript is not None:
            try:
                answers = transcript.expanduser().read_text(encoding="utf-8").splitlines()
            except OSError as exc:
                _fail(
                    f"could not read transcript {transcript}: {exc}. The empty draft remains "
                    f"under TICK_HOME and no agent was created."
                )
                return
            for text in answers:
                if not text.strip():
                    continue
                question = session.ask()
                if question is None:
                    _fail(
                        f"transcript {transcript} has answers after draft {draft_id} became "
                        "complete. Remove the extra lines and replay it; no agent was created."
                    )
                typer.echo(question)
                typer.echo(f"> {text}")
                response = session.answer(text)
                if response != session.ask():
                    typer.echo(response)
            if session.ask() is not None:
                _fail(
                    f"transcript {transcript} ended before draft {draft_id} was complete. "
                    f"Resume the draft by answering: {session.ask()}"
                )
            _show_draft(session.completed_draft)
            return

        while (question := session.ask()) is not None:
            text = typer.prompt(question)
            response = session.answer(text)
            if response != session.ask():
                typer.echo(response)
        _show_draft(session.completed_draft)
    except (InterviewError, ModelAgentError, ValueError) as exc:
        _fail(str(exc), code=2)


@draft_app.command("show")
def agent_draft_show(
    draft_id: Annotated[str, typer.Argument(help="The draft id printed by `agent interview`.")],
) -> None:
    """Show the candidate document in plain words and list every provenance value."""
    try:
        draft = InterviewSession(_home(), draft_id).completed_draft
    except (InterviewError, ValueError) as exc:
        _fail(str(exc), code=2)
        return
    _show_draft(draft)


def _show_draft(draft) -> None:
    """Render only from the validated document, followed by its provenance map."""
    for line in explain_draft(draft):
        typer.echo(line)
    typer.echo("")
    typer.echo("Provenance")
    for field, source in sorted(draft.provenance.items()):
        typer.echo(f"  {field:<32} {source}")
    typer.echo(f"  {'transcript_sha256':<32} {draft.transcript_sha256}")


@agent_app.command("adopt")
def agent_adopt(
    draft_id: Annotated[
        str,
        typer.Argument(help="A complete interview draft to create as an agent."),
    ],
    max_cancels: Annotated[
        int,
        typer.Option(
            "--max-cancels",
            help=(
                "How many order cancellations the agent may attempt per session. "
                "Required; there is no default."
            ),
        ),
    ],
    approve: Annotated[
        ApprovalMode | None,
        typer.Option(
            "--approve",
            help=(
                "Replace the approval mode explicitly. If omitted, use the mode you "
                "answered in the interview; per-order approval is the safe choice."
            ),
        ),
    ] = None,
) -> None:
    """Create one paper agent from a complete draft and record its provenance.

    The first ledger row is the adoption note. An incomplete or already adopted
    draft refuses without changing an agent or its record.
    """
    try:
        run = adopt_draft(draft_id, max_cancels=max_cancels, approval=approve)
    except (InterviewError, TickRuntimeError, SpecError, RecordError, ValueError) as exc:
        _fail(str(exc), code=2)
        return
    typer.echo(run.agent_id)
    typer.echo("adopted in paper mode; its first record carries the draft provenance.")


@agent_app.command("new")
def agent_new(
    text: Annotated[
        str,
        typer.Argument(
            help=(
                "What you want the agent to do, in your own words. Name the symbols "
                "and give every number — how much, how often, and the limits it runs "
                "under. The compiler translates what you wrote; it will not choose a "
                "security, a threshold or a limit for you."
            )
        ),
    ],
    model: Annotated[
        str,
        typer.Option(
            "--model",
            help=(
                "Which model does the translation. It runs on YOUR account: the key "
                "comes from ANTHROPIC_API_KEY in this shell and is never stored. The "
                "model id used is printed with the result."
            ),
        ),
    ] = DEFAULT_MODEL,
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            help="Where to write the compiled spec. Default: a file under TICK_HOME/compiled.",
        ),
    ] = None,
    add: Annotated[
        bool,
        typer.Option("--add", help="Also register the compiled spec as a rule agent."),
    ] = False,
    max_cancels: Annotated[
        int | None,
        typer.Option(
            "--max-cancels",
            help=(
                "Required with --add: how many order cancellations the agent may "
                "attempt per session. There is no default; repeated cancellation is a "
                "pattern brokers terminate connectivity over."
            ),
        ),
    ] = None,
    approve: Annotated[
        ApprovalMode,
        typer.Option(
            "--approve",
            help=(
                "With --add: 'each' (the default) asks you before every order; 'standing' does not."
            ),
        ),
    ] = ApprovalMode.EACH,
) -> None:
    """Compile your own words into a rule agent's strategy spec.

    The spec is a document you can read: a universe, a cadence, `when -> then`
    rules, and the cage of limits the runtime enforces. Read the explanation
    before you run it — it is rendered from the compiled document, so it says
    what the agent will actually do and what it cannot see.

    If your words are missing something — which symbols, what size, what the
    limits are — nothing is compiled and the questions to answer are printed
    instead. That is the compiler working: it translates, it does not invent.
    """
    if add and max_cancels is None:
        _fail(
            "--add needs --max-cancels: an agent's cancel guard has no default, "
            "because a limit nobody chose is not a limit. Nothing was written."
        )
        return
    try:
        proposer = AnthropicSpecProposer.for_environment()
        outcome = compile_text(text, proposer, model=model)
    except CompileError as exc:
        _fail(str(exc), code=2)
        return

    if isinstance(outcome, CompileRefusal):
        _report_refusal(outcome)
        return

    path = _write_compiled(outcome.spec, out)
    _report_compiled(outcome, path)
    if add and max_cancels is not None:
        _add_compiled(outcome.spec, max_cancels=max_cancels, approve=approve)


def _report_refusal(refusal: CompileRefusal) -> None:
    """Print the questions and exit non-zero. Nothing is written."""
    typer.secho(
        "Nothing was compiled: your words are missing something the spec needs, "
        "and Tick will not choose it for you.",
        err=True,
        fg=typer.colors.YELLOW,
    )
    for question in refusal.questions:
        typer.secho(f"  - {question}", err=True)
    typer.secho(
        f"Answer these in the text and run it again. ({refusal.model}, "
        f"refused by: {refusal.origin})",
        err=True,
    )
    raise typer.Exit(code=2)


def _write_compiled(spec: StrategySpec, out: Path | None) -> Path:
    """Write the spec where the user asked, or under TICK_HOME/compiled."""
    if out is not None:
        path = out.expanduser()
        if str(path.parent) not in ("", "."):
            path.parent.mkdir(parents=True, exist_ok=True)
    else:
        directory = _home() / "compiled"
        ensure_private_dir(directory)
        path = directory / f"{agent_id_for(spec)}.json"
    try:
        path.write_text(dump_spec(spec), encoding="utf-8")
    except OSError as exc:
        _fail(f"could not write the compiled spec to {path}: {exc}")
    return path


def _report_compiled(outcome: CompileResult, path: Path) -> None:
    """The explanation, then the path. Rendered from the document, not the model."""
    spec = outcome.spec
    typer.echo(f"{spec.name} — {len(spec.rules)} rule(s) over {len(spec.universe)} symbol(s)")
    typer.echo(
        f"compiled by {outcome.model} in {outcome.attempts} attempt(s). Every symbol "
        f"and every number in it came from your own words."
    )
    for item in outcome.explanation:
        typer.echo("")
        typer.echo(f"  {item.rule_id}: {item.what_it_does}")
        for blind in item.what_it_cannot_know:
            typer.echo(f"    it cannot know {blind}")
    typer.echo("")
    typer.echo(str(path))


def _add_compiled(spec: StrategySpec, *, max_cancels: int, approve: ApprovalMode) -> None:
    """Register a just-compiled spec as an agent, on the same checks as `agent add`."""
    try:
        check_cadence(spec.cadence)
    except CadenceRefused as exc:
        _fail(str(exc))
        return
    try:
        run = AgentRun.create(
            _home(),
            spec,
            max_cancels_per_session=max_cancels,
            approval=approve,
            created_at=datetime.now(UTC),
            instructions=None,
        )
    except TickRuntimeError as exc:
        _fail(str(exc))
        return
    typer.echo(run.agent_id)
    typer.echo(f"paper mode; run it with: tick run {run.agent_id} --once --market fixture:<path>")


@agent_app.command("add")
def agent_add(
    spec_file: Annotated[
        Path,
        typer.Argument(
            help=(
                "An agent document (JSON) you wrote or compiled: a strategy spec for a "
                "rule agent, or a model-driven agent's document."
            )
        ),
    ],
    max_cancels: Annotated[
        int,
        typer.Option(
            "--max-cancels",
            help=(
                "How many order cancellations this agent may attempt per session. "
                "Required: repeated cancellation is a pattern brokers terminate "
                "connectivity over, so there is no default."
            ),
        ),
    ],
    approve: Annotated[
        ApprovalMode,
        typer.Option(
            "--approve",
            help=(
                "'each' (the default) asks you before every order; 'standing' does not "
                "ask. Standing is your choice; some providers' terms ask for human review "
                "of financial decisions — see `tick provider check`."
            ),
        ),
    ] = ApprovalMode.EACH,
    instructions: Annotated[
        Path | None,
        typer.Option(
            "--instructions",
            help=(
                "Required for a model-driven agent: the file of YOUR instructions it "
                "runs. It is copied into the agent's directory. Tick ships no default "
                "instructions and will not write one for you. A rule agent takes none."
            ),
        ),
    ] = None,
) -> None:
    """Validate an agent document, copy it under TICK_HOME, and print its id.

    The copy is what the agent runs, and it is written read-only: an agent
    executes the document it was created for. A changed document is a new agent.

    A model-driven document also needs `--instructions <file>` — the words YOU
    wrote for it. They are copied in beside the document and can be edited
    afterwards; the hash of what was actually sent goes into every decision the
    agent records.
    """
    try:
        spec = load_agent_spec_file(spec_file)
    except SpecError as exc:
        _fail(str(exc))
        return
    words = _instructions_text(spec, instructions)
    try:
        # The cadence floor is a runtime property, not a spec one, so a spec
        # below it is valid and unrunnable. Say so here rather than at 09:31.
        check_cadence(spec.cadence)
    except CadenceRefused as exc:
        _fail(str(exc))
        return
    try:
        run = AgentRun.create(
            _home(),
            spec,
            max_cancels_per_session=max_cancels,
            approval=approve,
            created_at=datetime.now(UTC),
            instructions=words,
        )
    except TickRuntimeError as exc:
        _fail(str(exc))
        return
    typer.echo(run.agent_id)
    typer.echo(f"{spec.name} — {_describe_agent(spec)}")
    if isinstance(spec, ModelAgentSpec):
        typer.echo(
            f"provider {spec.provider}: check this machine can reach it with "
            f"`tick provider check {spec.provider}`."
        )
    typer.echo(f"paper mode; run it with: tick run {run.agent_id} --once --market fixture:<path>")


def _instructions_text(spec: AgentSpec, path: Path | None) -> str | None:
    """The user's own instructions, or `None`, or a refusal that says which.

    Read here rather than in `AgentRun.create` so the file's own problems — it
    is missing, it is a directory, it is empty — are reported against the path
    the person typed, before anything is written.
    """
    if isinstance(spec, ModelAgentSpec) and path is None:
        _fail(
            f"{spec.name!r} is a model-driven agent and runs the instructions YOU write. "
            f"Pass --instructions <file>. Tick ships none and will not write one; nothing "
            f"was created."
        )
    if path is None:
        return None
    if not isinstance(spec, ModelAgentSpec):
        _fail(
            f"{spec.name!r} is a rule agent: it executes its strategy spec mechanically "
            f"and reads no instructions file. Nothing was created."
        )
    try:
        text = path.expanduser().read_text(encoding="utf-8")
    except OSError as exc:
        _fail(f"could not read the instructions file {path}: {exc}. Nothing was created.")
        raise  # pragma: no cover - _fail always raises
    if not text.strip():
        _fail(f"{path} is empty. A model agent runs the words you wrote; nothing was created.")
    return text


def _describe_agent(spec: AgentSpec) -> str:
    """One line saying what kind of agent this is, in the words the audit requires."""
    symbols = f"{len(spec.universe)} symbol(s)"
    if isinstance(spec, ModelAgentSpec):
        return f"model-driven agent ({spec.model} via {spec.provider}) over {symbols}"
    return f"rule agent, {len(spec.rules)} rule(s) over {symbols}"


@app.command("agents")
def agents() -> None:
    """List the agents under TICK_HOME — rule agents and model-driven ones."""
    home = _home()
    ids = AgentRun.list_ids(home)
    if not ids:
        typer.echo(f"no agents under {home}. Add one with: tick agent add <spec.json>")
        return
    for agent_id in ids:
        try:
            summary = state_summary(AgentRun(home, agent_id))
        except (TickRuntimeError, SpecError, RecordError) as exc:
            # One unreadable agent must not hide the rest, and least of all
            # must it hide the `tick stop` the user came here to run.
            typer.echo(f"{agent_id}  UNREADABLE  {exc}")
            continue
        flag = " STOPPED" if summary["stopped"] else ""
        kind = "rule" if summary["model"] is None else f"model:{summary['model']}"
        typer.echo(
            f"{agent_id}  {summary['mode']}/{summary['approval']}  {kind}  "
            f"{summary['cadence']}  {summary['name']}{flag}"
        )


# ----------------------------------------------------------------------
# run
# ----------------------------------------------------------------------


@app.command("run")
def run(
    agent_id: Annotated[str, typer.Argument(help="The agent id printed by `tick agent add`.")],
    market: Annotated[
        str | None,
        typer.Option(
            "--market",
            help=(
                "Where prices come from in a PAPER run, and required for one. The only "
                "form is fixture:<path> — a directory or file of JSON bar series you "
                "supply. Tick ships no market data. A live run takes no --market: it "
                "prices from the broker it is trading through."
            ),
        ),
    ] = None,
    paper_cash: Annotated[
        str | None,
        typer.Option(
            "--paper-cash",
            help=(
                "What the simulated account starts a PAPER run with, e.g. 10000.00, and "
                "required for one: a simulation funded by a default is a number nobody "
                "chose. A live run takes none — the account is real."
            ),
        ),
    ] = None,
    once: Annotated[bool, typer.Option("--once", help="Run one tick and exit.")] = False,
    approval_window: Annotated[
        str | None,
        typer.Option(
            "--approval-window",
            help=(
                "How long EACH approval remains valid, e.g. 300s. Required whenever "
                "this launch uses per-order approval."
            ),
        ),
    ] = None,
    approve: Annotated[
        ApprovalMode | None,
        typer.Option(
            "--approve",
            help=("Change the agent's approval mode: 'each' asks per order, 'standing' does not."),
        ),
    ] = None,
    live: Annotated[
        bool,
        typer.Option(
            "--live",
            help=(
                "Place REAL orders in your Agentic account, through the grant on this "
                "machine. Needs `tick connect robinhood` and a confirmed, proven broker "
                "profile. The "
                "flip is recorded; without this flag every run is paper."
            ),
        ),
    ] = False,
    live_standing_ok: Annotated[
        bool,
        typer.Option(
            "--live-standing-ok",
            help=(
                "Required to run LIVE with standing approval — real orders placed "
                "without asking you first. A second flag on purpose: --live alone will "
                "not do it, and the acknowledgement is recorded."
            ),
        ),
    ] = False,
    server_url: Annotated[
        str | None,
        typer.Option(
            "--server-url",
            help="Must exactly match profile.server when supplied; the profile server is default.",
        ),
    ] = None,
    timeout: Annotated[
        float,
        typer.Option("--timeout", help="Seconds a live run waits on the broker."),
    ] = BROKER_TIMEOUT_SECONDS,
    at: Annotated[
        str | None,
        typer.Option(
            "--at",
            help=(
                "Evaluate as of this ISO 8601 moment instead of now, and read the "
                "fixture series as of the same moment. For replaying a day; "
                "requires --once."
            ),
        ),
    ] = None,
    run_id: Annotated[
        str | None,
        typer.Option("--run-id", hidden=True, help="Run identity supplied by a launcher."),
    ] = None,
    launch_source: Annotated[
        str,
        typer.Option("--launch-source", hidden=True, help="cli, api, or supervisor."),
    ] = "cli",
    launch_ticket: Annotated[
        Path | None,
        typer.Option("--launch-ticket", hidden=True, help="One-use live launch ticket."),
    ] = None,
) -> None:
    """Tick an agent — one tick with --once, otherwise on its cadence.

    A rule agent executes its spec mechanically. A model-driven agent asks its
    model once per tick, on YOUR API key, with YOUR instructions — and every
    intent it proposes then meets the same cage.

    Paper is the default on every path. Every tick verifies the agent's record
    before it does anything else; if the record will not verify the run stops,
    places nothing, and tells you which record failed and what to run next.
    """
    agent = _load(agent_id)
    if at is not None and not once:
        _fail("--at replays a single moment, so it requires --once.")
    now = _moment(at)

    if approve is not None:
        agent.save_state(agent.state.with_approval(approve))
    if launch_source not in {"cli", "api", "supervisor"}:
        _fail("--launch-source must be cli, api, or supervisor. Nothing was started.", code=2)
    window: ApprovalWindow | None = None
    if approval_window is not None:
        try:
            window = parse_approval_window(approval_window)
        except ValueError as exc:
            _fail(str(exc), code=2)
    if agent.state.approval is ApprovalMode.EACH and window is None and not once:
        _fail(
            "per-order approval needs --approval-window, for example --approval-window "
            "300s. Nothing was started; choose the window for this run.",
            code=2,
        )

    supplied_run_id = run_id
    run_id = run_id or secrets.token_hex(12)
    current_boot = boot_id()
    if live and (market is not None or paper_cash is not None):
        _fail(
            "a live run takes no --market and no --paper-cash: it prices from the "
            "broker it is trading through. Nothing was connected or placed.",
            code=2,
        )
    if live and agent.state.approval is ApprovalMode.STANDING and not live_standing_ok:
        _fail(
            "this agent has standing approval, so live needs --live-standing-ok as a "
            "second explicit flag. Nothing was connected or placed.",
            code=2,
        )
    if live and launch_ticket is None:
        if supplied_run_id is not None:
            _fail(
                "a launched live child needs --launch-ticket as well as --live. Nothing "
                "was connected or placed; launch live again.",
                code=2,
            )
        try:
            launch_ticket, fallback = create_launch_ticket(
                _home(),
                agent_id=agent.agent_id,
                run_id=run_id,
                approval_mode=agent.state.approval,
                standing_ok=live_standing_ok,
                created_at=now,
                env=os.environ,
                current_boot_id=current_boot,
            )
        except (OSError, ValueError) as exc:
            _fail(
                f"the one-use live ticket could not be created ({exc}). Nothing was "
                "connected or placed; correct the launch and try again.",
                code=2,
            )
        if fallback:
            typer.echo(
                "warning: no boot-volatile runtime directory was available; the ticket is "
                "under TICK_HOME and still checks the boot id before use."
            )
    if live:
        assert launch_ticket is not None
        try:
            consume_launch_ticket(
                launch_ticket,
                agent_id=agent.agent_id,
                run_id=run_id,
                approval_mode=agent.state.approval,
                standing_ok=live_standing_ok,
                current_boot_id=current_boot,
            )
        except LaunchError as exc:
            _fail(exc.reason, code=2)

    if supplied_run_id is not None:
        previous_lease = load_run_lease(_home(), agent.agent_id)
        if previous_lease is not None and previous_lease.run_id == run_id:
            previous_run_id = previous_lease.previous_run_id
            previous_run_mode = previous_lease.previous_run_mode
            previous_run_boot_id = previous_lease.previous_run_boot_id
        else:
            previous_run_id = previous_lease.run_id if previous_lease is not None else None
            previous_run_mode = previous_lease.mode if previous_lease is not None else None
            previous_run_boot_id = previous_lease.boot_id if previous_lease is not None else None
        _record_run_started(
            agent,
            now=now,
            run_id=run_id,
            boot=current_boot,
            live=live,
            launch_source=launch_source,
            previous_run_id=previous_run_id,
            previous_run_mode=previous_run_mode,
        )
        save_run_lease(
            _home(),
            RunLease(
                agent_id=agent.agent_id,
                run_id=run_id,
                boot_id=current_boot,
                pid=os.getpid(),
                mode=Mode.LIVE if live else Mode.PAPER,
                approval=agent.state.approval,
                launch_source=launch_source,
                started_at=now,
                previous_run_id=previous_run_id,
                previous_run_mode=previous_run_mode,
                previous_run_boot_id=previous_run_boot_id,
            ),
        )

    clock = MarketClock.for_2026()
    session: MCPSession | None = None
    if live:
        data, broker, session = _live_broker(
            agent,
            now=now,
            server_url=server_url,
            timeout=timeout,
            standing_ok=live_standing_ok,
            market=market,
            paper_cash=paper_cash,
        )
        sources = (DataSource.ROBINHOOD, DataSource.ROBINHOOD)
    else:
        _record_return_to_paper(agent, now=now)
        data, broker = _paper_broker(agent, now=now, market=market, paper_cash=paper_cash)
        sources = (DataSource.FIXTURE, DataSource.PAPER)

    runner = Runner(
        stamp=utc_clock,
        market_source=sources[0],
        broker_source=sources[1],
        evaluator=_evaluator_for(agent),
    )

    effective_approver = _ask
    if window is not None:
        effective_approver = _queued_approver(agent, window=window, run_id=run_id, terminal=_ask)

    prior_signals = {signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT)}

    def received(signum, _frame) -> None:
        stop_by_signal(agent, signal_name=signal.Signals(signum).name, at=datetime.now(UTC))
        raise KeyboardInterrupt

    for signum in prior_signals:
        signal.signal(signum, received)

    try:
        agent.require_verified_ledger()
        _record_pending_profile_observation(agent, now=now)
        if not live and not agent.stop_requested():
            # A stopped agent's tick records its stop and nothing else; opening
            # a simulated account it will never trade would be noise in the record.
            _note_paper_account(agent, broker, now)
        if once:
            outcomes = [
                runner.tick(
                    agent,
                    market=data,
                    broker=broker,
                    clock=clock,
                    notify=typer.echo,
                    approve=effective_approver,
                    now=now,
                )
            ]
        else:
            outcomes = runner.run(
                agent,
                market=data,
                broker=broker,
                clock=clock,
                scheduler=Scheduler(clock),
                notify=typer.echo,
                approve=effective_approver,
                now=lambda: datetime.now(UTC),
                sleep=time.sleep,
                poll_seconds=STOP_POLL_SECONDS,
                max_ticks=None,
            )
    except ModeNotWired as exc:
        _fail(str(exc), code=2)
        return
    except (TickRuntimeError, CadenceRefused, RecordError) as exc:
        _fail(str(exc))
        return
    except KeyboardInterrupt:  # pragma: no cover - interactive
        typer.echo("interrupted; nothing further was placed.")
        raise typer.Exit(code=130) from None
    finally:
        # The connection closes on every path out of the run, including the
        # failure paths: a session left open is a socket Robinhood keeps
        # counting against a limit they do not publish.
        if session is not None:
            session.close()
        for signum, previous in prior_signals.items():
            signal.signal(signum, previous)

    for outcome in outcomes:
        _report(outcome)
    if any(outcome.halted for outcome in outcomes):
        raise typer.Exit(code=1)


def _paper_broker(
    agent: AgentRun,
    *,
    now: datetime,
    market: str | None,
    paper_cash: str | None,
) -> tuple[FixtureMarketData, PaperBroker]:
    """The local simulation, and the fixtures that price it.

    Both options are required for a paper run and neither has a default: a
    simulation funded by a number nobody chose, priced from a series nobody
    named, is a result that describes nothing.
    """
    if market is None:
        _fail(
            "a paper run needs --market fixture:<path>. Tick ships no market data, so "
            "there is nothing to fall back to."
        )
    if paper_cash is None:
        _fail(
            "a paper run needs --paper-cash, e.g. --paper-cash 10000.00. A simulation "
            "funded by a default is a number nobody chose."
        )
    assert market is not None and paper_cash is not None  # _fail raises
    data = _market(market, now=now)
    return data, PaperBroker(
        data,
        starting_cash=_cash(paper_cash),
        max_cancels=agent.state.max_cancels_per_session,
    )


def _record_run_started(
    agent: AgentRun,
    *,
    now: datetime,
    run_id: str,
    boot: str,
    live: bool,
    launch_source: str,
    previous_run_id: str | None,
    previous_run_mode: Mode | None,
) -> None:
    """Write this process's first durable run record before any broker opens."""
    try:
        agent.require_verified_ledger()
        agent.ledger(clock=utc_clock).append(
            RecordKind.NOTE,
            {
                "event": "run_started",
                "mode": "live_armed" if live else "paper_by_default",
                "run_id": run_id,
                "boot_id": boot,
                "launch_source": launch_source,
                "approval": agent.state.approval.value,
                "previous_run_id": previous_run_id,
                "previous_run_mode": previous_run_mode.value if previous_run_mode else None,
                "at": now,
            },
            source=DataSource.RUNTIME,
        )
    except (TickRuntimeError, RecordError) as exc:
        _fail(f"{exc} Nothing was connected or placed.")


def _live_broker(
    agent: AgentRun,
    *,
    now: datetime,
    server_url: str | None,
    timeout: float,
    standing_ok: bool,
    market: str | None,
    paper_cash: str | None,
) -> tuple[ProfileBroker, ProfileBroker, MCPSession]:
    """Everything `--live` needs, in the order that keeps the safe state safe.

    The order is the point, and each step is before the next for a reason:

    1. **The paper-only options are refused.** A live run priced from a file
       while orders go to a real account is the worst pairing in the product.
    2. **The kill switch is checked before anything opens.** A stopped agent
       does not connect at all — invariant 8 reaching further back than the
       tick loop, because the cheapest connection to a broker who may terminate
       you for usage they judge excessive is the one you never made.
    3. **The ledger must verify.** Nothing is appended, and nothing is placed,
       on a chain that does not verify.
    4. **Standing approval needs its own flag.** Real orders placed without
       asking is a second decision, so it takes a second switch, and the
       acknowledgement is recorded.
    5. **The machine must be configured.** Local grant, profile, confirmation
       note and proofs are checked before a socket exists.
    6. **The profile server and complete inventory are verified.** Per-tool
       dependencies must be exact matches before anything is armed.
    7. **The flip is recorded, then announced.** The `mode_change` and
       `live_armed` evidence is written after session verification and before
       the first evaluation or order.

    The broker is returned twice on purpose: `ProfileBroker` is both the
    market-data port and the `BrokerPort` for a live run, so the price an order
    is sized at and the venue it is placed on are the same connection.
    """
    if market is not None or paper_cash is not None:
        _fail(
            "a live run takes no --market and no --paper-cash: it prices from the broker "
            "it is trading through, and the account is real. Nothing was placed.",
            code=2,
        )
    if agent.stop_requested():
        _fail(
            f"agent {agent.agent_id} is stopped: {agent.stop_reason()}. Nothing was "
            f"connected and nothing was placed. Remove {agent.stop_path} to let it run "
            f"again."
        )
    try:
        agent.require_verified_ledger()
    except (TickRuntimeError, RecordError) as exc:
        _fail(str(exc))

    approval = agent.state.approval
    if approval is ApprovalMode.STANDING and not standing_ok:
        _fail(
            "this agent has standing approval, so a live run would place real orders "
            "without asking you first. That takes a second, explicit flag: add "
            "--live-standing-ok if you mean it, or run it with --approve each. Nothing "
            "was connected and nothing was placed.",
            code=2,
        )

    readiness = check_local_live_ready(_home(), approval_mode=approval)
    if not readiness.ready:
        typer.secho("live trading is not set up on this machine:", err=True, fg=typer.colors.RED)
        for step in readiness.missing:
            typer.secho(f"  - {step}", err=True)
        _fail("Nothing was connected and nothing was placed.", code=2)
    assert readiness.profile is not None
    profile = readiness.profile
    if server_url is not None and server_url != profile.server:
        _fail(
            f"--server-url {server_url} does not match profile.server {profile.server}. "
            "The whole session is refused; use the confirmed profile server.",
            code=2,
        )
    session = _with_broker_session(
        profile.server, port=0, timeout_seconds=timeout, open_browser=False
    )
    try:
        verified = verify_session_profile(
            profile,
            session,
            server=profile.server,
            account_id=profile.account_id,
            confirmation_recorded=has_confirmation_note(_home(), profile.profile_hash),
        )
        _persist_session_observation(_home(), agent, profile, verified, now=now)
        session_readiness = check_live_ready(_home(), verified, approval_mode=approval)
    except BrokerError as exc:
        session.close()
        _fail(str(exc), code=2)
        raise AssertionError("unreachable") from exc
    if not session_readiness.ready:
        session.close()
        typer.secho("live dependencies are not ready:", err=True, fg=typer.colors.RED)
        for step in session_readiness.missing:
            typer.secho(f"  - {step}", err=True)
        _fail("Nothing was armed and nothing was placed.", code=2)

    _announce_live(agent, profile.account_id, approval=approval, standing_ok=standing_ok)
    _record_live_arming(
        agent,
        now=now,
        account_id=profile.account_id,
        approval=approval,
        standing_ok=standing_ok,
        profile_hash=profile.profile_hash,
        inventory_hash=verified.inventory_hash,
        profile_sanction=profile.sanction,
    )
    broker = ProfileBroker(
        verified,
        max_cancels=agent.state.max_cancels_per_session,
        kill_switch=agent.stop_requested,
        approval_mode=approval.value,
    )
    return broker, broker, session


def _persist_session_observation(
    home: Path,
    agent: AgentRun,
    profile,
    verified,
    *,
    now: datetime,
) -> None:
    """Persist status evidence; the in-memory binding remains authoritative on failure."""
    differences = diff_profile(profile, tuple(verified.contracts.values()))
    mapped_drift = any(
        state is ToolState.DRIFTED
        for name, state in verified.states.items()
        if name in profile.tools and profile.tools[name].category.callable
    )
    observed = build_profile(
        server=profile.server,
        account_id=profile.account_id,
        tools=profile.tools,
        inventory_hash=profile.inventory_hash,
        data_class=profile.data_class,
        sanction=profile.sanction,
        profile_format_version=profile.profile_format_version,
        canonicalizer_version=profile.canonicalizer_version,
        category_registry_version=profile.category_registry_version,
        state=ProfileState.DRIFTED if mapped_drift else ProfileState.CONFIRMED,
        observed_inventory_hash=verified.inventory_hash,
        drift=differences,
    )
    try:
        save_profile(home, observed)
        if differences:
            Ledger(agent.ledger_path, clock=lambda: now).append(
                RecordKind.NOTE,
                {
                    "event": "broker_profile_drifted",
                    "profile_hash": profile.profile_hash,
                    "inventory_hash": verified.inventory_hash,
                    "at": now,
                    "diff": [item.model_dump(mode="json") for item in differences],
                    "text": (
                        "the broker inventory differs from the confirmed profile; mapped "
                        "changes remain refused and new tools remain unmapped"
                    ),
                },
                source=DataSource.RUNTIME,
            )
    except (OSError, RecordError):
        # Authorization lives in VerifiedSessionProfile, never in this write.
        # A failed observation write cannot reopen a changed tool.
        return


def _record_pending_profile_observation(agent: AgentRun, *, now: datetime) -> None:
    """Carry a stored inventory difference into the next agent run's ledger once."""
    try:
        profile = load_profile(_home())
    except BrokerError:
        return
    if profile is None or not profile.drift or profile.observed_inventory_hash is None:
        return
    already_recorded = any(
        record.kind is RecordKind.NOTE
        and record.payload.get("event") == "broker_profile_drifted"
        and record.payload.get("profile_hash") == profile.profile_hash
        and record.payload.get("inventory_hash") == profile.observed_inventory_hash
        for record in read(agent.ledger_path)
    )
    if already_recorded:
        return
    Ledger(agent.ledger_path, clock=lambda: now).append(
        RecordKind.NOTE,
        {
            "event": "broker_profile_drifted",
            "profile_hash": profile.profile_hash,
            "inventory_hash": profile.observed_inventory_hash,
            "at": now,
            "diff": [item.model_dump(mode="json") for item in profile.drift],
            "text": (
                "the stored broker inventory observation differs from the confirmed "
                "profile; mapped changes remain refused and new tools remain unmapped"
            ),
        },
        source=DataSource.RUNTIME,
    )


def _announce_live(
    agent: AgentRun, account_id: str, *, approval: ApprovalMode, standing_ok: bool
) -> None:
    """The plain warning after verification and before any order.

    Plain, and not a wall: the four facts a person needs at the moment they
    flip a switch that spends their money.
    """
    typer.secho("LIVE MODE — real orders, real money.", err=True, fg=typer.colors.YELLOW)
    typer.echo(f"  agent {agent.agent_id} will trade account {account_id} and no other.")
    if approval is ApprovalMode.EACH:
        typer.echo("  you will be asked before every order.")
    else:
        typer.echo(
            "  standing approval: orders are placed WITHOUT asking you "
            f"(acknowledged with --live-standing-ok: {standing_ok})."
        )
    typer.echo(f"  stop it at any time with: tick stop {agent.agent_id}")
    typer.echo("  Tick places long orders only; a sell closes a position it already holds.")


def _record_live_arming(
    agent: AgentRun,
    *,
    now: datetime,
    account_id: str,
    approval: ApprovalMode,
    standing_ok: bool,
    profile_hash: str,
    inventory_hash: str,
    profile_sanction: str,
) -> None:
    """Write the flip and arming after session verification, before evaluation.

    Two records, because they answer different questions. `mode_change` is
    written only when the mode actually changes and says who flipped it and
    when — the switch being recorded IS invariant 2. The `live_armed` note is
    written on every live run and says what this particular run was allowed to
    do, which a `mode_change` from three weeks ago cannot.
    """
    ledger = Ledger(agent.ledger_path, clock=utc_clock)
    actor = local_actor(os.environ)
    state = agent.state
    lease = load_run_lease(agent.home, agent.agent_id)
    supervised = lease is not None and lease.pid == os.getpid() and lease.mode is Mode.LIVE
    try:
        if state.mode is not Mode.LIVE and not supervised:
            ledger.append(
                RecordKind.MODE_CHANGE,
                {
                    "event": "mode_change",
                    "from": state.mode.value,
                    "to": Mode.LIVE.value,
                    "by": actor,
                    "at": now,
                    "account_id": account_id,
                    "text": (
                        "this agent was switched to live from the command line on this "
                        "machine. Orders it places from here are real."
                    ),
                },
                source=DataSource.RUNTIME,
            )
            agent.save_state(state.with_mode(Mode.LIVE))
        ledger.append(
            RecordKind.NOTE,
            {
                "event": "live_armed",
                "by": actor,
                "at": now,
                "account_id": account_id,
                "approval": approval.value,
                "standing_approval_acknowledged": standing_ok,
                "max_cancels_per_session": state.max_cancels_per_session,
                "profile_hash": profile_hash,
                "inventory_hash": inventory_hash,
                "profile_sanction": profile_sanction,
            },
            source=DataSource.RUNTIME,
        )
    except (TickRuntimeError, RecordError) as exc:
        _fail(str(exc))


def _record_return_to_paper(agent: AgentRun, *, now: datetime) -> None:
    """A run without `--live` puts a live agent back in paper, and records it.

    Paper is the default on EVERY path, which means the absence of the flag is
    itself an instruction rather than a continuation. The de-escalation is
    recorded for the same reason the escalation is: the record answers "what
    mode was this agent in, and since when", and a silent step down leaves that
    question unanswerable.
    """
    state = agent.state
    lease = load_run_lease(agent.home, agent.agent_id)
    if lease is not None and lease.pid == os.getpid() and lease.mode is Mode.PAPER:
        return
    if state.mode is Mode.PAPER:
        return
    try:
        agent.require_verified_ledger()
        Ledger(agent.ledger_path, clock=utc_clock).append(
            RecordKind.MODE_CHANGE,
            {
                "event": "mode_change",
                "from": state.mode.value,
                "to": Mode.PAPER.value,
                "by": local_actor(os.environ),
                "at": now,
                "text": (
                    "this run was started without --live, so the agent is back in the "
                    "simulation. Live is an act, not a setting that persists."
                ),
            },
            source=DataSource.RUNTIME,
        )
    except (TickRuntimeError, RecordError) as exc:
        _fail(str(exc))
        return
    agent.save_state(state.with_mode(Mode.PAPER))


def _note_paper_account(agent: AgentRun, broker: PaperBroker, now: datetime) -> None:
    """Record what the simulated account starts this run holding.

    The paper account lives for the length of one `tick run`; it is not carried
    between invocations. Recording the opening balance means the ledger never
    implies a continuity it does not have.
    """
    state = broker.state()
    Ledger(agent.ledger_path, clock=utc_clock).append(
        RecordKind.NOTE,
        {
            "event": "paper_account_opened",
            "cash": state.cash,
            "positions": {symbol: position.qty for symbol, position in state.positions.items()},
            "at": now,
            "text": (
                "the simulated account for this run. It is not carried over from a "
                "previous run of this agent."
            ),
        },
        source=DataSource.PAPER,
    )


def _evaluator_for(agent: AgentRun):
    """What decides this agent's tick: its own rules, or its own model.

    Which one follows from the agent's document, never from a flag: an agent
    that could be ticked by a different decision procedure than the one it was
    created with would make its record describe something that did not happen.

    A model agent's client is built for the provider its document pins, from
    the USER's own environment — a key, or a login the user made — here and
    nowhere else in the CLI. Tick operates no model endpoint and stores no
    credential, so a missing one is a refusal with the fix in it rather than a
    fallback to anything of ours, and never a different provider.
    """
    spec = agent.spec
    if not isinstance(spec, ModelAgentSpec):
        return RuleEvaluator()
    try:
        instructions = agent.instructions()
    except TickRuntimeError as exc:
        _fail(str(exc), code=2)
        raise  # pragma: no cover - _fail always raises
    try:
        client = client_for(Provider(spec.provider))
    except ModelAgentError as exc:
        _fail(str(exc), code=2)
        raise  # pragma: no cover - _fail always raises
    typer.echo(
        f"model-driven agent; every tick is decided by {spec.model} via {spec.provider}, "
        f"on your own account."
    )
    return ModelAgent(spec, client=client, instructions=instructions)


@provider_app.command("list")
def provider_list() -> None:
    """The providers the official build ships an adapter for, and how each is reached.

    A closed list. Each entry is a path the provider documents for programmatic
    or third-party use; nothing here routes around a consumer-terms bar or
    alters client identity. Anything else is yours to add as an unsanctioned
    adapter, outside this build.
    """
    for info in PROVIDERS.values():
        available, _ = availability(info.provider)
        state = "available" if available else "not available"
        typer.echo(f"{info.provider.value:<10} {info.shape.value:<9} {state}")
        typer.echo(f"{'':<10} via {info.documented_path}")
        typer.echo(f"{'':<10} needs: {info.requires}")


@provider_app.command("install")
def provider_install(
    provider: Annotated[Provider, typer.Argument(help="Only codex can be installed by Tick")],
) -> None:
    """Install the pinned Codex CLI release under TICK_HOME/bin (Linux x86_64 boxes)."""
    from tick.serve.codex_install import CodexInstallError, default_fetch, install_codex

    if provider is not Provider.CODEX:
        _fail(f"{provider.value} has nothing to install: set its API key on the box instead.")
    try:
        result = install_codex(_home(), fetch=default_fetch)
    except CodexInstallError as error:
        _fail(f"{error.code}: {error.reason}")
    typer.echo(f"{result['code']}: {result['path']} ({result['release']})")


@provider_app.command("check")
def provider_check(
    provider: Annotated[Provider, typer.Argument(help="One of: " + ", ".join(Provider))],
) -> None:
    """Say whether this machine can reach a provider, and show its terms note once.

    The note is informational. The provider's terms are an agreement between
    you and the provider; Tick is not a party to it and enforces nothing on
    this machine. Every agent starts in per-order approval; standing mode is
    your call.
    """
    info = PROVIDERS[provider]
    available, line = availability(provider)
    typer.echo(line)
    typer.echo(f"reached via {info.documented_path}.")
    typer.echo("")
    typer.echo(f"About {provider.value}'s terms: {info.terms_note}")
    if not available:
        raise typer.Exit(code=2)


def _ask(intent: OrderIntent) -> bool:
    """Per-order approval, on stdin. A refusal to answer is a decline."""
    return typer.confirm(f"{intent.reason}. Place {intent.describe()}?", default=False)


def _queued_approver(
    agent: AgentRun,
    *,
    window: ApprovalWindow,
    run_id: str,
    terminal: Callable[[OrderIntent], bool],
) -> Callable[[OrderIntent], ApprovalDecision]:
    """Create immutable requests and let terminal or API commit the first result."""
    queue = ApprovalQueue.system(agent.home, agent.agent_id)

    def approve(intent: OrderIntent) -> ApprovalDecision:
        intent_payload = intent.model_dump(mode="json")
        evidence = {
            "price_asof": intent.price_asof,
            "price_source": intent.price_source,
            "est_price": intent.est_price,
            "est_notional": intent.est_notional,
        }
        request = queue.create(
            run_id=run_id,
            tick_id=secrets.token_hex(12),
            window=window,
            symbol=intent.symbol,
            side=intent.side.value,
            qty=intent.qty,
            est_price=intent.est_price,
            price_source=intent.price_source,
            data_class="display_only" if intent.price_source == "robinhood" else "local_fixture",
            est_notional=intent.est_notional,
            cage_checks=("session", "drawdown", "order_notional", "position", "positions"),
            proposed_by=intent.source,
            intent=intent_payload,
            evidence=evidence,
        )
        resolution = queue.wait(
            request.approval_id,
            run_id=run_id,
            stop_requested=agent.stop_requested,
            terminal_decision=lambda: terminal(intent),
            wait_for_change=time.sleep,
        )
        return ApprovalDecision(
            approved=resolution.outcome is ApprovalOutcome.APPROVED,
            decided_via=resolution.decided_via,
            outcome=resolution.outcome.value,
            note=resolution.reason,
            approval_id=request.approval_id,
            run_id=request.run_id,
            boot_id=request.boot_id,
            intent_hash=request.intent_hash,
            evidence_hash=request.evidence_hash,
        )

    return approve


# ----------------------------------------------------------------------
# pair / serve
# ----------------------------------------------------------------------


@pair_app.command("new")
def pair_new() -> None:
    """Create the pairing secret and print it once."""
    try:
        path, secret = create_secret(_home())
    except PairingError as exc:
        _fail(exc.reason, code=2)
        return
    typer.echo(secret)
    typer.echo(f"saved privately at {path}; copy the line above into the app now.")


@pair_app.command("rotate")
def pair_rotate() -> None:
    """Replace the pairing secret and print the replacement once."""
    try:
        path, secret = rotate_secret(_home())
    except PairingError as exc:
        _fail(exc.reason, code=2)
        return
    typer.echo(secret)
    typer.echo(f"replaced {path}; the old secret no longer authenticates the next request.")


@pair_app.command("show-path")
def pair_show_path() -> None:
    """Print where the secret lives without printing the secret."""
    typer.echo(pairing_secret_path(_home()))


@app.command("tunnel")
def tunnel_command(
    udp_port: Annotated[
        int,
        typer.Option("--udp-port", help="Required UDP port for direct Iroh connections."),
    ] = ...,
    relay_url: Annotated[
        str | None,
        typer.Option("--relay-url", help="Rendezvous URL for an own-machine box."),
    ] = None,
    no_relay: Annotated[
        bool,
        typer.Option("--no-relay", help="Disable rendezvous for a publicly addressed box."),
    ] = False,
    relay_from_account: Annotated[
        str | None,
        typer.Option(
            "--relay-from-account",
            help="Control-plane URL; reads TICK_ACCOUNT_SESSION for one relay lookup.",
        ),
    ] = None,
) -> None:
    """Expose the loopback box API over direct-only Iroh streams."""
    selected = sum((relay_url is not None, no_relay, relay_from_account is not None))
    if selected != 1:
        _fail(
            "choose exactly one of --relay-url, --relay-from-account, or --no-relay. "
            "Managed boxes use --no-relay; own machines can fetch rendezvous from the account.",
            code=2,
        )
    relay_token = None
    if relay_from_account is not None:
        account_session = os.environ.get("TICK_ACCOUNT_SESSION")
        if not account_session:
            _fail(
                "TICK_ACCOUNT_SESSION is missing. Sign in, export the current account session "
                "for this command, or pass --relay-url.",
                code=2,
            )
        from tick.tunnel.account import fetch_relay

        try:
            relay = fetch_relay(relay_from_account, account_session)
        except ValueError as exc:
            _fail(str(exc), code=2)
        relay_url = relay.url
        relay_token = relay.token
        typer.echo(
            "rendezvous loaded; keep this exact command running: "
            f"tick tunnel --udp-port {udp_port} --relay-from-account "
            f"{shlex.quote(relay_from_account)}"
        )
    from tick.tunnel import IrohEndpointPort, TunnelError, run_tunnel
    from tick.tunnel.server import forward_to_loopback, stderr_event

    try:
        asyncio.run(
            run_tunnel(
                home=_home(),
                udp_port=udp_port,
                relay_url=relay_url,
                relay_token=relay_token,
                endpoint_port=IrohEndpointPort(),
                forward=forward_to_loopback,
                log_event=stderr_event,
                now=lambda: datetime.now(UTC),
            )
        )
    except (TunnelError, PairingError, OSError, ValueError) as exc:
        _fail(str(exc), code=2)


@app.command("tunnel-info")
def tunnel_info_command() -> None:
    """Print the endpoint descriptor without exposing either private secret."""
    from tick.tunnel import load_tunnel_info

    try:
        info = load_tunnel_info(_home())
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        _fail(f"tunnel state is unavailable ({exc}). Start `tick tunnel` and retry.", code=2)
    typer.echo(json.dumps(info.json(), separators=(",", ":")))


@app.command("serve")
def serve_command(
    bind: Annotated[
        str,
        typer.Option("--bind", help="Listen on 127.0.0.1, or 0.0.0.0 only behind TLS."),
    ] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="TCP port for the box API.")] = 7433,
    behind_tls_proxy: Annotated[
        bool,
        typer.Option(
            "--behind-tls-proxy",
            help="Confirm that TLS or an authenticated overlay fronts a non-loopback bind.",
        ),
    ] = False,
) -> None:
    """Serve the authenticated box API; the paired phone is the control center."""
    if bind not in {"127.0.0.1", "0.0.0.0"}:
        _fail(
            "--bind must be 127.0.0.1 or 0.0.0.0. Use loopback unless a TLS proxy or "
            "authenticated overlay fronts this process.",
            code=2,
        )
    if bind == "0.0.0.0" and not behind_tls_proxy:
        _fail(
            "refusing --bind 0.0.0.0 without --behind-tls-proxy. Put Caddy TLS or a "
            "user-owned authenticated overlay in front, then start again.",
            code=2,
        )
    try:
        from tick.serve.pairing import load_secret

        load_secret(_home())
    except PairingError as exc:
        _fail(exc.reason, code=2)
    if bind == "0.0.0.0":
        typer.echo(
            "transport posture: Tick serves plain HTTP only behind your TLS proxy or "
            "authenticated overlay; TLS is the proxy's job."
        )
    from tick.serve.handlers import default_context
    from tick.serve.server import serve as serve_box

    if not os.environ.get("TICK_HOSTNAME"):
        typer.secho(
            "WARNING: TICK_HOSTNAME is not set; serve is plain HTTP. Keep it on loopback "
            "or behind an authenticated overlay; do not expose it publicly.",
            err=True,
            fg=typer.colors.RED,
        )
    typer.echo(f"serving {bind}:{port}; authenticated controls: on")
    os.environ["TICK_SERVE_PORT"] = str(port)
    serve_box(
        bind,
        port,
        context=default_context(_home(), os.environ),
    )


# ----------------------------------------------------------------------
# stop / status
# ----------------------------------------------------------------------


@app.command("stop")
def stop(
    agent_id: Annotated[str, typer.Argument(help="The agent to stop.")],
    reason: Annotated[
        str, typer.Option("--reason", help="Why it is being stopped; kept in the STOP file.")
    ] = "stopped from the command line",
) -> None:
    """Set the kill switch. No new order is placed until the file is removed.

    The switch is the presence of a `STOP` file in the agent's directory, so it
    works whether or not anything is running, and it is checked before every
    tick. Remove that file by hand to let the agent run again.
    """
    agent = _load(agent_id)
    path = agent.request_stop(reason=reason, at=datetime.now(UTC))
    typer.echo(f"stopped {agent.agent_id}: {agent.stop_reason()}")
    typer.echo(f"remove {path} to let it run again.")


@app.command("status")
def status(
    agent_id: Annotated[str, typer.Argument(help="The agent to describe.")],
) -> None:
    """Show what an agent is, what mode it is in, and whether its record verifies."""
    agent = _load(agent_id)
    summary = joined_agent_status(agent, pid_alive=_pid_alive, current_boot_id=boot_id())
    for key in (
        "agent_id",
        "name",
        "spec_id",
        "kind",
        "model",
        "cadence",
        "mode",
        "approval",
        "stopped",
        "session_date",
        "day_start_equity",
        "cancels_this_session",
        "max_cancels_per_session",
        "last_tick",
        "run_state",
        "current_mode",
        "previous_run_mode",
        "transition",
        "attention_required",
        "last_contact",
        "ledger",
    ):
        typer.echo(f"{key}: {summary[key]}")
    typer.echo(f"universe: {', '.join(summary['universe'])}")
    typer.echo(f"rules: {', '.join(summary['rules'])}")
    if summary["stopped"]:
        typer.echo(f"stop reason: {agent.stop_reason()}")
    typer.echo(f"record: {agent.verify_ledger()}")


def _pid_alive(pid: int) -> bool:
    """Read process liveness without signalling it to stop."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@app.command("doctor")
def doctor(
    port: Annotated[
        int, typer.Option("--port", help="Loopback port where tick serve should answer.")
    ] = 7433,
    ack_demotion: Annotated[
        str | None,
        typer.Option(
            "--ack-demotion",
            help="Acknowledge one observed reboot live-to-paper transition by run id.",
        ),
    ] = None,
) -> None:
    """Print every owner-run dependency in order; any live gate exits non-zero."""
    from tick.serve.doctor import codex_login_status, loopback_status, systemd_unit_fragments
    from tick.tunnel import tunnel_status

    home = _home()
    report = run_doctor(
        home,
        now=datetime.now(UTC),
        provider_status=codex_login_status,
        loopback_status=lambda: loopback_status(home, port),
        tunnel_status=lambda: tunnel_status(home),
        unit_fragments=systemd_unit_fragments,
        pid_alive=_pid_alive,
    )
    if ack_demotion is not None:
        observations = next(
            (
                list(check.detail.get("observations", []))
                for check in report.checks
                if check.name == "reboot demotion" and check.detail is not None
            ),
            [],
        )
        try:
            acknowledge_demotion(home, ack_demotion, observations)
        except ValueError as exc:
            _fail(str(exc), code=2)
        report = run_doctor(
            home,
            now=datetime.now(UTC),
            provider_status=codex_login_status,
            loopback_status=lambda: loopback_status(home, port),
            tunnel_status=lambda: tunnel_status(home),
            unit_fragments=systemd_unit_fragments,
            pid_alive=_pid_alive,
        )
    for check in report.checks:
        typer.echo(check.line())
    if not report.ready:
        raise typer.Exit(code=1)


@app.command("mcp", hidden=True)
def box_mcp(
    setup_session: Annotated[str | None, typer.Option("--setup-session", hidden=True)] = None,
) -> None:
    """Serve the local box tools over stdio for the user's chat provider."""
    from tick.mcpbox import run_stdio
    from tick.serve.handlers import default_context

    run_stdio(
        _home(),
        default_context(_home(), os.environ),
        setup_session_id=setup_session,
    )


# ----------------------------------------------------------------------
# ledger
# ----------------------------------------------------------------------


@ledger_app.command("show")
def ledger_show(
    agent_id: Annotated[str, typer.Argument(help="The agent whose record to read.")],
    verify_only: Annotated[
        bool,
        typer.Option("--verify", help="Only check the chain and report; print no records."),
    ] = False,
    tail: Annotated[int, typer.Option("--tail", help="Print only the last N records.")] = 0,
) -> None:
    """Print an agent's record, or verify it.

    Verification walks the whole chain and reports the first record that fails
    and why. A record is never edited or deleted, so a broken chain is not
    repaired here — see `tick ledger new`.
    """
    agent = _load(agent_id)
    result = agent.verify_ledger()
    if verify_only:
        typer.echo(str(result))
        raise typer.Exit(code=0 if result.ok else 1)
    if not result.ok:
        typer.echo(str(result))
        typer.echo(f"start a successor with: tick ledger new {agent.agent_id}")
        raise typer.Exit(code=1)
    records = list(read(agent.ledger_path))
    if tail > 0:
        records = records[-tail:]
    for record in records:
        typer.echo(encode_line(record).rstrip("\n"))


@ledger_app.command("new")
def ledger_new(
    agent_id: Annotated[str, typer.Argument(help="The agent whose record will not verify.")],
) -> None:
    """Start a successor ledger beside one that cannot be extended.

    The broken file is left exactly as it is, made read-only, and kept: it is
    evidence, and evidence is never overwritten. The new file's first record
    names the abandoned chain's last good record, its hash, and why
    verification failed, so the two are read together.

    Refused while the current record still verifies — a successor leaves a
    broken chain behind, it is not a way to start a fresh page.
    """
    agent = _load(agent_id)
    try:
        successor, note = agent.start_successor_ledger(clock=utc_clock)
    except (TickRuntimeError, RecordError) as exc:
        _fail(str(exc))
        return
    typer.echo(f"started {successor.name}")
    typer.echo(json.dumps(note.payload, indent=2, sort_keys=True))
    typer.echo(f"{agent.ledger_path.name} continues the record; the old file is kept, read-only.")


@ledger_app.command("export")
def ledger_export(
    agent_id: Annotated[str, typer.Argument(help="The agent whose record to export.")],
    for_evidence: Annotated[
        bool,
        typer.Option(
            "--for-evidence",
            help="Write a review copy with account identifiers and broker-derived rows redacted.",
        ),
    ] = False,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Destination; otherwise TICK_HOME/evidence/<agent>.jsonl."),
    ] = None,
) -> None:
    """Export a redacted evidence artifact; never export brokerage rows in clear."""
    if not for_evidence:
        _fail(
            "ledger export requires --for-evidence because an ordinary copy could move "
            "broker data off this box. Add the flag for a redacted copy."
        )
    agent = _load(agent_id)
    destination = out or (_home() / "evidence" / f"{agent.agent_id}.jsonl")
    try:
        path = export_evidence(agent.ledger_path, destination)
    except (OSError, ValueError, RecordError) as exc:
        _fail(str(exc))
        return
    typer.echo(path)
    typer.echo("broker-derived rows are hashes and kinds; account-shaped fields were removed.")


# ----------------------------------------------------------------------
# connect / disconnect / broker
# ----------------------------------------------------------------------


def _open_robinhood_session(
    server_url: str,
    storage: FileTokenStorage,
    loopback: LoopbackAuthorization,
    timeout_seconds: float,
) -> MCPSession:
    """Build the authorised MCP session for `server_url`.

    One seam, for one reason: it is the only place in the CLI where a socket
    could be opened, so it is the only place a test has to replace to drive the
    whole ceremony against an in-memory server. Everything above it — the
    disclosure, the ordering, the reporting, the refusal to print a token — is
    then exercised for real rather than mocked around.
    """
    provider = build_oauth_provider(server_url=server_url, storage=storage, loopback=loopback)
    return MCPSession(
        streamable_http_session(server_url, provider), timeout_seconds=timeout_seconds
    )


def _with_broker_session(
    server_url: str,
    *,
    port: int,
    timeout_seconds: float,
    open_browser: bool,
):
    """Open a session, or fail with words. The caller closes it."""
    storage = FileTokenStorage(_home())
    loopback = LoopbackAuthorization(
        port=port,
        timeout_seconds=timeout_seconds,
        open_browser=open_browser,
        announce=typer.echo,
        redirect_uri_override=None,
        on_callback=None,
    )
    loopback.__enter__()
    try:
        session = _open_robinhood_session(server_url, storage, loopback, timeout_seconds)
        session.open()
    except (AuthError, BrokerError) as exc:
        loopback.__exit__(None, None, None)
        _fail(str(exc), code=2)
        raise  # pragma: no cover - _fail always raises
    loopback.__exit__(None, None, None)
    return session


def _require_connected() -> FileTokenStorage:
    storage = FileTokenStorage(_home())
    if not storage.connected():
        _fail(
            "this machine is not connected to Robinhood. Run `tick connect robinhood` "
            "first; it explains what the grant covers before anything opens."
        )
    return storage


def _require_sanction(server_url: str, *, unsanctioned: bool) -> str:
    """Enforce the host pin before transport; community access is always explicit."""
    sanction = sanction_for(server_url)
    if sanction == "community" and not unsanctioned:
        _fail(
            f"{server_url} is outside Tick's broker host allowlist. Add --unsanctioned "
            "only if you intend to use this community server; nothing was connected.",
            code=2,
        )
    return sanction


@connect_app.command("robinhood")
def connect_robinhood(
    server_url: Annotated[
        str,
        typer.Option("--server-url", help="The Trading MCP to authorise against."),
    ] = ROBINHOOD_MCP_URL,
    port: Annotated[
        int,
        typer.Option(
            "--port",
            help=(
                "Which loopback port catches the redirect. 0 picks a free one. The "
                "listener binds 127.0.0.1 only and stops as soon as the code arrives."
            ),
        ),
    ] = 0,
    timeout: Annotated[
        float,
        typer.Option("--timeout", help="Seconds to wait for you to finish in the browser."),
    ] = BROKER_TIMEOUT_SECONDS,
    open_browser: Annotated[
        bool,
        typer.Option(
            "--open-browser/--no-open-browser",
            help="Also hand the URL to this machine's browser. It is always printed.",
        ),
    ] = True,
    yes: Annotated[
        bool,
        typer.Option("--yes", help="Skip the confirmation after the disclosure."),
    ] = False,
    unsanctioned: Annotated[
        bool,
        typer.Option(
            "--unsanctioned",
            help="Allow a server outside the official host pin and mark its profile community.",
        ),
    ] = False,
) -> None:
    """Authorise Tick against your Robinhood account, on this machine only.

    The disclosure below is printed before anything opens, every time. Read it:
    Robinhood's grant is wider than what Tick uses, and the difference is the
    part you are agreeing to.

    Nothing about your token is ever printed, logged, or sent anywhere. It is
    written to TICK_HOME/robinhood/ readable by you alone.
    """
    sanction = _require_sanction(server_url, unsanctioned=unsanctioned)
    typer.echo(disclosure_text())
    typer.echo("")
    if not yes and not typer.confirm("Open the authorization page now?", default=False):
        typer.echo("Nothing was authorised and nothing was written.")
        raise typer.Exit(code=1)

    session = _with_broker_session(
        server_url, port=port, timeout_seconds=timeout, open_browser=open_browser
    )
    try:
        tools = session.list_tools()
    except BrokerError as exc:
        _fail(str(exc), code=2)
        return
    finally:
        session.close()

    storage = FileTokenStorage(_home())
    typer.echo("")
    typer.echo(f"Connected to {session.server_name or server_url}.")
    typer.echo(f"{len(tools)} tool(s) discovered. Nothing about them is assumed.")
    typer.echo(f"server sanction: {sanction}")
    typer.echo(f"The grant is stored in {storage.directory}, readable by you alone.")
    typer.echo("Next: tick broker propose")


@disconnect_app.command("robinhood")
def disconnect_robinhood() -> None:
    """Delete this machine's copy of the Robinhood grant.

    Local only, because the credential is local: there is no Tick-side copy to
    revoke. Revoke the grant at Robinhood as well if you want it gone for good;
    this command says so rather than implying it did that for you.
    """
    storage = FileTokenStorage(_home())
    removed = storage.forget()
    if not removed:
        typer.echo(f"nothing to remove: {storage.directory} holds no grant.")
        return
    for path in removed:
        typer.echo(f"removed {path}")
    typer.echo("Revoke the grant at Robinhood too; this removed only the local copy.")


@broker_app.command("tools")
def broker_tools(
    server_url: Annotated[
        str, typer.Option("--server-url", help="The Trading MCP to ask.")
    ] = ROBINHOOD_MCP_URL,
    timeout: Annotated[
        float, typer.Option("--timeout", help="Seconds to wait for the broker.")
    ] = BROKER_TIMEOUT_SECONDS,
) -> None:
    """List the tools the broker actually declares, with their inputs.

    This is invariant 7 made visible: Tick assumes nothing about a broker's
    tool names or shapes, so the first thing you can do with a connection is
    read what it really offers.
    """
    _require_connected()
    session = _with_broker_session(server_url, port=0, timeout_seconds=timeout, open_browser=False)
    try:
        tools = session.list_tools()
    except BrokerError as exc:
        _fail(str(exc), code=2)
        return
    finally:
        session.close()
    for tool in tools:
        _describe_tool(tool)
    typer.echo("")
    typer.echo(f"{len(tools)} tool(s). Propose them with: tick broker propose")


def _describe_tool(tool: DiscoveredTool) -> None:
    typer.echo("")
    typer.echo(tool.name)
    if tool.description:
        typer.echo(f"  {tool.description.strip().splitlines()[0]}")
    required = set(tool.required_inputs())
    for name in tool.input_properties():
        kind = (tool.input_schema.get("properties") or {}).get(name, {}).get("type", "?")
        flag = "required" if name in required else "optional"
        typer.echo(f"    {name}: {kind} ({flag})")
    if not tool.input_properties():
        typer.echo("    (no arguments)")
    typer.echo(f"    output schema: {'declared' if tool.output_properties() else 'not declared'}")


def _proposal_path(home: Path) -> Path:
    return profile_path(home).with_name("proposal.json")


def _load_proposal(home: Path) -> ProfileProposal:
    path = _proposal_path(home)
    if not path.exists():
        _fail(
            f"no proposal exists at {path}. Run `tick broker propose` first; no tool was confirmed."
        )
    try:
        return ProfileProposal.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        _fail(
            f"the proposal at {path} cannot be read ({exc}). Run `tick broker propose` "
            "again; no tool was confirmed."
        )
        raise AssertionError("unreachable") from exc


def _parse_assignments(values: list[str] | None, *, option: str) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for value in values or ():
        if "=" not in value:
            _fail(f"{option} {value!r} must have the form name=value; nothing was written.", code=2)
        name, raw = value.split("=", 1)
        if not name.strip() or not raw:
            _fail(f"{option} {value!r} must have a name and value; nothing was written.", code=2)
        try:
            parsed[name.strip()] = json.loads(raw)
        except json.JSONDecodeError:
            parsed[name.strip()] = raw
    return parsed


@broker_app.command("propose")
def broker_propose(
    account: Annotated[
        str | None,
        typer.Option(
            "--account",
            help="Legacy local account binding; normally discover it after read.accounts.",
        ),
    ] = None,
    server_url: Annotated[
        str, typer.Option("--server-url", help="The MCP server whose complete inventory to pin.")
    ] = ROBINHOOD_MCP_URL,
    timeout: Annotated[
        float, typer.Option("--timeout", help="Seconds to wait for the broker inventory.")
    ] = BROKER_TIMEOUT_SECONDS,
    unsanctioned: Annotated[
        bool,
        typer.Option("--unsanctioned", help="Permit a community server outside the host pin."),
    ] = False,
) -> None:
    """Categorize the complete live inventory and write broker/proposal.json.

    This deterministic pass writes a proposal, never authorization. Unknown or
    conflicting tools remain unmapped; deterministic denial always wins.
    """
    _require_connected()
    _require_sanction(server_url, unsanctioned=unsanctioned)
    session = _with_broker_session(server_url, port=0, timeout_seconds=timeout, open_browser=False)
    try:
        proposal = propose_profile(
            session.list_tools(),
            server=server_url,
            account_id=account,
            proposed_at=datetime.now(UTC),
        )
    except (BrokerError, ValueError) as exc:
        _fail(str(exc), code=2)
        return
    finally:
        session.close()
    path = write_private_file(_proposal_path(_home()), proposal.model_dump_json(indent=2))
    typer.echo(f"proposal: {path}")
    typer.echo(f"inventory_hash: {proposal.inventory_hash}")
    for name, tool in proposal.tools.items():
        category = tool.category.value if tool.category is not None else "unmapped"
        typer.echo(f"{name}: {category}")
        for argument, template in sorted(tool.arguments.items()):
            typer.echo(f"  argument {argument} = {template}")
        for role, result_path in sorted(tool.result.items()):
            typer.echo(f"  {role} read from {result_path}")
        if tool.note:
            typer.echo(f"  note: {tool.note}")
    typer.echo("Nothing is callable yet. Review this file, then run `tick broker confirm`.")


@broker_app.command("confirm")
def broker_confirm(
    tool: Annotated[
        list[str] | None,
        typer.Option(
            "--tool",
            help="Name a callable tool to confirm. Order tools still ask for an explicit y.",
        ),
    ] = None,
    yes_all_reads: Annotated[
        bool,
        typer.Option(
            "--yes-all-reads",
            help="Confirm every proposed read; this never confirms an order tool.",
        ),
    ] = False,
    recategorize: Annotated[
        tuple[str, str] | None,
        typer.Option(
            "--recategorize",
            help="Override one non-denied proposal: --recategorize TOOL CATEGORY.",
        ),
    ] = None,
    fixed: Annotated[
        list[str] | None,
        typer.Option(
            "--fixed",
            help="Bind a fixed literal as tool.argument=value; repeat for each literal.",
        ),
    ] = None,
    unsanctioned: Annotated[
        bool,
        typer.Option("--unsanctioned", help="Confirm a community server outside the host pin."),
    ] = False,
) -> None:
    """Confirm tools individually and write profile.json plus a ledger note.

    Reads may be accepted together with ``--yes-all-reads``. No flag confirms
    unnamed order tools: each one must be named and answered with ``y``.
    """
    home = _home()
    proposal = _load_proposal(home)
    _require_sanction(proposal.server, unsanctioned=unsanctioned)
    selected = set(tool or ())
    unknown = selected - set(proposal.tools)
    if unknown:
        _fail(f"unknown proposed tools: {', '.join(sorted(unknown))}; nothing was written.", code=2)
    categories: dict[str, Category | None] = {
        name: proposed.category for name, proposed in proposal.tools.items()
    }
    if recategorize is not None:
        name, raw_category = recategorize
        if name not in proposal.tools:
            _fail(f"{name!r} is not in the proposal; nothing was written.", code=2)
        proposed = proposal.tools[name]
        if proposed.category is not None and proposed.category.denied:
            _fail(
                f"{name} is deterministically {proposed.category.value}; denied tools "
                "cannot be recategorized or called. Change the category registry in code "
                "only if that policy is wrong.",
                code=2,
            )
        try:
            categories[name] = Category(raw_category)
        except ValueError:
            _fail(f"{raw_category!r} is not a broker category; nothing was written.", code=2)
    fixed_values = _parse_assignments(fixed, option="--fixed")
    existing = load_profile(home)
    now = datetime.now(UTC)
    if existing is not None:
        differences = diff_profile(
            existing, tuple(proposed.contract for proposed in proposal.tools.values())
        )
        if differences:
            typer.echo("full contract diff since the active profile:")
            for difference in differences:
                typer.echo(f"  {difference.sentence()}")
    confirmed: dict[str, ProfileTool] = {}
    for name, proposed in proposal.tools.items():
        category = categories[name]
        typer.echo("")
        typer.echo(f"{name}: {category.value if category is not None else 'unmapped'}")
        if proposed.contract.annotations:
            typer.echo(
                f"  MCP annotations (UNTRUSTED HINTS): {dict(proposed.contract.annotations)}"
            )
        if category is None:
            typer.echo("  not callable; left unmapped")
            continue
        if category.denied:
            typer.echo("  denied by policy; cannot be confirmed into the callable router")
            confirmed[name] = ProfileTool(
                category=category,
                contract=proposed.contract,
                arguments={},
                result={},
                confirmed_contract_hash=None,
                mapping_hash=mapping_hash(category, {}, {}),
                confirmed_at=None,
                confirmed_by=None,
                categorizer_version=CATEGORIZER_VERSION,
                proved_contract_hash=None,
                proved_mapping_hash=None,
                proved_at=None,
                proof=None,
            )
            continue
        arguments = dict(proposed.arguments)
        prefix = name + "."
        for key, value in fixed_values.items():
            if key.startswith(prefix):
                arguments[key[len(prefix) :]] = value
        current_mapping_hash = mapping_hash(category, arguments, proposed.result)
        prior = existing.tools.get(name) if existing is not None else None
        for argument, value in sorted(arguments.items()):
            typer.echo(f"  argument {argument} = {value}")
        for role, path in sorted(proposed.result.items()):
            typer.echo(f"  result {role} <- {path}")
        if prior is not None and prior.contract.contract_hash != proposed.contract.contract_hash:
            typer.echo(f"  previous contract_hash: {prior.contract.contract_hash}")
            typer.echo(f"  current contract_hash:  {proposed.contract.contract_hash}")
            for label in (
                "title",
                "description",
                "input_schema",
                "output_schema",
                "execution",
                "annotations",
            ):
                before = getattr(prior.contract, label)
                after = getattr(proposed.contract, label)
                if before != after:
                    typer.echo(
                        f"  {label} before: "
                        f"{json.dumps(before, sort_keys=True, ensure_ascii=False)}"
                    )
                    typer.echo(
                        f"  {label} after:  {json.dumps(after, sort_keys=True, ensure_ascii=False)}"
                    )
        if prior is not None and prior.mapping_hash != current_mapping_hash:
            typer.echo(f"  previous mapping_hash: {prior.mapping_hash}")
            typer.echo(f"  current mapping_hash:  {current_mapping_hash}")
        unchanged = bool(
            prior
            and prior.category is category
            and prior.contract.contract_hash == proposed.contract.contract_hash
            and prior.mapping_hash == current_mapping_hash
        )
        if unchanged:
            confirmed[name] = prior  # type: ignore[assignment]
            typer.echo("  unchanged confirmation and proof carried forward")
            continue
        should_confirm = name in selected or (yes_all_reads and category.value.startswith("read."))
        if category.value.startswith("order."):
            if name not in selected:
                typer.echo("  order tool left unmapped: name it with --tool and confirm it")
                continue
            should_confirm = typer.confirm(
                f"Confirm {name} as {category.value} with these exact arguments and results?",
                default=False,
            )
        elif not should_confirm:
            should_confirm = typer.confirm(
                f"Confirm {name} as {category.value} with these exact arguments and results?",
                default=False,
            )
        if not should_confirm:
            typer.echo("  left unmapped; it remains uncallable")
            continue
        try:
            confirmed[name] = ProfileTool(
                category=category,
                contract=proposed.contract,
                arguments=arguments,
                result=proposed.result,
                confirmed_contract_hash=proposed.contract.contract_hash,
                mapping_hash=current_mapping_hash,
                confirmed_at=now,
                confirmed_by="terminal",
                categorizer_version=CATEGORIZER_VERSION,
                proved_contract_hash=None,
                proved_mapping_hash=None,
                proved_at=None,
                proof=None,
            )
        except ValueError as exc:
            _fail(
                f"{name} cannot be confirmed: {exc}. Fix the proposal; nothing was written.", code=2
            )
    profile = build_profile(
        server=proposal.server,
        account_id=proposal.account_id,
        tools=confirmed,
        inventory_hash=proposal.inventory_hash,
        data_class="display_only",
        sanction=proposal.sanction,
        profile_format_version=PROFILE_FORMAT_VERSION,
        canonicalizer_version=CANONICALIZER_VERSION,
        category_registry_version=CATEGORY_REGISTRY_VERSION,
        state=ProfileState.CONFIRMED,
        observed_inventory_hash=proposal.inventory_hash,
        drift=(),
    )
    path = confirm_profile(home, profile, actor=local_actor(os.environ), at=now)
    typer.echo("")
    typer.echo(f"confirmed profile: {path}")
    typer.echo(f"profile_hash: {profile.profile_hash}")
    typer.echo("Next: run `tick broker prove` with the probe inputs your read tools require.")


@broker_app.command("prove")
def broker_prove(
    probe: Annotated[
        list[str] | None,
        typer.Option(
            "--probe",
            help=(
                "A user-supplied probe input as name=value; repeat for symbol, count, "
                "or order fields."
            ),
        ),
    ] = None,
    timeout: Annotated[
        float, typer.Option("--timeout", help="Seconds to wait for each proof call.")
    ] = BROKER_TIMEOUT_SECONDS,
    unsanctioned: Annotated[
        bool,
        typer.Option("--unsanctioned", help="Use a confirmed community profile."),
    ] = False,
) -> None:
    """Exercise mapped reads/preflight and write proof results plus a ledger note.

    Probe values come from ``--probe`` and are never invented. Place, replace,
    and cancel are never called; preflight proves only preflight.
    """
    home = _home()
    profile = load_profile(home)
    if profile is None:
        _fail("no broker profile exists. Run `tick broker propose`, then `tick broker confirm`.")
    assert profile is not None
    _require_sanction(profile.server, unsanctioned=unsanctioned)
    _require_connected()
    session = _with_broker_session(
        profile.server, port=0, timeout_seconds=timeout, open_browser=False
    )
    at = datetime.now(UTC)
    try:
        verified = verify_session_profile(
            profile,
            session,
            server=profile.server,
            account_id=profile.account_id,
            confirmation_recorded=has_confirmation_note(home, profile.profile_hash),
        )
        proven, outcomes = prove_profile(
            profile,
            verified,
            probe_values=_parse_assignments(probe, option="--probe"),
            at=at,
        )
        save_profile(home, proven)
        Ledger(profile_path(home).with_name("records.jsonl"), clock=lambda: at).append(
            RecordKind.NOTE,
            {
                "event": "profile_proven",
                "profile_hash": proven.profile_hash,
                "inventory_hash": verified.inventory_hash,
                "at": at,
                "outcome": {
                    name: result.model_dump(mode="json") for name, result in outcomes.items()
                },
            },
            source=DataSource.RUNTIME,
        )
    except BrokerError as exc:
        _fail(str(exc), code=2)
        return
    finally:
        session.close()
    for name, result in outcomes.items():
        state = "proved" if result.success else "UNRESOLVED"
        typer.echo(f"{name}: {state}")
        for role in result.resolved:
            typer.echo(f"  {role}: resolved")
        for role, reason in result.unresolved.items():
            typer.echo(f"  {role}: Unavailable — {reason}")
    typer.echo(f"wrote profile_proven to {profile_path(home).with_name('records.jsonl')}")


@broker_app.command("status")
def broker_status(
    timeout: Annotated[
        float, typer.Option("--timeout", help="Seconds to wait for the live inventory.")
    ] = BROKER_TIMEOUT_SECONDS,
    unsanctioned: Annotated[
        bool,
        typer.Option("--unsanctioned", help="Inspect a confirmed community profile."),
    ] = False,
) -> None:
    """Verify live contracts, print per-tool state, and write drift observation.

    The command writes only mutable observation fields in ``profile.json``;
    it never accepts drift or changes what the user confirmed.
    """
    home = _home()
    storage = FileTokenStorage(home)
    typer.echo(f"grant: {'stored' if storage.connected() else 'none'} in {storage.directory}")
    try:
        profile = load_profile(home)
    except BrokerError as exc:
        _fail(str(exc))
        return
    if profile is None:
        typer.echo(f"profile state: none at {profile_path(home)}")
        typer.echo("run: tick broker propose")
        return
    typer.echo(f"server: {profile.server}")
    typer.echo(f"sanction: {profile.sanction}")
    typer.echo(f"account: {profile.account_id}")
    typer.echo(f"profile_hash: {profile.profile_hash}")
    if not storage.connected():
        typer.echo(f"profile state: {profile.state.value}; live inventory was not fetched")
        typer.echo("run: tick connect robinhood")
        return
    _require_sanction(profile.server, unsanctioned=unsanctioned)
    session = _with_broker_session(
        profile.server, port=0, timeout_seconds=timeout, open_browser=False
    )
    try:
        verified = verify_session_profile(
            profile,
            session,
            server=profile.server,
            account_id=profile.account_id,
            confirmation_recorded=has_confirmation_note(home, profile.profile_hash),
        )
        contracts = tuple(verified.contracts.values())
        differences = diff_profile(profile, contracts)
        mapped_drift = any(
            state is ToolState.DRIFTED
            for name, state in verified.states.items()
            if name in profile.tools and profile.tools[name].category.callable
        )
        state = ProfileState.DRIFTED if mapped_drift else ProfileState.CONFIRMED
        observed = build_profile(
            server=profile.server,
            account_id=profile.account_id,
            tools=profile.tools,
            inventory_hash=profile.inventory_hash,
            data_class=profile.data_class,
            sanction=profile.sanction,
            profile_format_version=profile.profile_format_version,
            canonicalizer_version=profile.canonicalizer_version,
            category_registry_version=profile.category_registry_version,
            state=state,
            observed_inventory_hash=verified.inventory_hash,
            drift=differences,
        )
        save_profile(home, observed)
    except BrokerError as exc:
        _fail(str(exc), code=2)
        return
    finally:
        session.close()
    typer.echo(f"profile state: {state.value}")
    typer.echo(f"inventory_hash: {verified.inventory_hash}")
    typer.echo(
        "confirmation ledger: "
        + ("present" if verified.confirmation_recorded else "MISSING — live readiness refuses")
    )
    for name in sorted(verified.states):
        stored = profile.tools.get(name)
        category = stored.category.value if stored is not None else "unmapped"
        proof = "proved" if stored is not None and stored.proved else "not proved"
        typer.echo(f"{name}: {category} | {verified.states[name].value} | {proof}")
    unmapped_names = [
        name for name, state_value in verified.states.items() if state_value is ToolState.UNMAPPED
    ]
    typer.echo(f"unmapped (these refuse): {', '.join(sorted(unmapped_names)) or 'nothing'}")
    if differences:
        typer.echo("drift diff:")
        for difference in differences:
            typer.echo(f"  {difference.sentence()}")


def main() -> None:  # pragma: no cover - console-script entry point
    """The console script. Importing this module runs nothing."""
    app()
