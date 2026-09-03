"""`tick commons` — explicit opt-in, keys, public passes, and operator seeding."""

from __future__ import annotations

import importlib
import json
import os
import shlex
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from tick.records import tick_home, write_private_file

from .canonical import canonical_hash, canonical_json
from .client import CommonsClient, CommonsClientError
from .keys import contributor_id, generate_key, load_key
from .models import ClaimBody, ScreenCriterion, ScreenOperator, utc_now

app = typer.Typer(
    help="Read and, only after opt-in, contribute signed facts about public subjects.",
    no_args_is_help=True,
)


def _home() -> Path:
    return tick_home(os.environ)


def _client() -> CommonsClient:
    url = os.environ.get("COMMONS_URL")
    if not url:
        raise ValueError("COMMONS_URL is not set; name the commons service and try again")
    return CommonsClient(url, load_key(_home()))


def _fail(exc: Exception) -> None:
    typer.secho(str(exc), err=True, fg=typer.colors.RED)
    raise typer.Exit(code=1)


@app.command("keygen")
def keygen() -> None:
    """Generate this box's pseudonymous Ed25519 contributor key."""
    try:
        key = generate_key(_home())
    except (FileExistsError, OSError, ValueError) as exc:
        _fail(exc)
        return
    typer.echo(contributor_id(key))
    typer.echo("private key written under TICK_HOME/commons with mode 0600")


@app.command("pass")
def read_pass(
    ticker: Annotated[str, typer.Argument(help="Public ticker, such as XYZ.")],
    observed_before: Annotated[
        str | None,
        typer.Option(help="Only facts knowable by this timezone-aware ISO moment."),
    ] = None,
) -> None:
    """Read the released, impersonal claim set for one public subject."""
    try:
        cutoff = datetime.fromisoformat(observed_before) if observed_before is not None else None
        result = _client().pass_for(ticker, cutoff)
    except (CommonsClientError, OSError, ValueError) as exc:
        _fail(exc)
        return
    typer.echo(canonical_json(result))


def _cutoff(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value is not None else None


def _criterion(expression: str) -> ScreenCriterion:
    parts = shlex.split(expression)
    if len(parts) < 2:
        raise ValueError(
            "--where needs '<predicate> <op> <value>'; add an explicit comparison and retry"
        )
    predicate_id, raw_op, *operands = parts
    try:
        op = ScreenOperator(raw_op)
    except ValueError as exc:
        raise ValueError(
            "screen op must be lt, lte, gt, gte, eq, between, or exists; choose one and retry"
        ) from exc
    if op is ScreenOperator.EXISTS:
        return ScreenCriterion(predicate_id=predicate_id, op=op)
    if op is ScreenOperator.BETWEEN:
        return ScreenCriterion(predicate_id=predicate_id, op=op, values=tuple(operands))
    value = operands[0] if len(operands) == 1 else None
    return ScreenCriterion(predicate_id=predicate_id, op=op, value=value)


@app.command("screen")
def screen(
    where: Annotated[
        list[str],
        typer.Option("--where", help="Explicit '<predicate> <op> <value>'; repeat to require all."),
    ],
    observed_before: Annotated[
        str | None,
        typer.Option(help="Only facts knowable by this timezone-aware ISO moment."),
    ] = None,
) -> None:
    """Read an explicit screen at the shared release cursor; this writes and spends nothing."""
    try:
        result = _client().screen(
            tuple(_criterion(item) for item in where), _cutoff(observed_before)
        )
    except (CommonsClientError, OSError, ValueError) as exc:
        _fail(exc)
        return
    typer.echo(canonical_json(result))


@app.command("graph")
def graph(
    ticker: Annotated[str, typer.Argument(help="Public ticker, such as XYZ.")],
    depth: Annotated[int, typer.Option(help="Claim-edge depth: 1 or 2.")] = 1,
    observed_before: Annotated[
        str | None,
        typer.Option(help="Only facts knowable by this timezone-aware ISO moment."),
    ] = None,
) -> None:
    """Read claim-backed neighbors and evidence at the shared release cursor."""
    try:
        result = _client().graph_for(ticker, depth, _cutoff(observed_before))
    except (CommonsClientError, OSError, ValueError) as exc:
        _fail(exc)
        return
    typer.echo(canonical_json(result))


@app.command("credits")
def credits(
    observed_before: Annotated[
        str | None,
        typer.Option(help="Only ledger entries visible by this timezone-aware ISO moment."),
    ] = None,
) -> None:
    """Read this key's computed credits at the shared release cursor; no value is stored."""
    try:
        result = _client().credits(_cutoff(observed_before))
    except (CommonsClientError, OSError, ValueError) as exc:
        _fail(exc)
        return
    typer.echo(canonical_json(result.model_dump(mode="python", by_alias=True)))


@app.command("reverify")
def reverify(
    claim_id: Annotated[str, typer.Argument(help="Released checked claim identity.")],
) -> None:
    """Write a signed deterministic recheck, then print its current release cursor."""
    try:
        client = _client()
        result = client.reverify(claim_id)
        release_id = client.claim(claim_id, None).release_id
    except (CommonsClientError, OSError, ValueError) as exc:
        _fail(exc)
        return
    typer.echo(canonical_json({"release_id": release_id, "reverification": result}))


@app.command("dispute")
def dispute(
    claim_id: Annotated[str, typer.Argument(help="Released claim identity.")],
    source_id: Annotated[str, typer.Option("--source", help="Existing evidence source identity.")],
    reason_code: Annotated[str, typer.Option("--reason", help="Factual dispute reason code.")],
) -> None:
    """Write an evidence-backed dispute, then print the claim's release cursor."""
    try:
        client = _client()
        result = client.dispute(claim_id, source_id, reason_code)
        release_id = client.claim(claim_id, None).release_id
    except (CommonsClientError, OSError, ValueError) as exc:
        _fail(exc)
        return
    typer.echo(canonical_json({"release_id": release_id, "dispute": result.dispute}))


@app.command("seed")
def seed(
    database_path: Annotated[Path, typer.Option("--db", help="Local commons SQLite file.")],
    ciks: Annotated[
        list[str] | None,
        typer.Option("--cik", help="SEC CIK to ingest; repeat for each public issuer."),
    ] = None,
    frames: Annotated[
        list[str] | None,
        typer.Option("--frames", help="Checked XBRL concept:period frame; repeat as needed."),
    ] = None,
    sic: Annotated[
        bool,
        typer.Option("--sic", help="Write SIC concept membership through the ordinary gate."),
    ] = False,
) -> None:
    """Run deterministic EDGAR ingestion into a local service database."""
    try:
        module = importlib.import_module("services.commons.ingest")
        database_module = importlib.import_module("services.commons.database")
        registry_module = importlib.import_module("services.commons.registry")
        service_dir = Path(module.__file__).resolve().parent
        database = database_module.CommonsDatabase(
            str(database_path),
            service_dir / "schema.sql",
            registry_module.load_registry(service_dir / "predicates.json"),
        )
        user_agent = os.environ.get("COMMONS_USER_AGENT", "")
        if not ciks and not frames:
            raise ValueError("seed needs --cik or --frames; name public SEC data and retry")
        fetcher = module.EdgarFetcher(module.time.monotonic, module.time.sleep)
        operator_key = load_key(_home())
        claims: tuple[str, ...] = ()
        if ciks:
            claims += module.ingest(
                ciks=ciks,
                database=database,
                fetcher=fetcher,
                user_agent=user_agent,
                operator_key=operator_key,
                clock=utc_now,
            )
        if frames:
            claims += module.ingest_frames(
                frames=frames,
                database=database,
                fetcher=fetcher,
                user_agent=user_agent,
                operator_key=operator_key,
                clock=utc_now,
            )
        if sic:
            claims += module.ingest_sic(
                ciks=ciks or (),
                database=database,
                fetcher=fetcher,
                user_agent=user_agent,
                operator_key=operator_key,
                clock=utc_now,
            )
        database.cut_release("local EDGAR seed", utc_now())
    except (CommonsClientError, OSError, ValueError) as exc:
        _fail(exc)
        return
    typer.echo(f"accepted {len(claims)} checked claims and cut a local release")


@app.command("opt-in")
def opt_in(
    yes: Annotated[
        bool, typer.Option("--yes", help="Confirm after printing the outward shape.")
    ] = False,
) -> None:
    """Show the exact outward shape, then record this box's explicit opt-in."""
    typer.echo(json.dumps(ClaimBody.model_json_schema(), sort_keys=True, indent=2))
    if not yes and not typer.confirm(
        "Only signed claims about public subjects leave this box. Opt in?", default=False
    ):
        typer.echo("not opted in; you can still read passes or work locally")
        return
    try:
        try:
            key = load_key(_home())
        except FileNotFoundError:
            key = generate_key(_home())
        marker = {
            "enabled_at": utc_now(),
            "contributor_id": contributor_id(key),
            "outward_schema_hash": canonical_hash(ClaimBody.model_json_schema()),
        }
        write_private_file(_home() / "commons" / "opt-in.json", canonical_json(marker) + "\n")
    except (OSError, ValueError) as exc:
        _fail(exc)
        return
    typer.echo("commons contribution is enabled; `tick commons pass XYZ` remains read-only")
