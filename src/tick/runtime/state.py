"""One agent on disk: its spec, its state, its ledger generations, its STOP file.

    agents/<agent_id>/
        spec.json          the spec this agent runs — a copy, written 0400
        state.json         mode, approval, the session's opening equity, counters
        records.jsonl      the ledger (slice 03)
        records.002.jsonl  a successor ledger, if one was ever started
        STOP               the kill switch: the file's PRESENCE is the switch

Everything lives under `TICK_HOME` on the user's own machine, which is what
makes the vendor posture architectural rather than promised (invariant 1).
Nothing in this module opens a socket.

Four decisions are worth reading before the code.

**The agent id is the spec's identity, truncated for typing.** `agent_id` is
the first twelve hex characters of `spec_id`, so two agents cannot run the same
document under different names by accident. Twelve characters are not an
identity check, though, so the FULL `spec_id` is stored in `state.json` and
re-derived from `spec.json` every time an agent is loaded: an edited spec
stops the agent instead of quietly changing what it does.

**The kill switch is a file, and its presence is the whole state.** No flag in
`state.json`, no field to be true or false. A file either exists or it does
not, `stop_requested()` is one `Path.exists()`, and any process, script or
person with the directory can set it — including when Tick itself is wedged
(invariant 8: the kill switch always wins, on every path).

**Nothing is defaulted that the user must choose.** `max_cancels_per_session`
has no default here and none in the CLI: a cancel guard nobody set is not a
guard. `day_start_equity` is `None` until a session actually opens, and it is
never zero — a drawdown limit computed against a fabricated opening balance is
invariant 5's failure with the safety machinery attached.

**A ledger is never repaired, only succeeded.** `start_successor_ledger`
implements the ruling: the broken file is left byte-for-byte as it is and made
read-only, a new generation is opened beside it, and its first record is a
`note` carrying the abandoned head's seq and hash and the verification reason.
There is no `--force`, no skip-the-bad-line and no automatic roll.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from pydantic import AwareDatetime, BaseModel, ConfigDict, model_validator

from tick.agents import (
    AgentSpec,
    ModelAgentSpec,
    agent_spec_id,
    dump_agent_spec,
    is_model_agent,
    load_agent_spec_file,
)
from tick.records import (
    DataSource,
    Ledger,
    LedgerCorrupt,
    Record,
    RecordKind,
    VerifyResult,
    agent_ledger_path,
    ensure_private_dir,
    read,
    verify,
)
from tick.spec import ExactDecimal

from .errors import LedgerQuarantined, RuntimeStateError
from .modes import ApprovalMode, Mode

__all__ = [
    "AGENT_ID_LENGTH",
    "INSTRUCTIONS_FILE",
    "SPEC_FILE",
    "STATE_FILE",
    "STOP_FILE",
    "AgentRun",
    "AgentState",
    "agent_id_for",
    "agents_dir",
    "state_summary",
]

#: How much of `spec_id` names the agent's directory. Long enough to type and
#: to be unique among the agents one person runs; never used as an identity
#: check, which is what the stored full `spec_id` is for.
AGENT_ID_LENGTH = 12

SPEC_FILE = "spec.json"
STATE_FILE = "state.json"
STOP_FILE = "STOP"

#: A model agent's own instructions, written by the USER. Tick ships none and
#: completes none; an agent whose file is missing or empty refuses to run
#: rather than acquiring a strategy nobody wrote. It is NOT written read-only
#: like `spec.json`, because editing it is the expected thing to do — which is
#: why its hash goes into every decision record rather than into the agent id.
INSTRUCTIONS_FILE = "instructions.md"

#: The agent's own files are private: they say what the user owns and what
#: their agents did with it.
FILE_MODE = 0o600

#: The spec copy is written read-only. It is not tamper-proof — the owner of a
#: file can always chmod it — but an edit has to be deliberate, and the
#: `spec_id` check catches it on the next load either way.
SPEC_MODE = 0o400

_AGENT_ID = re.compile(rf"\A[0-9a-f]{{{AGENT_ID_LENGTH}}}\Z")

#: `records.jsonl` is generation 1; `records.002.jsonl` and after are the
#: successors `tick ledger new` opens. The active ledger is the highest
#: generation present, and an old one is never rewritten or removed.
_SUCCESSOR = re.compile(r"\Arecords\.(\d{3,})\.jsonl\Z")


def agents_dir(home: str | os.PathLike[str]) -> Path:
    """`<home>/agents` — where every agent's directory lives."""
    return Path(home) / "agents"


def agent_id_for(spec: AgentSpec) -> str:
    """The directory name an agent running `spec` gets — either kind of document."""
    return agent_spec_id(spec)[:AGENT_ID_LENGTH]


class AgentState(BaseModel):
    """`state.json` — everything about a run that is not the spec.

    Frozen and closed. Changing it means writing a new one, which is the same
    discipline the ledger has, for a much weaker reason: this file is state,
    not evidence, and it is rewritten in place.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_id: str
    #: The full `spec_id` of `spec.json`, re-derived and compared on every load.
    spec_id: str
    mode: Mode
    approval: ApprovalMode
    #: The cancel-ratio guard. Required: repeated cancellation is a pattern
    #: brokers terminate connectivity over, and a limit nobody chose is none.
    max_cancels_per_session: int
    #: The ET session these counters belong to; `None` before the first tick.
    session_date: date | None
    #: What the account was worth when this session's first tick ran. `None`
    #: until then, never zero, never guessed.
    day_start_equity: ExactDecimal | None
    cancels_this_session: int
    last_tick: AwareDatetime | None
    created_at: AwareDatetime

    @model_validator(mode="after")
    def _check(self) -> AgentState:
        if not _AGENT_ID.match(self.agent_id):
            raise ValueError(
                f"agent_id {self.agent_id!r} is not {AGENT_ID_LENGTH} lowercase hex "
                f"characters; it names a directory and is derived from the spec id"
            )
        if not self.spec_id.startswith(self.agent_id):
            raise ValueError(
                f"agent {self.agent_id} claims spec {self.spec_id}, which does not begin "
                f"with its id; the agent id IS the first {AGENT_ID_LENGTH} characters"
            )
        if self.max_cancels_per_session < 0:
            raise ValueError(
                f"max_cancels_per_session ({self.max_cancels_per_session}) must be >= 0"
            )
        if self.cancels_this_session < 0:
            raise ValueError(f"cancels_this_session ({self.cancels_this_session}) must be >= 0")
        if self.day_start_equity is not None and self.day_start_equity <= 0:
            raise ValueError(
                f"day_start_equity ({self.day_start_equity}) must be > 0 where it is known; "
                f"a drawdown limit is a percentage of it, and it is None until a session "
                f"opens rather than zero"
            )
        if (self.day_start_equity is None) != (self.session_date is None):
            raise ValueError(
                "session_date and day_start_equity are set together: the opening equity "
                "belongs to a session, and a session with no opening equity cannot be "
                "checked against a drawdown limit"
            )
        return self

    def _changed(self, **changes: Any) -> AgentState:
        """A new state with `changes` applied, RE-VALIDATED.

        Deliberately not `model_copy(update=...)`, which assigns fields without
        running the validators: every rule above — the opening equity is
        positive, it travels with its session, a binary float is refused —
        would be bypassed by the one path that ever changes this document.
        """
        return AgentState.model_validate(self.model_dump() | changes)

    def opened(self, *, session_date: date, equity: Decimal) -> AgentState:
        """This state with a new session's opening equity, counters reset."""
        return self._changed(
            session_date=session_date,
            day_start_equity=equity,
            cancels_this_session=0,
        )

    def ticked(self, at: datetime) -> AgentState:
        return self._changed(last_tick=at)

    def cancelled(self) -> AgentState:
        return self._changed(cancels_this_session=self.cancels_this_session + 1)

    def with_mode(self, mode: Mode) -> AgentState:
        return self._changed(mode=mode)

    def with_approval(self, approval: ApprovalMode) -> AgentState:
        return self._changed(approval=approval)


class AgentRun:
    """One agent's directory, and everything the runtime does to it.

    `home` is `TICK_HOME`; it is always passed in, never read from the
    environment here, so a test cannot write into a developer's real `~/.tick`
    by forgetting to patch something.
    """

    def __init__(self, home: str | os.PathLike[str], agent_id: str) -> None:
        if not _AGENT_ID.match(agent_id):
            raise RuntimeStateError(
                f"{agent_id!r} is not an agent id ({AGENT_ID_LENGTH} lowercase hex "
                f"characters). Run `tick agents` to see the ones that exist."
            )
        self.home = Path(home)
        self.agent_id = agent_id
        self.directory = agents_dir(self.home) / agent_id

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"AgentRun({self.agent_id!r})"

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    @property
    def spec_path(self) -> Path:
        return self.directory / SPEC_FILE

    @property
    def state_path(self) -> Path:
        return self.directory / STATE_FILE

    @property
    def stop_path(self) -> Path:
        return self.directory / STOP_FILE

    @property
    def exists(self) -> bool:
        return self.state_path.exists() and self.spec_path.exists()

    # ------------------------------------------------------------------
    # Creation and loading
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        home: str | os.PathLike[str],
        spec: AgentSpec,
        *,
        max_cancels_per_session: int,
        approval: ApprovalMode,
        created_at: datetime,
        instructions: str | None,
    ) -> AgentRun:
        """Write a new agent's directory and return it.

        Adding the same spec twice is not an error and not a second agent: the
        id is the spec's own hash, so the second add finds the directory, checks
        that the spec in it is byte-identical, and hands the same agent back.
        A directory holding a DIFFERENT spec under this id is refused loudly.

        `instructions` is required to be PASSED and may be `None`, which is a
        different thing from having a default. A model agent runs the user's
        own instructions, so `None` for one is refused; a rule agent runs its
        document, so instructions for one are refused too. Neither is quietly
        ignored: an agent that silently dropped the words a person wrote for it
        would run something nobody chose.
        """
        agent_id = agent_id_for(spec)
        run = cls(home, agent_id)
        _check_instructions(spec, instructions)
        if run.exists:
            existing = run.spec
            if agent_spec_id(existing) != agent_spec_id(spec):
                raise RuntimeStateError(
                    f"agent {agent_id} already exists and runs a different spec "
                    f"({agent_spec_id(existing)}). Nothing was overwritten."
                )
            run._adopt_existing_instructions(instructions)
            return run
        ensure_private_dir(run.directory)
        _write_private(run.spec_path, dump_agent_spec(spec), mode=SPEC_MODE)
        if instructions is not None:
            _write_private(run.instructions_path, instructions, mode=FILE_MODE)
        state = AgentState(
            agent_id=agent_id,
            spec_id=agent_spec_id(spec),
            mode=Mode.PAPER,
            approval=approval,
            max_cancels_per_session=max_cancels_per_session,
            session_date=None,
            day_start_equity=None,
            cancels_this_session=0,
            last_tick=None,
            created_at=created_at,
        )
        run.save_state(state)
        return run

    @classmethod
    def load(cls, home: str | os.PathLike[str], agent_id: str) -> AgentRun:
        """An existing agent, with its spec checked against its recorded id."""
        run = cls(home, agent_id)
        if not run.exists:
            raise RuntimeStateError(
                f"no agent {agent_id} under {agents_dir(home)}. Add one with "
                f"`tick agent add <spec.json>`, or list them with `tick agents`."
            )
        state = run.state
        actual = agent_spec_id(run.spec)
        if actual != state.spec_id:
            raise RuntimeStateError(
                f"agent {agent_id}: {run.spec_path} now hashes to {actual}, but this "
                f"agent was created to run {state.spec_id}. The spec an agent executes "
                f"is fixed at creation; add the changed spec as a new agent instead of "
                f"editing this one."
            )
        return run

    @staticmethod
    def list_ids(home: str | os.PathLike[str]) -> list[str]:
        """Every agent id under `home`, sorted. Ignores anything else in there."""
        root = agents_dir(home)
        if not root.is_dir():
            return []
        return sorted(
            entry.name
            for entry in root.iterdir()
            if entry.is_dir() and _AGENT_ID.match(entry.name) and (entry / STATE_FILE).exists()
        )

    # ------------------------------------------------------------------
    # The spec and the state
    # ------------------------------------------------------------------

    @property
    def spec(self) -> AgentSpec:
        """The document this agent runs, read from its own immutable copy.

        Either kind. Which one it is comes from the document itself, not from
        `state.json`, so an agent created before model agents existed reads
        back exactly as it always did.
        """
        if not self.spec_path.exists():
            raise RuntimeStateError(f"agent {self.agent_id} has no {SPEC_FILE}")
        return load_agent_spec_file(self.spec_path)

    @property
    def instructions_path(self) -> Path:
        """Where a model agent's own instructions live. The user writes this file."""
        return self.directory / INSTRUCTIONS_FILE

    def instructions(self) -> str:
        """The user's own instructions, or a refusal naming the file to write.

        Tick has no default to fall back on and will not invent one, so this
        raises rather than returning an empty string that something downstream
        could treat as "no preference".
        """
        try:
            text = self.instructions_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeStateError(
                f"agent {self.agent_id} has no {INSTRUCTIONS_FILE} ({exc}). A model "
                f"agent runs the instructions YOU wrote; Tick ships none. Write them "
                f"to {self.instructions_path} and run it again."
            ) from exc
        if not text.strip():
            raise RuntimeStateError(
                f"agent {self.agent_id}: {self.instructions_path} is empty. A model "
                f"agent runs the instructions YOU wrote, and Tick will not supply one."
            )
        return text

    def _adopt_existing_instructions(self, instructions: str | None) -> None:
        """Refuse to replace instructions already on disk with different ones.

        Adding the same document twice hands back the same agent. If the second
        add points at a different instructions file, that is a change to what
        the agent does, and it is made by editing the file — never silently by
        an `add` that looked like a no-op.
        """
        if instructions is None:
            return
        current = self.instructions_path.read_text(encoding="utf-8")
        if current != instructions:
            raise RuntimeStateError(
                f"agent {self.agent_id} already exists with different instructions. "
                f"Nothing was overwritten: edit {self.instructions_path} directly if "
                f"you want this agent to run different words."
            )

    @property
    def state(self) -> AgentState:
        """`state.json`, parsed. Numbers stay exact on the way in."""
        try:
            text = self.state_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeStateError(
                f"agent {self.agent_id}: cannot read {STATE_FILE}: {exc}"
            ) from exc
        try:
            document = json.loads(text, parse_float=Decimal)
        except json.JSONDecodeError as exc:
            raise RuntimeStateError(
                f"agent {self.agent_id}: {self.state_path} is not JSON: {exc}"
            ) from exc
        try:
            return AgentState.model_validate(document)
        except ValueError as exc:
            raise RuntimeStateError(
                f"agent {self.agent_id}: {self.state_path} is not agent state: {exc}"
            ) from exc

    def save_state(self, state: AgentState) -> AgentState:
        """Write `state.json`. The agent id in it must be this agent's."""
        if state.agent_id != self.agent_id:
            raise RuntimeStateError(
                f"refusing to write state for {state.agent_id} into {self.agent_id}'s directory"
            )
        ensure_private_dir(self.directory)
        payload = json.dumps(state.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        _write_private(self.state_path, payload, mode=FILE_MODE)
        return state

    # ------------------------------------------------------------------
    # The kill switch
    # ------------------------------------------------------------------

    def stop_requested(self) -> bool:
        """True when the kill switch is set. One existence check, no parsing."""
        return self.stop_path.exists()

    def stop_reason(self) -> str:
        """The first line of the STOP file — the reason — or a stated default.

        The first line only: the file also carries when the stop was requested,
        and a notification that quoted both would read as one run-on sentence.
        The whole file is what `tick status` shows.
        """
        try:
            text = self.stop_path.read_text(encoding="utf-8")
        except OSError:
            return "the kill switch is set"
        first = text.strip().splitlines()[0].strip() if text.strip() else ""
        return first or "the kill switch is set"

    def request_stop(self, *, reason: str, at: datetime) -> Path:
        """Set the kill switch. Idempotent: an existing STOP is left alone.

        Left alone on purpose — rewriting it would replace the first reason and
        the first time, which is the pair somebody will want later.
        """
        if not reason.strip():
            raise ValueError("a stop must say why; a halt with no reason is a mystery")
        if at.tzinfo is None or at.utcoffset() is None:
            raise ValueError("a stop time must be timezone-aware")
        ensure_private_dir(self.directory)
        try:
            descriptor = os.open(
                self.stop_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                FILE_MODE,
            )
        except FileExistsError:
            return self.stop_path
        try:
            os.fchmod(descriptor, FILE_MODE)
            os.write(
                descriptor,
                f"{reason.strip()}\nrequested at {at.isoformat()}\n".encode(),
            )
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        directory = os.open(self.directory, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return self.stop_path

    @contextmanager
    def dispatch_gate(self) -> Iterator[bool]:
        """Serialize STOP's final check with entry into a broker operation.

        The caller checks the yielded boolean and enters the broker call while the
        gate is held.  ``request_stop`` creates STOP without needing this lock; after
        its O_EXCL linearization point every later gate observes it.  A broker call
        already inside the gate may complete and must never be retried.
        """
        descriptor = os.open(self.directory / "dispatch.lock", os.O_CREAT | os.O_RDWR, FILE_MODE)
        try:
            os.fchmod(descriptor, FILE_MODE)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield not self.stop_requested()
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    # ------------------------------------------------------------------
    # Ledger generations
    # ------------------------------------------------------------------

    @property
    def first_ledger_path(self) -> Path:
        return agent_ledger_path(self.home, self.agent_id)

    def ledger_paths(self) -> list[Path]:
        """Every ledger generation this agent has, oldest first."""
        first = self.first_ledger_path
        successors = sorted(
            (
                path
                for path in self.directory.glob("records.*.jsonl")
                if _SUCCESSOR.match(path.name)
            ),
            key=lambda path: int(_SUCCESSOR.match(path.name).group(1)),
        )
        return ([first] if first.exists() else []) + successors

    @property
    def ledger_path(self) -> Path:
        """The generation being written to: the newest one that exists."""
        paths = self.ledger_paths()
        return paths[-1] if paths else self.first_ledger_path

    def ledger(self, *, clock: Callable[[], datetime]) -> Ledger:
        """The active ledger, stamped by `clock`. `clock` is required."""
        return Ledger(self.ledger_path, clock=clock)

    def verify_ledger(self) -> VerifyResult:
        """Walk the active generation and report. Never raises for a broken chain."""
        return verify(self.ledger_path)

    def require_verified_ledger(self) -> VerifyResult:
        """The check every tick makes first, and the refusal it raises.

        A runtime that cannot record must not trade: an order placed against a
        record that has already been rewritten is an order with no evidence. So
        this refuses BEFORE anything is placed and before anything is appended,
        and it names the one command that moves things forward.
        """
        result = self.verify_ledger()
        if not result.ok:
            raise LedgerQuarantined(
                self.agent_id,
                seq=result.first_bad_seq,
                reason=result.reason or "the chain does not verify",
                next_step=f"tick ledger new {self.agent_id}",
            )
        return result

    def verified_head(self) -> Record | None:
        """The last record that verifies, even in a ledger that later breaks.

        This is what a successor quotes. Reading stops at the first break, so
        what comes back is the head of the intact prefix — the last thing the
        chain still proves.
        """
        tail: Record | None = None
        try:
            for record in read(self.ledger_path):
                tail = record
        except LedgerCorrupt:
            pass
        return tail

    def start_successor_ledger(self, *, clock: Callable[[], datetime]) -> tuple[Path, Record]:
        """Open the next generation beside a ledger that will not verify.

        Refuses when the current ledger is intact: a successor is how a broken
        chain is left behind, not a way to start a fresh page. The old file is
        kept exactly as it is and made read-only; the new file's first record
        names what was abandoned, at which seq, at which hash, and why.
        """
        current = self.ledger_path
        if not current.exists():
            raise RuntimeStateError(
                f"agent {self.agent_id} has no ledger yet, so there is nothing to succeed."
            )
        result = self.verify_ledger()
        if result.ok:
            raise RuntimeStateError(
                f"agent {self.agent_id}: {current.name} verifies ({result.count} records), "
                f"so it is still the ledger. A successor is started only for a chain that "
                f"cannot be extended; nothing was changed."
            )
        head = self.verified_head()
        generation = _generation_of(current) + 1
        successor = self.directory / f"records.{generation:03d}.jsonl"
        if successor.exists():
            raise RuntimeStateError(
                f"agent {self.agent_id}: {successor.name} already exists. Nothing was overwritten."
            )
        os.chmod(current, SPEC_MODE)
        ledger = Ledger(successor, clock=clock)
        note = ledger.append(
            RecordKind.NOTE,
            {
                "event": "ledger_succeeded",
                "predecessor": current.name,
                "predecessor_verified_records": result.count,
                "predecessor_first_bad_seq": result.first_bad_seq,
                "predecessor_head_seq": head.seq if head is not None else None,
                "predecessor_head_hash": head.hash if head is not None else None,
                "reason": result.reason,
                "text": (
                    f"{current.name} stopped verifying and was left in place, read-only. "
                    f"This ledger continues the agent's record; the two are read together."
                ),
            },
            source=DataSource.RUNTIME,
        )
        return successor, note

    # ------------------------------------------------------------------
    # The cancel-ratio guard
    # ------------------------------------------------------------------

    def cancels_remaining(self, state: AgentState) -> int:
        """How many more cancels this session may attempt."""
        return max(0, state.max_cancels_per_session - state.cancels_this_session)


def _generation_of(path: Path) -> int:
    match = _SUCCESSOR.match(path.name)
    return int(match.group(1)) if match is not None else 1


def _write_private(path: Path, text: str, *, mode: int) -> None:
    """Write `text` to `path` with `mode`, creating it private from the start.

    Not `write_text` then `chmod`: between those two calls the file exists with
    the process umask, and what is in these files is what the user owns.
    """
    ensure_private_dir(path.parent)
    if path.exists():
        os.chmod(path, FILE_MODE)
        path.unlink()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        os.write(descriptor, text.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _check_instructions(spec: AgentSpec, instructions: str | None) -> None:
    """Refuse the two ways an agent could be created with the wrong words.

    A model agent with no instructions would have to acquire them from
    somewhere, and the only somewhere is Tick — which authors none. A rule
    agent with instructions would be carrying words nothing reads, which is a
    person believing their agent does something it does not.
    """
    if is_model_agent(spec):
        if instructions is None:
            raise RuntimeStateError(
                f"{spec.name!r} is a model-driven agent and needs the instructions YOU "
                f"wrote for it. Tick ships none and will not write one. Nothing was "
                f"created."
            )
        if not instructions.strip():
            raise RuntimeStateError(
                f"{spec.name!r} was given empty instructions. A model agent runs the "
                f"words you wrote; whitespace is not an instruction. Nothing was created."
            )
    elif instructions is not None:
        raise RuntimeStateError(
            f"{spec.name!r} is a rule agent: it executes its strategy spec mechanically "
            f"and reads no instructions file. Nothing was created — an agent holding "
            f"words nothing reads is worse than one that refused."
        )


def state_summary(run: AgentRun) -> dict[str, Any]:
    """A plain-JSON view of an agent, for `tick status` and `tick agents`."""
    state = run.state
    spec = run.spec
    model = spec.model if isinstance(spec, ModelAgentSpec) else None
    return {
        "agent_id": run.agent_id,
        "name": spec.name,
        "spec_id": state.spec_id,
        # A deterministic agent is a rule agent; a model-driven one says so and
        # shows the model id. Neither acquires a grander word than it earns.
        "kind": "model_agent" if model is not None else "rule_agent",
        "model": model,
        "universe": list(spec.universe),
        "rules": [rule.id for rule in getattr(spec, "rules", [])],
        "cadence": spec.cadence.kind,
        "mode": state.mode.value,
        "approval": state.approval.value,
        "stopped": run.stop_requested(),
        "session_date": state.session_date.isoformat() if state.session_date else None,
        "day_start_equity": (
            str(state.day_start_equity) if state.day_start_equity is not None else None
        ),
        "cancels_this_session": state.cancels_this_session,
        "max_cancels_per_session": state.max_cancels_per_session,
        "last_tick": state.last_tick.isoformat() if state.last_tick else None,
        "ledger": run.ledger_path.name,
        "ledger_generations": [path.name for path in run.ledger_paths()],
    }
