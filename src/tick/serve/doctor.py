"""Operating-system observations for the otherwise pure doctor checklist."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from tick.serve.pairing import PairingError, load_secret

__all__ = ["codex_login_status", "loopback_status", "systemd_unit_fragments"]


def codex_login_status() -> tuple[bool, str]:
    """Require both release executables, then ask the CLI about its login."""
    executable = shutil.which("codex")
    if executable is None:
        return (
            False,
            "run `tick provider install codex` to install Codex and its Code Mode host, then "
            "run `codex login` and doctor again.",
        )
    if shutil.which("codex-code-mode-host") is None:
        return (
            False,
            "the Codex Code Mode host is missing. Run `tick provider install codex`, then run "
            "doctor again.",
        )
    help_result = subprocess.run(  # noqa: S603
        [executable, "login", "--help"], capture_output=True, text=True, check=False, timeout=10
    )
    help_text = help_result.stdout + help_result.stderr
    if "status" in help_text.lower():
        result = subprocess.run(  # noqa: S603
            [executable, "login", "status"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        sentence = " ".join((result.stdout or result.stderr).split())
        if result.returncode == 0:
            return True, sentence or "codex reports a logged-in session."
        return False, (sentence or "codex reports no login") + ". Run `codex login`."
    auth = Path.home() / ".codex" / "auth.json"
    if auth.exists():
        return (
            True,
            "codex is installed; ~/.codex/auth.json is present (its contents were not read).",
        )
    return False, "codex is installed but ~/.codex/auth.json is absent; run `codex login`."


def loopback_status(home: Path, port: int) -> tuple[bool, str]:
    """Make one authenticated loopback status request and expose no credential."""
    try:
        secret = load_secret(home)
    except PairingError as exc:
        return False, exc.reason
    request = Request(
        f"http://127.0.0.1:{port}/v1/status",
        headers={"Authorization": f"Bearer {secret}"},
    )
    try:
        with urlopen(request, timeout=2) as response:  # noqa: S310 - fixed loopback target
            if response.status == 200:
                return True, f"tick serve is reachable on authenticated loopback port {port}."
            return False, f"tick serve answered {response.status}; inspect it locally and retry."
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        return False, f"start `tick serve --bind 127.0.0.1 --port {port}` ({exc})."


def systemd_unit_fragments() -> tuple[bool, str, tuple[str, ...]]:
    """Read enabled Tick unit text so persistent live authority is visible."""
    roots = (Path("/etc/systemd/system"), Path("/run/systemd/system"))
    paths: list[Path] = []
    for root in roots:
        paths.extend(root.glob("tick*.service"))
        paths.extend(root.glob("tick*.service.d/*.conf"))
    fragments: list[str] = []
    for path in sorted(set(paths)):
        try:
            fragments.append(f"{path}:\n{path.read_text(encoding='utf-8')}")
        except OSError:
            continue
    if not fragments:
        return (
            False,
            "no enabled Tick unit fragments were readable; inspect systemd before live use.",
            (),
        )
    return (
        True,
        f"inspected {len(fragments)} Tick unit fragment(s); no persistent --live was found.",
        tuple(fragments),
    )
