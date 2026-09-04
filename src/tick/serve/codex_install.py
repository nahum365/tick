"""Install the pinned Codex CLI release on the box, on the user's behalf.

Why this exists: a managed box is created without Codex, and the owner's
first live run stopped at ``codex_not_installed``. The box fetches one
pinned release from the CLI's own GitHub releases, verifies its SHA-256,
and places the binary under ``TICK_HOME/bin``; the CLI prepends that
directory to ``PATH`` so every ``shutil.which("codex")`` in the runtime
finds it. No credential is involved: login stays the CLI's own device flow.
"""

from __future__ import annotations

import hashlib
import io
import platform
import stat
import tarfile
from collections.abc import Callable, Mapping
from pathlib import Path

import httpx

CODEX_RELEASE_TAG = "rust-v0.149.0"
_RELEASE_ROOT = f"https://github.com/openai/codex/releases/download/{CODEX_RELEASE_TAG}"
#: Linux builds only: the managed box is a Linux droplet. Each entry is the asset
#: name and the SHA-256 of that exact file, verified before anything is written.
CODEX_ASSETS: Mapping[str, tuple[str, str]] = {
    "x86_64": (
        "codex-x86_64-unknown-linux-musl.tar.gz",
        "7368b2055ed02157fea2695bb9f5af3ee7b0e40c5a3bebc81dfc596704244cfd",
    ),
}
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024


class CodexInstallError(Exception):
    """A refusal with a stable code and a sentence saying what the user can still do."""

    def __init__(self, code: str, reason: str) -> None:
        super().__init__(reason)
        self.code = code
        self.reason = reason


def codex_bin_dir(home: Path) -> Path:
    return home / "bin"


def codex_binary_path(home: Path) -> Path:
    return codex_bin_dir(home) / "codex"


def default_fetch(url: str) -> bytes:
    """Download one release asset; the only network call this module makes."""
    with httpx.Client(follow_redirects=True, timeout=120.0) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.content


def install_codex(
    home: Path,
    *,
    fetch: Callable[[str], bytes],
    machine: str | None = None,
    system: str | None = None,
) -> Mapping[str, str]:
    """Fetch, verify and place the pinned Codex binary; refuse loudly otherwise."""
    system = system or platform.system()
    machine = machine or platform.machine()
    if system != "Linux" or machine not in CODEX_ASSETS:
        raise CodexInstallError(
            "CODEX_UNSUPPORTED_PLATFORM",
            f"Tick installs Codex only on Linux x86_64 boxes; this box is {system}/{machine}. "
            "Install the Codex CLI yourself and start device login again.",
        )
    asset, expected = CODEX_ASSETS[machine]
    url = f"{_RELEASE_ROOT}/{asset}"
    try:
        payload = fetch(url)
    except Exception as error:  # noqa: BLE001 - the reason is shown to the user
        raise CodexInstallError(
            "CODEX_DOWNLOAD_FAILED",
            f"the box could not download {asset} ({type(error).__name__}). Check the box's "
            "outbound network, then tap Install again.",
        ) from error
    if len(payload) > MAX_ARCHIVE_BYTES:
        raise CodexInstallError(
            "CODEX_ARCHIVE_UNEXPECTED",
            "the downloaded Codex archive is larger than any published release. Nothing was "
            "installed; tap Install again later or install the CLI yourself.",
        )
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected:
        raise CodexInstallError(
            "CODEX_CHECKSUM_MISMATCH",
            f"the downloaded Codex archive does not match the pinned SHA-256 for "
            f"{CODEX_RELEASE_TAG}. Nothing was installed; tap Install again or install the CLI "
            "yourself.",
        )
    binary = _single_binary(payload)
    target_dir = codex_bin_dir(home)
    target_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
    target = codex_binary_path(home)
    staged = target.with_name("codex.partial")
    staged.write_bytes(binary)
    staged.chmod(stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    staged.replace(target)
    return {
        "code": "CODEX_INSTALLED",
        "path": str(target),
        "release": CODEX_RELEASE_TAG,
        "sha256": digest,
        "reason": f"Codex {CODEX_RELEASE_TAG} is installed on the box. You can start device login.",
    }


def _single_binary(payload: bytes) -> bytes:
    """The release tarball holds exactly one regular file, the binary."""
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
            members = [m for m in archive.getmembers() if m.isfile()]
            if len(members) != 1 or any(".." in m.name or m.name.startswith("/") for m in members):
                raise CodexInstallError(
                    "CODEX_ARCHIVE_UNEXPECTED",
                    "the Codex archive did not contain exactly one binary. Nothing was "
                    "installed; install the CLI yourself or retry after a Tick update.",
                )
            extracted = archive.extractfile(members[0])
            if extracted is None:
                raise CodexInstallError(
                    "CODEX_ARCHIVE_UNEXPECTED",
                    "the Codex archive member could not be read. Nothing was installed.",
                )
            return extracted.read()
    except tarfile.TarError as error:
        raise CodexInstallError(
            "CODEX_ARCHIVE_UNEXPECTED",
            "the downloaded Codex archive is not a readable tarball. Nothing was installed; "
            "tap Install again.",
        ) from error
