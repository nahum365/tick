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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

CODEX_RELEASE_TAG = "rust-v0.149.0"
_RELEASE_ROOT = f"https://github.com/openai/codex/releases/download/{CODEX_RELEASE_TAG}"


@dataclass(frozen=True, slots=True)
class PinnedAsset:
    """One release archive and the exact executable it is allowed to install."""

    archive: str
    sha256: str
    member: str
    destination: str


#: Linux builds only: the managed box is a Linux droplet. Codex and its Code Mode
#: host are release-matched because chat MCP calls require both in rust-v0.149.0.
CODEX_ASSETS: Mapping[str, tuple[PinnedAsset, ...]] = {
    "x86_64": (
        PinnedAsset(
            archive="codex-x86_64-unknown-linux-musl.tar.gz",
            sha256="7368b2055ed02157fea2695bb9f5af3ee7b0e40c5a3bebc81dfc596704244cfd",
            member="codex-x86_64-unknown-linux-musl",
            destination="codex",
        ),
        PinnedAsset(
            archive="codex-code-mode-host-x86_64-unknown-linux-musl.tar.gz",
            sha256="3600a45ac2b09fe3c995f4f49860131fea388b46c409c82a0266fc4d0342a04c",
            member="codex-code-mode-host-x86_64-unknown-linux-musl",
            destination="codex-code-mode-host",
        ),
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


def codex_code_mode_host_path(home: Path) -> Path:
    return codex_bin_dir(home) / "codex-code-mode-host"


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
) -> Mapping[str, Any]:
    """Fetch, verify and place both release-matched executables or refuse."""
    system = system or platform.system()
    machine = machine or platform.machine()
    if system != "Linux" or machine not in CODEX_ASSETS:
        raise CodexInstallError(
            "CODEX_UNSUPPORTED_PLATFORM",
            f"Tick installs Codex only on Linux x86_64 boxes; this box is {system}/{machine}. "
            "Install the Codex CLI yourself and start device login again.",
        )
    binaries: list[tuple[PinnedAsset, bytes, str]] = []
    for asset in CODEX_ASSETS[machine]:
        url = f"{_RELEASE_ROOT}/{asset.archive}"
        try:
            payload = fetch(url)
        except Exception as error:  # noqa: BLE001 - the reason is shown to the user
            raise CodexInstallError(
                "CODEX_DOWNLOAD_FAILED",
                f"the box could not download {asset.archive} ({type(error).__name__}). Check "
                "the box's outbound network, then tap Install again.",
            ) from error
        if len(payload) > MAX_ARCHIVE_BYTES:
            raise CodexInstallError(
                "CODEX_ARCHIVE_UNEXPECTED",
                "a downloaded Codex archive is larger than any published release. Nothing "
                "was installed; tap Install again later or install both executables yourself.",
            )
        digest = hashlib.sha256(payload).hexdigest()
        if digest != asset.sha256:
            raise CodexInstallError(
                "CODEX_CHECKSUM_MISMATCH",
                f"the downloaded {asset.archive} does not match the pinned SHA-256 for "
                f"{CODEX_RELEASE_TAG}. Nothing was installed; tap Install again or install "
                "both executables yourself.",
            )
        binaries.append((asset, _single_binary(payload, member=asset.member), digest))

    target_dir = codex_bin_dir(home)
    target_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    target_dir.chmod(0o700)
    installed: dict[str, dict[str, str]] = {}
    staged: list[tuple[Path, Path, PinnedAsset, str]] = []
    executable_mode = stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
    for asset, binary, digest in binaries:
        target = target_dir / asset.destination
        partial = target.with_name(f"{asset.destination}.partial")
        partial.write_bytes(binary)
        partial.chmod(executable_mode)
        staged.append((partial, target, asset, digest))
    for partial, target, asset, digest in staged:
        partial.replace(target)
        installed[asset.destination] = {
            "path": str(target),
            "release": CODEX_RELEASE_TAG,
            "sha256": digest,
        }
    codex = installed["codex"]
    return {
        "code": "CODEX_INSTALLED",
        "path": codex["path"],
        "release": CODEX_RELEASE_TAG,
        "sha256": codex["sha256"],
        "paths": {name: receipt["path"] for name, receipt in installed.items()},
        "assets": installed,
        "reason": (
            f"Codex and its Code Mode host from {CODEX_RELEASE_TAG} are installed on the box. "
            "You can start device login."
        ),
    }


def _single_binary(payload: bytes, *, member: str) -> bytes:
    """The release tarball holds exactly one regular file, the binary."""
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
            members = [m for m in archive.getmembers() if m.isfile()]
            if (
                len(members) != 1
                or members[0].name != member
                or ".." in members[0].name
                or members[0].name.startswith("/")
            ):
                raise CodexInstallError(
                    "CODEX_ARCHIVE_UNEXPECTED",
                    f"a Codex archive did not contain exactly {member}. Nothing was installed; "
                    "install both executables yourself or retry after a Tick update.",
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
