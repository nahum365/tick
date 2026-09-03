"""The local token store — the file that makes invariant 1 an architecture.

CLAUDE.md invariant 1: credentials never leave the user's box. That is not a
promise this file makes, it is a shape it has. The OAuth access token, the
refresh token and the dynamically registered client credentials are written to
two files under `TICK_HOME/robinhood/`, mode 0600, and there is no second
writer, no queue, no upload and no Tick-side store to receive them. The
package that holds these bytes reaches Robinhood and the loopback address and
nothing else, which `tests/test_product_constraints.py` scans for.

Three details are deliberate:

- **The mode is set at creation, not after.** A file created world-readable
  and chmodded a microsecond later was world-readable for a microsecond, and
  on a shared machine that is the whole attack. `os.open` with `O_CREAT` and
  `0o600` never has the wider mode.
- **A write replaces the file's contents in place, and the mode is re-asserted
  every time.** A refresh rewrites the token file several times a day; a store
  that only got the mode right on the first write would be one `umask` away
  from leaking on the second.
- **A store that finds a wider mode refuses to read.** If something has made
  the file group- or world-readable, the honest response is to stop and say
  so. Reading it anyway would let the runtime carry on using a credential the
  machine has already shown to someone else.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from pydantic import ValidationError

from tick.records import PRIVATE_FILE_MODE, ensure_private_dir, write_private_file

from .errors import TokenStoreError

__all__ = [
    "CLIENT_FILE",
    "FileTokenStorage",
    "ROBINHOOD_DIR",
    "TOKEN_FILE",
    "TOKEN_FILE_MODE",
    "robinhood_dir",
]

#: Everything this slice keeps lives in one directory under `TICK_HOME`.
ROBINHOOD_DIR = "robinhood"

#: The OAuth grant: access token, refresh token, scope, expiry.
TOKEN_FILE = "tokens.json"

#: What dynamic client registration handed back — a client id, and on some
#: servers a secret. Registration is per machine, so it is as local as the token.
CLIENT_FILE = "client.json"

#: Readable and writable by the owner, by nobody else, ever. The same mode
#: every file under `TICK_HOME` that describes the user's brokerage gets.
TOKEN_FILE_MODE = PRIVATE_FILE_MODE


def robinhood_dir(home: str | os.PathLike[str]) -> Path:
    """`<home>/robinhood` — created private if it is not there yet."""
    return ensure_private_dir(Path(home) / ROBINHOOD_DIR)


def _read_private(path: Path) -> str | None:
    """Read `path`, or refuse if anyone but the owner can read it."""
    if not path.exists():
        return None
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise TokenStoreError(
            f"{path} is mode {mode:04o}: it is readable or writable by someone other "
            f"than you. Tick will not use a credential the machine has already shown "
            f"to another account. Run `chmod 600 {path}` if you trust this machine, or "
            f"delete it and connect again."
        )
    return path.read_text(encoding="utf-8")


class FileTokenStorage:
    """`TokenStorage` over two files under `TICK_HOME/robinhood/`.

    `home` is required and is the `TICK_HOME` the runtime resolved. There is no
    default: a store that picked its own directory would be a second place
    credentials can land, and the whole point of this class is that there is
    exactly one.
    """

    def __init__(self, home: str | os.PathLike[str]) -> None:
        self._dir = Path(home) / ROBINHOOD_DIR

    @property
    def directory(self) -> Path:
        return self._dir

    @property
    def token_path(self) -> Path:
        return self._dir / TOKEN_FILE

    @property
    def client_path(self) -> Path:
        return self._dir / CLIENT_FILE

    def connected(self) -> bool:
        """Whether a grant is on disk. Says nothing about whether it still works."""
        return self.token_path.exists()

    def forget(self) -> list[Path]:
        """Delete the grant and the registered client; return what was removed.

        The user must be able to end this at any moment without editing files
        by hand, and the deletion is local because the credential is local:
        there is nowhere else to revoke it from on Tick's side.
        """
        removed: list[Path] = []
        for path in (self.token_path, self.client_path):
            if path.exists():
                path.unlink()
                removed.append(path)
        return removed

    # ------------------------------------------------------------------
    # mcp.client.auth.TokenStorage
    # ------------------------------------------------------------------

    async def get_tokens(self) -> OAuthToken | None:
        raw = _read_private(self.token_path)
        if raw is None:
            return None
        return self._parse(OAuthToken, raw, self.token_path, "OAuth grant")

    async def set_tokens(self, tokens: OAuthToken) -> None:
        write_private_file(self.token_path, tokens.model_dump_json(indent=2, exclude_none=True))

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        raw = _read_private(self.client_path)
        if raw is None:
            return None
        return self._parse(OAuthClientInformationFull, raw, self.client_path, "registered client")

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        write_private_file(
            self.client_path, client_info.model_dump_json(indent=2, exclude_none=True)
        )

    @staticmethod
    def _parse(model, raw: str, path: Path, what: str):
        """Validate `raw` as `model`, or refuse — never a half-understood credential."""
        try:
            return model.model_validate_json(raw)
        except ValidationError as exc:
            raise TokenStoreError(
                f"{path} does not hold a readable {what}: {exc}. Delete it and run "
                f"`tick connect robinhood` again; nothing else on this machine reads "
                f"that file."
            ) from exc
