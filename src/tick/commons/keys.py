"""Generate and load the pseudonymous Ed25519 contributor key on the box."""

from __future__ import annotations

import base64
import os
from pathlib import Path

from nacl.signing import SigningKey

from tick.records.home import PRIVATE_FILE_MODE, tick_home, write_private_file

KEY_PATH = Path("commons/key")


def _encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _decode(text: str) -> bytes:
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)


def contributor_id(key: SigningKey) -> str:
    """The public key is the contributor's unlinkable commons identity."""
    return _encode(bytes(key.verify_key))


def generate_key(home: Path | None = None) -> SigningKey:
    """Create a private key under `TICK_HOME`, refusing to replace one."""
    root = home if home is not None else tick_home(os.environ)
    path = root / KEY_PATH
    if path.exists():
        raise FileExistsError(
            f"a commons key already exists at {path}; run `tick commons keygen` only on a new box"
        )
    key = SigningKey.generate()
    write_private_file(path, _encode(bytes(key)) + "\n")
    return key


def load_key(home: Path | None = None) -> SigningKey:
    """Load the box key and reassert its owner-only file mode."""
    root = home if home is not None else tick_home(os.environ)
    path = root / KEY_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"no commons key exists at {path}; run `tick commons keygen` and try again"
        )
    os.chmod(path, PRIVATE_FILE_MODE)
    try:
        return SigningKey(_decode(path.read_text(encoding="utf-8").strip()))
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"the commons key at {path} is unreadable; restore it from backup or create a new box"
        ) from exc
