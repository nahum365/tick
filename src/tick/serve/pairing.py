"""Private high-entropy pairing credentials for the box API."""

from __future__ import annotations

import base64
import os
import secrets
from pathlib import Path

from tick.records import ensure_private_dir

__all__ = [
    "PairingError",
    "create_secret",
    "load_secret",
    "pairing_secret_path",
    "rotate_secret",
]

SECRET_BYTES = 32
ENCODED_LENGTH = 43
FILE_MODE = 0o600


class PairingError(Exception):
    """A credential refusal whose sentence never includes the credential."""

    def __init__(self, code: str, reason: str) -> None:
        self.code = code
        self.reason = reason
        super().__init__(f"{code}: {reason}")


def pairing_secret_path(home: str | os.PathLike[str]) -> Path:
    return Path(home) / "pairing" / "secret"


def _generate() -> str:
    return base64.urlsafe_b64encode(secrets.token_bytes(SECRET_BYTES)).rstrip(b"=").decode("ascii")


def _validate(value: str) -> str:
    if len(value) != ENCODED_LENGTH or "=" in value:
        raise PairingError(
            "pairing_secret_weak",
            "the pairing secret is not the fixed 32-byte base64url form. Run `tick pair "
            "rotate` on the box before serving operational requests.",
        )
    try:
        decoded = base64.b64decode(value + "=", altchars=b"-_", validate=True)
    except ValueError as exc:
        raise PairingError(
            "pairing_secret_invalid",
            "the pairing secret is not valid base64url. Run `tick pair rotate` on the box.",
        ) from exc
    if len(decoded) != SECRET_BYTES:
        raise PairingError(
            "pairing_secret_weak",
            "the pairing secret has less than 32 bytes of entropy. Run `tick pair rotate` "
            "on the box before serving operational requests.",
        )
    canonical = base64.urlsafe_b64encode(decoded).rstrip(b"=").decode("ascii")
    if canonical != value:
        raise PairingError(
            "pairing_secret_invalid",
            "the pairing secret is not canonical base64url. Run `tick pair rotate` on the box.",
        )
    return value


def load_secret(home: str | os.PathLike[str]) -> str:
    """Read and validate the credential without ever including it in an error."""
    path = pairing_secret_path(home)
    try:
        value = path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as exc:
        raise PairingError(
            "pairing_secret_missing",
            f"no pairing secret can be read at {path} ({exc}). Run `tick pair new` on the box.",
        ) from exc
    mode = path.stat().st_mode & 0o777
    if mode != FILE_MODE:
        raise PairingError(
            "pairing_secret_permissions",
            f"the pairing secret at {path} has mode {mode:04o}, not 0600. Fix its "
            "permissions before serving requests.",
        )
    return _validate(value)


def create_secret(home: str | os.PathLike[str]) -> tuple[Path, str]:
    """Create once and return the value so the CLI can print it exactly once."""
    path = pairing_secret_path(home)
    ensure_private_dir(path.parent)
    value = _generate()
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, FILE_MODE)
    except FileExistsError as exc:
        raise PairingError(
            "pairing_secret_exists",
            f"a pairing secret already exists at {path}. Use `tick pair rotate` to replace it.",
        ) from exc
    try:
        os.fchmod(descriptor, FILE_MODE)
        os.write(descriptor, (value + "\n").encode("ascii"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)
    return path, value


def rotate_secret(home: str | os.PathLike[str]) -> tuple[Path, str]:
    """Atomically replace the credential; the next request reloads this file."""
    path = pairing_secret_path(home)
    ensure_private_dir(path.parent)
    value = _generate()
    temporary = path.with_name(".secret.rotate")
    try:
        temporary.unlink()
    except FileNotFoundError:
        pass
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, FILE_MODE)
    try:
        os.fchmod(descriptor, FILE_MODE)
        os.write(descriptor, (value + "\n").encode("ascii"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    os.chmod(path, FILE_MODE)
    _fsync_directory(path.parent)
    return path, value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
