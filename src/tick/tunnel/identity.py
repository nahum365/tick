"""Derive the tunnel identity from the pairing capability on the same box."""

from __future__ import annotations

import base64
import hashlib
import hmac

from tick.serve.pairing import PairingError

INFO = b"tick-tunnel-v1"
KEY_BYTES = 32


def _pairing_bytes(pairing_secret: str) -> bytes:
    try:
        decoded = base64.b64decode(pairing_secret + "=", altchars=b"-_", validate=True)
    except ValueError as exc:
        raise PairingError(
            "pairing_secret_invalid",
            "the pairing secret cannot derive a tunnel identity. Run `tick pair rotate` "
            "on the box, then pair again.",
        ) from exc
    if len(decoded) != KEY_BYTES:
        raise PairingError(
            "pairing_secret_weak",
            "the pairing secret cannot derive a 32-byte tunnel identity. Run `tick pair "
            "rotate` on the box, then pair again.",
        )
    return decoded


def derive_secret_key_bytes(pairing_secret: str) -> bytes:
    """HKDF binds the endpoint identity to the private pairing capability.

    RFC 5869 uses an all-zero hash-length salt when no salt is supplied. Keeping
    this tiny implementation beside the invariant makes the Python and CryptoKit
    derivations reviewable byte for byte without adding another dependency.
    """
    ikm = _pairing_bytes(pairing_secret)
    pseudorandom_key = hmac.new(bytes(hashlib.sha256().digest_size), ikm, hashlib.sha256).digest()
    return hmac.new(pseudorandom_key, INFO + b"\x01", hashlib.sha256).digest()


def endpoint_id_for_pairing_secret(pairing_secret: str) -> str:
    """Return the public id without logging or persisting the pairing value."""
    import iroh

    return str(iroh.SecretKey.from_bytes(derive_secret_key_bytes(pairing_secret)).public())
