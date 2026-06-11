"""Ed25519 signing and verification for promoted global hash records.

The signed payload is a canonical, sorted JSON encoding of the identifying
fields (``hash_id`` and the three perceptual hashes) plus the ``status`` and
``campaign_id``. Canonicalization is deterministic so a record signed on the
authority verifies byte-for-byte on every consumer. Keys are Ed25519, encoded
base64 for transport in environment variables / config.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass

from nacl.exceptions import BadSignatureError
from nacl.signing import SigningKey, VerifyKey


@dataclass(frozen=True, slots=True)
class HashRecord:
    """The signable identity of a global hash record."""

    hash_id: str
    phash: int
    dhash: int
    whash: int
    status: str = "promoted"
    campaign_id: str | None = None


def canonical_bytes(record: HashRecord) -> bytes:
    """Return the deterministic byte encoding that is signed and verified."""
    payload = {
        "hash_id": record.hash_id,
        "phash": record.phash,
        "dhash": record.dhash,
        "whash": record.whash,
        "status": record.status,
        "campaign_id": record.campaign_id,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def generate_keypair() -> tuple[str, str]:
    """Generate a new Ed25519 keypair as ``(private_b64, public_b64)``.

    Used by deployment tooling on the signing authority; never at request time.
    """
    signing_key = SigningKey.generate()
    private_b64 = base64.b64encode(bytes(signing_key)).decode("ascii")
    public_b64 = base64.b64encode(bytes(signing_key.verify_key)).decode("ascii")
    return private_b64, public_b64


def sign_record(record: HashRecord, private_key_b64: str) -> str:
    """Sign ``record`` with the base64 Ed25519 private key; return base64 sig."""
    if not private_key_b64:
        raise ValueError("signing private key is not configured")
    signing_key = SigningKey(base64.b64decode(private_key_b64))
    signature = signing_key.sign(canonical_bytes(record)).signature
    return base64.b64encode(signature).decode("ascii")


def verify_record(record: HashRecord, signature_b64: str | None, public_key_b64: str) -> bool:
    """Verify ``record`` against ``signature_b64`` using the base64 public key.

    Returns ``False`` (never raises) for a missing signature, a missing public
    key, malformed base64, or a signature that does not match — so a consumer
    can safely reject any record that fails to verify.
    """
    if not signature_b64 or not public_key_b64:
        return False
    try:
        verify_key = VerifyKey(base64.b64decode(public_key_b64))
        verify_key.verify(canonical_bytes(record), base64.b64decode(signature_b64))
    except (BadSignatureError, ValueError, TypeError):
        return False
    return True
