"""Ed25519 signing and verification for promoted global hash records.

The signed payload is a canonical, sorted JSON encoding of the identifying
fields (``hash_id`` and the three perceptual hashes) plus the ``status`` and
``campaign_id``. Canonicalization is deterministic so a record signed on the
authority verifies byte-for-byte on every consumer. Keys are Ed25519, encoded
base64 for transport in environment variables / config.

**Key lifecycle.** A signature is stored as ``"{key_id}:{signature_b64}"`` so
the verifier knows which key signed it. The ``key_id`` is folded into the signed
payload, binding the signature to that key. A :class:`Keyring` holds every
currently-valid public key (keyed by id) so keys can be *rotated* with an
overlap window (old and new both valid) and individually *revoked* (a revoked
id is never trusted even if its public key is still listed). Signatures written
before versioning carry no ``key_id`` (a bare base64 string); those verify
against the keyring's ``legacy_public_key_b64`` using the pre-versioning payload
encoding, so old records keep verifying after the upgrade.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from dataclasses import dataclass, field

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


@dataclass(frozen=True, slots=True)
class Keyring:
    """The set of public keys a consumer will trust for global-hash records.

    * ``keys`` maps ``key_id`` -> base64 public key; every entry is currently
      valid, which is what enables rotation with an overlap window.
    * ``revoked`` is a set of ``key_id`` values that must never be trusted, even
      if still present in ``keys`` — revocation wins over presence.
    * ``legacy_public_key_b64`` verifies pre-versioning signatures that carry no
      ``key_id`` (kept for backward-compatible migration).
    """

    keys: Mapping[str, str] = field(default_factory=dict)
    revoked: frozenset[str] = frozenset()
    legacy_public_key_b64: str = ""

    def public_key(self, key_id: str | None) -> str:
        """Return the trusted public key for ``key_id`` (``""`` if not trusted).

        ``None`` selects the legacy key. A revoked or unknown ``key_id`` yields
        ``""`` so verification fails closed.
        """
        if key_id is None:
            return self.legacy_public_key_b64
        if key_id in self.revoked:
            return ""
        return self.keys.get(key_id, "")


def canonical_bytes(record: HashRecord, *, key_id: str | None = None) -> bytes:
    """Return the deterministic byte encoding that is signed and verified.

    When ``key_id`` is given it is included in the payload, binding the
    signature to that key. ``None`` reproduces the pre-versioning encoding
    byte-for-byte so legacy signatures still verify.
    """
    payload: dict[str, object] = {
        "hash_id": record.hash_id,
        "phash": record.phash,
        "dhash": record.dhash,
        "whash": record.whash,
        "status": record.status,
        "campaign_id": record.campaign_id,
    }
    if key_id is not None:
        payload["key_id"] = key_id
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _split_signature(signature_b64: str) -> tuple[str | None, str]:
    """Split a stored signature into ``(key_id, signature_b64)``.

    Versioned signatures are ``"{key_id}:{sig}"``; a bare base64 string (no
    ``:``, which never appears in base64) is a legacy signature -> ``key_id`` is
    ``None``.
    """
    key_id, sep, sig = signature_b64.partition(":")
    if not sep:
        return None, signature_b64
    return key_id, sig


def generate_keypair() -> tuple[str, str]:
    """Generate a new Ed25519 keypair as ``(private_b64, public_b64)``.

    Used by deployment tooling on the signing authority; never at request time.
    """
    signing_key = SigningKey.generate()
    private_b64 = base64.b64encode(bytes(signing_key)).decode("ascii")
    public_b64 = base64.b64encode(bytes(signing_key.verify_key)).decode("ascii")
    return private_b64, public_b64


def sign_record(record: HashRecord, private_key_b64: str, *, key_id: str | None = None) -> str:
    """Sign ``record`` with the base64 Ed25519 private key; return the signature.

    With ``key_id`` the result is ``"{key_id}:{signature_b64}"`` and the id is
    folded into the signed payload. Without it (``None``) the legacy bare-base64
    format is produced for backward compatibility. ``key_id`` must not contain
    ``":"`` (the field separator).
    """
    if not private_key_b64:
        raise ValueError("signing private key is not configured")
    if key_id is not None and ":" in key_id:
        raise ValueError("key_id must not contain ':'")
    signing_key = SigningKey(base64.b64decode(private_key_b64))
    signature = signing_key.sign(canonical_bytes(record, key_id=key_id)).signature
    signature_b64 = base64.b64encode(signature).decode("ascii")
    if key_id is None:
        return signature_b64
    return f"{key_id}:{signature_b64}"


def verify_record(record: HashRecord, signature_b64: str | None, keys: str | Keyring) -> bool:
    """Verify ``record`` against ``signature_b64`` using the trusted key(s).

    ``keys`` may be a :class:`Keyring` (rotation/revocation aware) or a single
    base64 public key, which is treated as the legacy key. Returns ``False``
    (never raises) for a missing signature, a key that is unknown/revoked/absent,
    malformed base64, or a signature that does not match — so a consumer can
    safely reject any record that fails to verify.
    """
    if not signature_b64:
        return False
    keyring = keys if isinstance(keys, Keyring) else Keyring(legacy_public_key_b64=keys or "")
    key_id, sig = _split_signature(signature_b64)
    public_key_b64 = keyring.public_key(key_id)
    if not public_key_b64:
        return False
    try:
        verify_key = VerifyKey(base64.b64decode(public_key_b64))
        verify_key.verify(canonical_bytes(record, key_id=key_id), base64.b64decode(sig))
    except (BadSignatureError, ValueError, TypeError):
        return False
    return True
