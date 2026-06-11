"""Ed25519 signing/verification of promoted global hash records."""

from __future__ import annotations

import base64

from optimus.globaldb.signing import (
    HashRecord,
    canonical_bytes,
    generate_keypair,
    sign_record,
    verify_record,
)


def _record() -> HashRecord:
    return HashRecord(hash_id="abc", phash=1, dhash=2, whash=3, campaign_id="camp")


def test_sign_then_verify_roundtrip() -> None:
    priv, pub = generate_keypair()
    sig = sign_record(_record(), priv)
    assert verify_record(_record(), sig, pub) is True


def test_verify_rejects_tampered_record() -> None:
    priv, pub = generate_keypair()
    sig = sign_record(_record(), priv)
    tampered = HashRecord(hash_id="abc", phash=999, dhash=2, whash=3, campaign_id="camp")
    assert verify_record(tampered, sig, pub) is False


def test_verify_rejects_wrong_key() -> None:
    priv, _ = generate_keypair()
    _, other_pub = generate_keypair()
    sig = sign_record(_record(), priv)
    assert verify_record(_record(), sig, other_pub) is False


def test_verify_rejects_missing_signature() -> None:
    _, pub = generate_keypair()
    assert verify_record(_record(), None, pub) is False
    assert verify_record(_record(), "", pub) is False


def test_verify_rejects_missing_public_key() -> None:
    priv, _ = generate_keypair()
    sig = sign_record(_record(), priv)
    assert verify_record(_record(), sig, "") is False


def test_verify_rejects_malformed_base64() -> None:
    _, pub = generate_keypair()
    assert verify_record(_record(), "not!base64!!", pub) is False


def test_canonical_bytes_is_deterministic_and_sorted() -> None:
    a = canonical_bytes(_record())
    b = canonical_bytes(_record())
    assert a == b
    # Keys must be sorted so consumers verify byte-for-byte.
    assert a.index(b'"campaign_id"') < a.index(b'"hash_id"') < a.index(b'"phash"')


def test_sign_requires_private_key() -> None:
    import pytest

    with pytest.raises(ValueError, match="signing private key"):
        sign_record(_record(), "")


def test_generated_keys_are_valid_base64() -> None:
    priv, pub = generate_keypair()
    assert len(base64.b64decode(priv)) == 32
    assert len(base64.b64decode(pub)) == 32
