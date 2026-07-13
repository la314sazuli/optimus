"""Ed25519 signing/verification of promoted global hash records."""

from __future__ import annotations

import base64

from optimus.globaldb.signing import (
    HashRecord,
    Keyring,
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


# --- key lifecycle: id/versioning, rotation, revocation, migration ---


def test_versioned_signature_carries_key_id() -> None:
    priv, pub = generate_keypair()
    sig = sign_record(_record(), priv, key_id="k1")
    assert sig.startswith("k1:")
    assert verify_record(_record(), sig, Keyring(keys={"k1": pub})) is True


def test_verify_across_rotation_overlap_window() -> None:
    # Two keys valid at once: records signed under either verify.
    priv1, pub1 = generate_keypair()
    priv2, pub2 = generate_keypair()
    keyring = Keyring(keys={"k1": pub1, "k2": pub2})
    sig_old = sign_record(_record(), priv1, key_id="k1")
    sig_new = sign_record(_record(), priv2, key_id="k2")
    assert verify_record(_record(), sig_old, keyring) is True
    assert verify_record(_record(), sig_new, keyring) is True


def test_verify_rejects_revoked_key_id() -> None:
    priv, pub = generate_keypair()
    sig = sign_record(_record(), priv, key_id="k1")
    # Key still listed, but revoked → rejected (revocation wins over presence).
    keyring = Keyring(keys={"k1": pub}, revoked=frozenset({"k1"}))
    assert verify_record(_record(), sig, keyring) is False


def test_verify_rejects_unknown_key_id() -> None:
    priv, pub = generate_keypair()
    sig = sign_record(_record(), priv, key_id="k1")
    assert verify_record(_record(), sig, Keyring(keys={"k2": pub})) is False


def test_verify_rejects_key_id_swap_replay() -> None:
    # An attacker relabels a k1 signature as k2 to dodge a k1 revocation. The
    # key_id is inside the signed payload, so the bytes no longer verify.
    priv1, pub1 = generate_keypair()
    _, pub2 = generate_keypair()
    sig = sign_record(_record(), priv1, key_id="k1")
    _, sig_b64 = sig.split(":", 1)
    forged = f"k2:{sig_b64}"
    keyring = Keyring(keys={"k1": pub1, "k2": pub2}, revoked=frozenset({"k1"}))
    assert verify_record(_record(), forged, keyring) is False


def test_verify_rejects_tampered_versioned_record() -> None:
    priv, pub = generate_keypair()
    sig = sign_record(_record(), priv, key_id="k1")
    tampered = HashRecord(hash_id="abc", phash=999, dhash=2, whash=3, campaign_id="camp")
    assert verify_record(tampered, sig, Keyring(keys={"k1": pub})) is False


def test_legacy_signature_verifies_against_legacy_key() -> None:
    # Pre-versioning records (bare signature, no key_id) still verify via the
    # keyring's legacy key — backward-compatible migration.
    priv, pub = generate_keypair()
    legacy_sig = sign_record(_record(), priv)
    assert ":" not in legacy_sig
    assert verify_record(_record(), legacy_sig, Keyring(legacy_public_key_b64=pub)) is True
    # A bare public-key argument is treated as the legacy key too.
    assert verify_record(_record(), legacy_sig, pub) is True


def test_versioned_signature_needs_keyring_not_bare_key() -> None:
    # A single bare public key cannot satisfy a versioned signature (no key_id
    # mapping) — fails closed rather than mis-verifying.
    priv, pub = generate_keypair()
    sig = sign_record(_record(), priv, key_id="k1")
    assert verify_record(_record(), sig, pub) is False


def test_sign_rejects_key_id_with_separator() -> None:
    import pytest

    priv, _ = generate_keypair()
    with pytest.raises(ValueError, match="must not contain"):
        sign_record(_record(), priv, key_id="bad:id")


def test_canonical_bytes_includes_key_id_when_present() -> None:
    with_id = canonical_bytes(_record(), key_id="k1")
    without_id = canonical_bytes(_record())
    assert b'"key_id":"k1"' in with_id
    assert b"key_id" not in without_id
