"""Signed global hash database: submission, promotion, signing, verification.

A submitted hash starts as a ``candidate``. Promotion to ``promoted`` requires
approvals from a configurable number of distinct verified moderators in
*different* guilds (optionally restricted to a trusted-guild allowlist so sybil
guilds cannot manufacture consensus); on promotion the canonical hash record is
signed with an Ed25519 key held only by the signing-authority deployment. Each
signature carries a ``key_id`` so consumers can verify against a rotating set of
valid keys and reject revoked ones. Consumers verify the signature on load and
reject unsigned, invalid, or revoked records. Submitters carry a reputation
score that gates whether they may submit at all.

The pure logic (canonicalization, signing, verification, promotion eligibility,
reputation arithmetic) lives here so it is fully unit-testable; the database and
Redis side effects live in :mod:`optimus.globaldb.service`.
"""

from __future__ import annotations

from optimus.globaldb.promotion import (
    CONFIRM_DELTA,
    REJECT_DELTA,
    REPUTATION_SUBMIT_THRESHOLD,
    ApprovalRecord,
    PromotionDecision,
    can_submit,
    evaluate_promotion,
    reputation_after,
)
from optimus.globaldb.signing import (
    HashRecord,
    Keyring,
    canonical_bytes,
    generate_keypair,
    sign_record,
    verify_record,
)

__all__ = [
    "CONFIRM_DELTA",
    "REJECT_DELTA",
    "REPUTATION_SUBMIT_THRESHOLD",
    "ApprovalRecord",
    "HashRecord",
    "Keyring",
    "PromotionDecision",
    "can_submit",
    "canonical_bytes",
    "evaluate_promotion",
    "generate_keypair",
    "reputation_after",
    "sign_record",
    "verify_record",
]
