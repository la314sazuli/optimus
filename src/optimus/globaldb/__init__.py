"""Signed global hash database: submission, promotion, signing, verification.

A submitted hash starts as a ``candidate``. Promotion to ``promoted`` requires
approvals from at least two distinct verified moderators from *different*
guilds; on promotion the canonical hash record is signed with an Ed25519 key
held only by the signing-authority deployment. Consumers verify the signature
on load and reject unsigned, invalid, or revoked records. Submitters carry a
reputation score that gates whether they may submit at all.

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
    "PromotionDecision",
    "can_submit",
    "canonical_bytes",
    "evaluate_promotion",
    "generate_keypair",
    "reputation_after",
    "sign_record",
    "verify_record",
]
