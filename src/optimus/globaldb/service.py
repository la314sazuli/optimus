"""Service layer for the signed, federated global hash database.

Composes the pure promotion/reputation arithmetic (:mod:`.promotion`) and
Ed25519 signing (:mod:`.signing`) with the database repositories and a Redis
rate limiter to implement the full candidate lifecycle:

* **submit** — gated by submitter reputation and a per-user Redis rate limit;
* **approve** — idempotent per moderator; once approvals come from
  :data:`MIN_DISTINCT_APPROVERS` distinct moderators in *different* guilds the
  candidate is **promoted** and signed with the authority's private key;
* **revoke** — flips a promoted/candidate hash to ``revoked``;
* **verify** — consumers check each promoted record's signature against the
  configured public key and reject anything unsigned or invalid.

The private signing key is read only here, only on the signing-authority
deployment, and only from configuration sourced from the environment — it is
never persisted, logged, or shipped to workers (which hold the public key only).
"""

from __future__ import annotations

from dataclasses import dataclass

from optimus.core.ratelimit import RateLimit, RateLimiter
from optimus.db.models import GlobalHash
from optimus.db.repositories import GlobalHashRepository, GlobalSubmitterRepository
from optimus.globaldb.promotion import (
    ApprovalRecord,
    can_submit,
    evaluate_promotion,
)
from optimus.globaldb.signing import HashRecord, sign_record, verify_record

#: Default per-user submission budget: 5 candidates, refilling at 1 per minute.
SUBMIT_RATE = RateLimit(capacity=5.0, refill_rate=1.0 / 60.0)


class SubmissionDenied(Exception):  # noqa: N818 - control-flow signal, not an error
    """Raised when a submission is refused by reputation gate or rate limit."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class PromotionResult:
    """The outcome of recording an approval."""

    promoted: bool
    distinct_guilds: int
    distinct_approvers: int


class GlobalHashService:
    """Coordinates submission, approval/promotion, signing, and revocation."""

    def __init__(
        self,
        hashes: GlobalHashRepository,
        submitters: GlobalSubmitterRepository,
        rate_limiter: RateLimiter,
        *,
        signing_private_key_b64: str = "",
        signing_public_key_b64: str = "",
        submit_rate: RateLimit = SUBMIT_RATE,
    ) -> None:
        self._hashes = hashes
        self._submitters = submitters
        self._rl = rate_limiter
        self._private_key = signing_private_key_b64
        self._public_key = signing_public_key_b64
        self._submit_rate = submit_rate

    async def submit(
        self,
        *,
        hash_id: str,
        phash: int,
        dhash: int,
        whash: int,
        submitter_user_id: int,
        submitter_guild_id: int,
    ) -> GlobalHash:
        """Submit a candidate after the reputation gate and per-user rate limit.

        Raises :class:`SubmissionDenied` (``below_threshold`` / ``rate_limited``)
        if the submitter is under-reputation or over budget.
        """
        submitter = await self._submitters.get_or_create(submitter_user_id)
        if not can_submit(submitter.reputation):
            raise SubmissionDenied("below_threshold")
        if not await self._rl.acquire(f"globalsubmit:{submitter_user_id}", self._submit_rate):
            raise SubmissionDenied("rate_limited")

        row = await self._hashes.submit_candidate(
            hash_id=hash_id,
            phash=phash,
            dhash=dhash,
            whash=whash,
            submitter_user_id=submitter_user_id,
            submitter_guild_id=submitter_guild_id,
        )
        await self._submitters.record_submission(submitter_user_id)
        return row

    async def approve(
        self, *, hash_id: str, approver_user_id: int, approver_guild_id: int
    ) -> PromotionResult:
        """Record a moderator approval; promote + sign when the gate is met.

        Promotion requires approvals from at least :data:`MIN_DISTINCT_APPROVERS`
        *distinct* moderators in *distinct* guilds (same user or same guild does
        not stack). On promotion the submitter's reputation is credited.
        """
        approvals = await self._hashes.add_approval(
            hash_id=hash_id,
            approver_user_id=approver_user_id,
            approver_guild_id=approver_guild_id,
        )
        decision = evaluate_promotion(
            [ApprovalRecord(a.approver_user_id, a.approver_guild_id) for a in approvals]
        )
        row = await self._hashes.get(hash_id)
        if row is None:
            raise KeyError(hash_id)

        if decision.promotable and row.status != "promoted":
            signature = self._sign(row)
            await self._hashes.promote(hash_id, signature=signature)
            if row.submitter_user_id is not None:
                await self._submitters.adjust_reputation(row.submitter_user_id, confirmed=1)
            return PromotionResult(
                promoted=True,
                distinct_guilds=decision.distinct_guilds,
                distinct_approvers=decision.distinct_approvers,
            )
        return PromotionResult(
            promoted=row.status == "promoted",
            distinct_guilds=decision.distinct_guilds,
            distinct_approvers=decision.distinct_approvers,
        )

    async def revoke(self, hash_id: str) -> None:
        """Revoke a hash and dock the submitter's reputation."""
        row = await self._hashes.get(hash_id)
        if row is None:
            raise KeyError(hash_id)
        await self._hashes.revoke(hash_id)
        if row.submitter_user_id is not None:
            await self._submitters.adjust_reputation(row.submitter_user_id, rejected=1)

    async def verified_promoted(self) -> list[GlobalHash]:
        """Return promoted hashes whose signature verifies under the public key.

        Consumers call this on pull/load; any record with a missing or invalid
        signature is dropped so an unsigned or tampered row is never trusted.
        """
        rows = await self._hashes.list_promoted()
        return [row for row in rows if self._verify(row)]

    def _sign(self, row: GlobalHash) -> str:
        return sign_record(self._record(row), self._private_key)

    def _verify(self, row: GlobalHash) -> bool:
        return verify_record(self._record(row), row.signature, self._public_key)

    @staticmethod
    def _record(row: GlobalHash) -> HashRecord:
        return HashRecord(
            hash_id=row.hash_id,
            phash=row.phash,
            dhash=row.dhash,
            whash=row.whash,
            status="promoted",
            campaign_id=row.campaign_id,
        )
