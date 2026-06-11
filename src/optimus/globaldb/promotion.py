"""Pure promotion eligibility and submitter reputation arithmetic.

Promotion rule (2-of-3 workflow): a candidate is promotable once it has been
approved by at least :data:`MIN_DISTINCT_APPROVERS` distinct moderators who
belong to *different* guilds. Approvals from the same user or from the same
guild do not stack — this prevents a single actor (or a single compromised
guild's mod team) from promoting a hash unilaterally.

Reputation: a submitter gains :data:`CONFIRM_DELTA` when one of their hashes is
confirmed and loses :data:`REJECT_DELTA` when one is rejected; submissions are
gated below :data:`REPUTATION_SUBMIT_THRESHOLD`.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Distinct approvers (from distinct guilds) required to promote a candidate.
MIN_DISTINCT_APPROVERS = 2

#: Reputation change applied when a submitter's hash is confirmed.
CONFIRM_DELTA = 1
#: Reputation change (subtracted) when a submitter's hash is rejected.
REJECT_DELTA = 2
#: Minimum reputation required to submit a candidate to the global database.
REPUTATION_SUBMIT_THRESHOLD = 0


@dataclass(frozen=True, slots=True)
class ApprovalRecord:
    """One moderator approval toward promoting a candidate."""

    approver_user_id: int
    approver_guild_id: int


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    """The outcome of evaluating a candidate's approvals."""

    promotable: bool
    distinct_guilds: int
    distinct_approvers: int


def evaluate_promotion(
    approvals: list[ApprovalRecord], *, min_distinct: int = MIN_DISTINCT_APPROVERS
) -> PromotionDecision:
    """Decide whether a candidate's approvals justify promotion.

    Distinct *guilds* is the gate: an approval only counts if it comes from a
    guild not already represented, and from a user not already counted. This
    enforces "N distinct moderators from N different guilds".
    """
    seen_users: set[int] = set()
    seen_guilds: set[int] = set()
    for approval in approvals:
        if approval.approver_user_id in seen_users:
            continue
        if approval.approver_guild_id in seen_guilds:
            continue
        seen_users.add(approval.approver_user_id)
        seen_guilds.add(approval.approver_guild_id)
    distinct = len(seen_guilds)
    return PromotionDecision(
        promotable=distinct >= min_distinct,
        distinct_guilds=distinct,
        distinct_approvers=len(seen_users),
    )


def reputation_after(current: int, *, confirmed: int = 0, rejected: int = 0) -> int:
    """Return the reputation after ``confirmed`` confirms and ``rejected`` rejects."""
    return current + confirmed * CONFIRM_DELTA - rejected * REJECT_DELTA


def can_submit(reputation: int, *, threshold: int = REPUTATION_SUBMIT_THRESHOLD) -> bool:
    """Whether a submitter at ``reputation`` is allowed to submit a candidate."""
    return reputation >= threshold
