"""2-of-3 promotion gate and submitter reputation arithmetic."""

from __future__ import annotations

from optimus.globaldb.promotion import (
    CONFIRM_DELTA,
    REJECT_DELTA,
    ApprovalRecord,
    can_submit,
    evaluate_promotion,
    reputation_after,
)


def test_default_requires_three_distinct_guilds() -> None:
    # The default anti-sybil floor is three: two guilds are not enough.
    two = [ApprovalRecord(1, 100), ApprovalRecord(2, 200)]
    assert evaluate_promotion(two).promotable is False
    three = [*two, ApprovalRecord(3, 300)]
    assert evaluate_promotion(three).promotable is True


def test_two_distinct_guilds_promote_at_custom_threshold() -> None:
    approvals = [ApprovalRecord(1, 100), ApprovalRecord(2, 200)]
    decision = evaluate_promotion(approvals, min_distinct=2)
    assert decision.promotable is True
    assert decision.distinct_guilds == 2


def test_untrusted_guilds_do_not_corroborate() -> None:
    # Only allowlisted guilds count; sybil guilds outside it are ignored even
    # when they outnumber the threshold.
    approvals = [ApprovalRecord(1, 900), ApprovalRecord(2, 901), ApprovalRecord(3, 902)]
    decision = evaluate_promotion(approvals, min_distinct=2, trusted_guild_ids={100, 200})
    assert decision.promotable is False
    assert decision.distinct_guilds == 0


def test_trusted_guilds_corroborate() -> None:
    approvals = [
        ApprovalRecord(1, 900),  # untrusted, ignored
        ApprovalRecord(2, 100),
        ApprovalRecord(3, 200),
    ]
    decision = evaluate_promotion(approvals, min_distinct=2, trusted_guild_ids={100, 200})
    assert decision.promotable is True
    assert decision.distinct_guilds == 2


def test_same_guild_does_not_stack() -> None:
    # Two different moderators, but both in the same guild.
    approvals = [ApprovalRecord(1, 100), ApprovalRecord(2, 100)]
    decision = evaluate_promotion(approvals)
    assert decision.promotable is False
    assert decision.distinct_guilds == 1


def test_same_user_does_not_stack() -> None:
    approvals = [ApprovalRecord(1, 100), ApprovalRecord(1, 200)]
    decision = evaluate_promotion(approvals)
    assert decision.promotable is False
    assert decision.distinct_approvers == 1


def test_single_approval_not_promotable() -> None:
    assert evaluate_promotion([ApprovalRecord(1, 100)]).promotable is False


def test_three_distinct_guilds_promote() -> None:
    approvals = [ApprovalRecord(1, 100), ApprovalRecord(2, 200), ApprovalRecord(3, 300)]
    decision = evaluate_promotion(approvals)
    assert decision.promotable is True
    assert decision.distinct_guilds == 3


def test_custom_min_distinct() -> None:
    approvals = [ApprovalRecord(1, 100), ApprovalRecord(2, 200)]
    assert evaluate_promotion(approvals, min_distinct=3).promotable is False


def test_reputation_after_confirm_and_reject() -> None:
    assert reputation_after(0, confirmed=1) == CONFIRM_DELTA
    assert reputation_after(0, rejected=1) == -REJECT_DELTA
    assert reputation_after(5, confirmed=2, rejected=1) == 5 + 2 * CONFIRM_DELTA - REJECT_DELTA


def test_can_submit_threshold() -> None:
    assert can_submit(0) is True
    assert can_submit(-1) is False
    assert can_submit(-1, threshold=-5) is True
