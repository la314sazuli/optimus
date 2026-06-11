"""Global hash DB service: submission gating, 2-of-3 promotion, signing, revoke."""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from optimus.core.ratelimit import InMemoryRateLimiter, RateLimit
from optimus.db.repositories import GlobalHashRepository, GlobalSubmitterRepository
from optimus.globaldb.promotion import REJECT_DELTA
from optimus.globaldb.service import GlobalHashService, SubmissionDenied
from optimus.globaldb.signing import HashRecord, generate_keypair, verify_record


def _service(session: AsyncSession, *, priv: str = "", pub: str = "") -> GlobalHashService:
    return GlobalHashService(
        GlobalHashRepository(session),
        GlobalSubmitterRepository(session),
        InMemoryRateLimiter(),
        signing_private_key_b64=priv,
        signing_public_key_b64=pub,
    )


async def _submit(svc: GlobalHashService, hash_id: str = "h1", *, user: int = 1) -> None:
    await svc.submit(
        hash_id=hash_id,
        phash=10,
        dhash=20,
        whash=30,
        submitter_user_id=user,
        submitter_guild_id=900,
    )


@pytest.mark.asyncio
async def test_submit_creates_candidate(session: AsyncSession) -> None:
    svc = _service(session)
    await _submit(svc)
    row = await GlobalHashRepository(session).get("h1")
    assert row is not None
    assert row.status == "candidate"
    assert row.submitter_user_id == 1


@pytest.mark.asyncio
async def test_submit_records_submission_count(session: AsyncSession) -> None:
    svc = _service(session)
    await _submit(svc)
    submitter = await GlobalSubmitterRepository(session).get_or_create(1)
    assert submitter.submitted == 1


@pytest.mark.asyncio
async def test_submit_below_threshold_denied(session: AsyncSession) -> None:
    # Drive the submitter's reputation below zero, then attempt to submit.
    await GlobalSubmitterRepository(session).adjust_reputation(7, rejected=1)
    svc = _service(session)
    with pytest.raises(SubmissionDenied) as exc:
        await _submit(svc, user=7)
    assert exc.value.reason == "below_threshold"


@pytest.mark.asyncio
async def test_submit_rate_limited(session: AsyncSession) -> None:
    svc = GlobalHashService(
        GlobalHashRepository(session),
        GlobalSubmitterRepository(session),
        InMemoryRateLimiter(),
        submit_rate=RateLimit(capacity=1.0, refill_rate=1e-9),
    )
    await _submit(svc, "a", user=5)
    with pytest.raises(SubmissionDenied) as exc:
        await _submit(svc, "b", user=5)
    assert exc.value.reason == "rate_limited"


@pytest.mark.asyncio
async def test_two_distinct_guild_approvals_promote_and_sign(session: AsyncSession) -> None:
    priv, pub = generate_keypair()
    svc = _service(session, priv=priv, pub=pub)
    await _submit(svc, "h1", user=1)

    first = await svc.approve(hash_id="h1", approver_user_id=11, approver_guild_id=100)
    assert first.promoted is False
    second = await svc.approve(hash_id="h1", approver_user_id=22, approver_guild_id=200)
    assert second.promoted is True

    row = await GlobalHashRepository(session).get("h1")
    assert row is not None
    assert row.status == "promoted"
    assert row.signature is not None
    record = HashRecord(hash_id=row.hash_id, phash=row.phash, dhash=row.dhash, whash=row.whash)
    assert verify_record(record, row.signature, pub) is True


@pytest.mark.asyncio
async def test_same_guild_approvals_do_not_promote(session: AsyncSession) -> None:
    priv, pub = generate_keypair()
    svc = _service(session, priv=priv, pub=pub)
    await _submit(svc, "h1")
    await svc.approve(hash_id="h1", approver_user_id=11, approver_guild_id=100)
    result = await svc.approve(hash_id="h1", approver_user_id=22, approver_guild_id=100)
    assert result.promoted is False
    row = await GlobalHashRepository(session).get("h1")
    assert row is not None
    assert row.status == "candidate"


@pytest.mark.asyncio
async def test_promotion_credits_submitter_reputation(session: AsyncSession) -> None:
    priv, pub = generate_keypair()
    svc = _service(session, priv=priv, pub=pub)
    await _submit(svc, "h1", user=42)
    await svc.approve(hash_id="h1", approver_user_id=11, approver_guild_id=100)
    await svc.approve(hash_id="h1", approver_user_id=22, approver_guild_id=200)
    submitter = await GlobalSubmitterRepository(session).get_or_create(42)
    assert submitter.confirmed == 1
    assert submitter.reputation == 1


@pytest.mark.asyncio
async def test_duplicate_approval_is_idempotent(session: AsyncSession) -> None:
    priv, pub = generate_keypair()
    svc = _service(session, priv=priv, pub=pub)
    await _submit(svc, "h1")
    await svc.approve(hash_id="h1", approver_user_id=11, approver_guild_id=100)
    result = await svc.approve(hash_id="h1", approver_user_id=11, approver_guild_id=100)
    assert result.promoted is False
    assert result.distinct_guilds == 1


@pytest.mark.asyncio
async def test_revoke_marks_revoked_and_docks_reputation(session: AsyncSession) -> None:
    svc = _service(session)
    await _submit(svc, "h1", user=42)
    await svc.revoke("h1")
    row = await GlobalHashRepository(session).get("h1")
    assert row is not None
    assert row.status == "revoked"
    submitter = await GlobalSubmitterRepository(session).get_or_create(42)
    assert submitter.rejected == 1
    assert submitter.reputation == -REJECT_DELTA


@pytest.mark.asyncio
async def test_verified_promoted_drops_unsigned_and_invalid(session: AsyncSession) -> None:
    priv, pub = generate_keypair()
    svc = _service(session, priv=priv, pub=pub)
    repo = GlobalHashRepository(session)

    # A correctly promoted+signed hash.
    await _submit(svc, "good")
    await svc.approve(hash_id="good", approver_user_id=11, approver_guild_id=100)
    await svc.approve(hash_id="good", approver_user_id=22, approver_guild_id=200)

    # A promoted hash with a bogus signature (e.g. tampered in transit).
    await _submit(svc, "bad", user=2)
    await repo.promote("bad", signature="not-a-valid-signature")

    verified = await svc.verified_promoted()
    ids = {row.hash_id for row in verified}
    assert "good" in ids
    assert "bad" not in ids


@pytest.mark.asyncio
async def test_revoke_unknown_raises(session: AsyncSession) -> None:
    svc = _service(session)
    with pytest.raises(KeyError):
        await svc.revoke("missing")
