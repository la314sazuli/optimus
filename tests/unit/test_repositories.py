"""Repository tests on aiosqlite, focusing on guild_id scoping."""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from optimus.db.models import (
    Appeal,
    Detection,
    Guild,
    GuildHash,
    GuildTrustedUser,
    GuildWhitelist,
    ModAction,
)
from optimus.db.repositories import (
    DetectionRepository,
    GuildHashRepository,
    GuildPurgeRepository,
    GuildRepository,
    UserOptoutRepository,
    WhitelistRepository,
)


async def _make_guild(session: AsyncSession, guild_id: int) -> None:
    repo = GuildRepository(session)
    await repo.upsert(Guild(guild_id=guild_id))


async def test_guild_upsert_and_safe_mode(session: AsyncSession) -> None:
    repo = GuildRepository(session)
    await repo.upsert(Guild(guild_id=1, sensitivity="strict"))
    guild = await repo.get(1)
    assert guild is not None
    assert guild.sensitivity == "strict"
    assert guild.safe_mode is False
    await repo.set_safe_mode(1, True)
    refreshed = await repo.get(1)
    assert refreshed is not None and refreshed.safe_mode is True


async def test_set_safe_mode_unknown_guild_raises(session: AsyncSession) -> None:
    repo = GuildRepository(session)
    with pytest.raises(KeyError):
        await repo.set_safe_mode(999, True)


async def test_guild_hash_repo_scopes_to_guild(session: AsyncSession) -> None:
    await _make_guild(session, 1)
    await _make_guild(session, 2)

    repo1 = GuildHashRepository(session, guild_id=1)
    repo2 = GuildHashRepository(session, guild_id=2)

    await repo1.add(GuildHash(hash_id="h1", phash=1, dhash=2, whash=3, guild_id=999))
    await repo2.add(GuildHash(hash_id="h2", phash=4, dhash=5, whash=6))

    # add() forces the repository's guild scope regardless of the passed value.
    stored = await repo1.get("h1")
    assert stored is not None and stored.guild_id == 1

    # Cross-guild reads return nothing.
    assert await repo1.get("h2") is None
    assert await repo2.get("h1") is None

    active1 = await repo1.list_active()
    assert [h.hash_id for h in active1] == ["h1"]


async def test_guild_hash_remove_is_scoped(session: AsyncSession) -> None:
    await _make_guild(session, 1)
    await _make_guild(session, 2)
    repo1 = GuildHashRepository(session, guild_id=1)
    repo2 = GuildHashRepository(session, guild_id=2)
    await repo1.add(GuildHash(hash_id="x", phash=1, dhash=1, whash=1))

    # Removing from the wrong guild scope deletes nothing.
    assert await repo2.remove("x") == 0
    assert await repo1.remove("x") == 1
    assert await repo1.get("x") is None


async def test_whitelist_scoped(session: AsyncSession) -> None:
    await _make_guild(session, 7)
    repo = WhitelistRepository(session, guild_id=7)
    await repo.add(GuildWhitelist(phash=1, dhash=2, whash=3, reason="legit", guild_id=999))
    entries = await repo.list()
    assert len(entries) == 1
    assert entries[0].guild_id == 7


async def test_detection_idempotency_scoped(session: AsyncSession) -> None:
    await _make_guild(session, 5)
    repo = DetectionRepository(session, guild_id=5)
    await repo.record(
        Detection(
            message_id=10,
            channel_id=11,
            attachment_id=12,
            uploader_id=13,
            distances={"phash": 0},
            verdict="scam",
            idempotency_key="10:12",
        )
    )
    found = await repo.get_by_idempotency_key("10:12")
    assert found is not None and found.guild_id == 5

    other = DetectionRepository(session, guild_id=6)
    assert await other.get_by_idempotency_key("10:12") is None

    recent = await repo.list_recent()
    assert len(recent) == 1


# --- opt-out / right to erasure ------------------------------------------------


async def test_optout_is_idempotent(session: AsyncSession) -> None:
    repo = UserOptoutRepository(session)
    assert await repo.is_opted_out(42) is False
    await repo.opt_out(42)
    await repo.opt_out(42)  # second call must not raise or duplicate
    assert await repo.is_opted_out(42) is True


async def test_purge_user_removes_rows_but_keeps_tombstone(session: AsyncSession) -> None:
    await _make_guild(session, 1)
    det_repo = DetectionRepository(session, guild_id=1)
    await det_repo.record(
        Detection(
            message_id=1,
            channel_id=2,
            attachment_id=3,
            uploader_id=99,
            distances={},
            verdict="scam",
            idempotency_key="k1",
        )
    )
    session.add(GuildTrustedUser(guild_id=1, user_id=99))
    await session.flush()

    repo = UserOptoutRepository(session)
    await repo.opt_out(99)
    deleted = await repo.purge_user(99)

    assert deleted >= 2
    assert await det_repo.get_by_idempotency_key("k1") is None
    # The opt-out tombstone survives so the user stays excluded.
    assert await repo.is_opted_out(99) is True


# --- full guild erasure (/delete_server_data) ----------------------------------


async def test_guild_purge_removes_every_owned_row(session: AsyncSession) -> None:
    await _make_guild(session, 1)
    await _make_guild(session, 2)
    det_repo = DetectionRepository(session, guild_id=1)
    await det_repo.record(
        Detection(
            message_id=1,
            channel_id=2,
            attachment_id=3,
            uploader_id=5,
            distances={},
            verdict="scam",
            idempotency_key="g1",
        )
    )
    detection = await det_repo.get_by_idempotency_key("g1")
    assert detection is not None
    session.add(Appeal(guild_id=1, detection_id=detection.id, user_id=5))
    session.add(ModAction(guild_id=1, actor_id=7, action="ban"))
    session.add(GuildHash(hash_id="h", phash=1, dhash=1, whash=1, guild_id=1))
    # A second guild's data must be untouched.
    other = DetectionRepository(session, guild_id=2)
    await other.record(
        Detection(
            message_id=9,
            channel_id=9,
            attachment_id=9,
            uploader_id=9,
            distances={},
            verdict="scam",
            idempotency_key="keep",
        )
    )
    await session.flush()

    deleted = await GuildPurgeRepository(session, guild_id=1).purge()

    assert deleted >= 4
    assert await GuildRepository(session).get(1) is None
    assert await det_repo.get_by_idempotency_key("g1") is None
    # Guild 2 is left intact.
    assert await GuildRepository(session).get(2) is not None
    assert await other.get_by_idempotency_key("keep") is not None
