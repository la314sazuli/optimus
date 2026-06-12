"""Integration tests for the detection index manager (build + invalidation)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession

from optimus.db.models import GlobalHash, Guild, GuildHash
from optimus.services.detection.index import IndexManager

GUILD_ID = 100


def _scope_factory(session: AsyncSession):  # type: ignore[no-untyped-def]
    @asynccontextmanager
    async def scope() -> AsyncIterator[AsyncSession]:
        yield session

    return scope


async def _add_guild_hash(session: AsyncSession, hash_id: str, phash: int) -> None:
    session.add(
        GuildHash(
            guild_id=GUILD_ID,
            hash_id=hash_id,
            phash=phash,
            dhash=0,
            whash=0,
            ahash=0,
            source="local",
            status="active",
        )
    )
    await session.flush()


async def test_guild_index_builds_and_caches(session: AsyncSession) -> None:
    session.add(Guild(guild_id=GUILD_ID))
    await _add_guild_hash(session, "h1", 1234)
    await session.commit()

    mgr = IndexManager(_scope_factory(session))
    idx = await mgr.guild_index(GUILD_ID)
    assert len(idx) == 1
    # Cached: returns the same instance until invalidated.
    assert await mgr.guild_index(GUILD_ID) is idx


async def test_guild_index_invalidation_reloads(session: AsyncSession) -> None:
    session.add(Guild(guild_id=GUILD_ID))
    await _add_guild_hash(session, "h1", 1234)
    await session.commit()

    mgr = IndexManager(_scope_factory(session))
    idx = await mgr.guild_index(GUILD_ID)
    assert len(idx) == 1

    await _add_guild_hash(session, "h2", 5678)
    await session.commit()

    # Stale cache still reports the old size...
    assert len(await mgr.guild_index(GUILD_ID)) == 1
    # ...until invalidation rebuilds it.
    await mgr.invalidate(GUILD_ID)
    reloaded = await mgr.guild_index(GUILD_ID)
    assert len(reloaded) == 2
    assert reloaded is not idx


async def test_guild_index_lru_evicts_least_recently_used(session: AsyncSession) -> None:
    session.add(Guild(guild_id=GUILD_ID))
    await _add_guild_hash(session, "h1", 1234)
    await session.commit()

    mgr = IndexManager(_scope_factory(session), max_guilds=2)
    # Build for the populated guild plus two empty guilds (repo filters by id).
    idx_main = await mgr.guild_index(GUILD_ID)
    await mgr.guild_index(201)
    # Touch the populated guild so it becomes most-recent; 201 is now LRU.
    assert await mgr.guild_index(GUILD_ID) is idx_main
    # A third distinct guild exceeds the cap and evicts the LRU (201).
    await mgr.guild_index(202)

    assert mgr.cached_guilds() == [GUILD_ID, 202]
    # The evicted guild rebuilds on demand (a fresh, non-identical instance).
    rebuilt = await mgr.guild_index(201)
    assert rebuilt is not None
    # Rebuilding 201 now evicts the next LRU (GUILD_ID).
    assert mgr.cached_guilds() == [202, 201]


async def test_guild_index_invalidate_respects_lru_cap(session: AsyncSession) -> None:
    session.add(Guild(guild_id=GUILD_ID))
    await session.commit()

    mgr = IndexManager(_scope_factory(session), max_guilds=1)
    await mgr.guild_index(GUILD_ID)
    # Invalidating a different guild rebuilds it and evicts over-cap entries.
    await mgr.invalidate(301)
    assert mgr.cached_guilds() == [301]


async def test_guild_index_unbounded_by_default(session: AsyncSession) -> None:
    session.add(Guild(guild_id=GUILD_ID))
    await session.commit()

    mgr = IndexManager(_scope_factory(session))
    for gid in range(400, 410):
        await mgr.guild_index(gid)
    assert len(mgr.cached_guilds()) == 10


async def test_guild_index_builds_mirror_from_stored_columns(session: AsyncSession) -> None:
    session.add(Guild(guild_id=GUILD_ID))
    # A row whose mirror (flip) hashes were stored at indexing time.
    session.add(
        GuildHash(
            guild_id=GUILD_ID,
            hash_id="m1",
            phash=1000,
            dhash=2000,
            whash=3000,
            ahash=4000,
            mphash=1111,
            mdhash=2222,
            mwhash=3333,
            mahash=4444,
            source="local",
            status="active",
        )
    )
    await session.commit()

    mgr = IndexManager(_scope_factory(session))
    idx = await mgr.guild_index(GUILD_ID)
    # __len__ counts distinct sources, not the internal mirror sibling.
    assert len(idx) == 1
    # The original phash and the stored mirror phash both resolve to source "m1".
    assert [k.hash_id for k in idx.candidates(1000, 0)] == ["m1"]
    mirror_hits = idx.candidates(1111, 0)
    assert [k.hash_id for k in mirror_hits] == ["m1"]
    # The mirror candidate carries the flipped hashes for scoring.
    assert mirror_hits[0].as_dict() == {"phash": 1111, "dhash": 2222, "whash": 3333, "ahash": 4444}


async def test_guild_index_skips_mirror_when_columns_null(session: AsyncSession) -> None:
    session.add(Guild(guild_id=GUILD_ID))
    await _add_guild_hash(session, "h1", 1234)  # no mirror columns
    await session.commit()

    mgr = IndexManager(_scope_factory(session))
    idx = await mgr.guild_index(GUILD_ID)
    assert len(idx) == 1
    # No mirror sibling indexed: only the original phash matches.
    assert [k.hash_id for k in idx.candidates(1234, 0)] == ["h1"]


async def test_global_index_invalidation(session: AsyncSession) -> None:
    session.add(GlobalHash(hash_id="g1", phash=42, dhash=0, whash=0, status="promoted"))
    session.add(GlobalHash(hash_id="g-cand", phash=43, dhash=0, whash=0, status="candidate"))
    await session.commit()

    mgr = IndexManager(_scope_factory(session))
    gx = await mgr.global_index()
    # Only promoted hashes are indexed.
    assert len(gx) == 1

    session.add(GlobalHash(hash_id="g2", phash=44, dhash=0, whash=0, status="promoted"))
    await session.commit()
    await mgr.invalidate(None)
    assert len(await mgr.global_index()) == 2
