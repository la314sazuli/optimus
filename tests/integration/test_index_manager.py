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
