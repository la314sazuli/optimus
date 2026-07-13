"""Tests for hot-path guild config: DB load with related rows and Redis caching."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import fakeredis.aioredis
from sqlalchemy.ext.asyncio import AsyncSession

from optimus.core.config import Sensitivity
from optimus.core.guild_config import GuildConfig, GuildConfigCache, load_from_db
from optimus.db.models import (
    Guild,
    GuildChannelIgnored,
    GuildRoleIgnored,
    GuildTrustedUser,
)


async def test_load_from_db_unconfigured_guild_returns_default(session: AsyncSession) -> None:
    config = await load_from_db(session, 12345)
    assert config == GuildConfig.default(12345)
    assert config.sensitivity is Sensitivity.BALANCED
    assert config.ignored_channels == frozenset()


async def test_load_from_db_hydrates_related_rows(session: AsyncSession) -> None:
    session.add(
        Guild(
            guild_id=1,
            sensitivity="strict",
            optin_scan_bots=True,
            safe_mode=True,
        )
    )
    session.add_all(
        [
            GuildChannelIgnored(guild_id=1, channel_id=100),
            GuildChannelIgnored(guild_id=1, channel_id=101),
            GuildRoleIgnored(guild_id=1, role_id=200),
            GuildTrustedUser(guild_id=1, user_id=300),
        ]
    )
    await session.flush()

    config = await load_from_db(session, 1)
    assert config.sensitivity is Sensitivity.STRICT
    assert config.scan_bots is True
    assert config.safe_mode is True
    assert config.ignored_channels == frozenset({100, 101})
    assert config.ignored_roles == frozenset({200})
    assert config.trusted_users == frozenset({300})


def test_guild_config_json_roundtrip() -> None:
    original = GuildConfig(
        guild_id=7,
        sensitivity=Sensitivity.PERMISSIVE,
        scan_bots=True,
        safe_mode=False,
        ignored_channels=frozenset({1, 2}),
        ignored_roles=frozenset({3}),
        trusted_users=frozenset({4, 5}),
    )
    assert GuildConfig.from_json(original.to_json()) == original


@asynccontextmanager
async def _loader(session: AsyncSession) -> AsyncIterator[AsyncSession]:
    yield session


async def test_cache_populates_on_miss_and_serves_on_hit(session: AsyncSession) -> None:
    session.add(Guild(guild_id=5, sensitivity="strict", optin_scan_bots=True))
    await session.flush()

    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    cache = GuildConfigCache(redis, lambda guild_id=None: _loader(session), ttl_seconds=120)

    # Miss: loads from DB and writes through to Redis with the TTL.
    first = await cache.get(5)
    assert first.sensitivity is Sensitivity.STRICT
    assert await redis.get("optimus:guildcfg:5") is not None
    assert 0 < await redis.ttl("optimus:guildcfg:5") <= 120

    # Hit: served from Redis even after the underlying row changes.
    guild = await session.get(Guild, 5)
    assert guild is not None
    guild.sensitivity = "permissive"
    await session.flush()
    second = await cache.get(5)
    assert second.sensitivity is Sensitivity.STRICT  # stale cache wins until invalidated
    await redis.aclose()


async def test_cache_invalidate_forces_reload(session: AsyncSession) -> None:
    session.add(Guild(guild_id=8, sensitivity="strict"))
    await session.flush()

    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    cache = GuildConfigCache(redis, lambda guild_id=None: _loader(session))

    await cache.get(8)
    guild = await session.get(Guild, 8)
    assert guild is not None
    guild.sensitivity = "permissive"
    await session.flush()

    await cache.invalidate(8)
    assert await redis.get("optimus:guildcfg:8") is None
    reloaded = await cache.get(8)
    assert reloaded.sensitivity is Sensitivity.PERMISSIVE
    await redis.aclose()


async def test_cache_without_redis_always_hits_db(session: AsyncSession) -> None:
    session.add(Guild(guild_id=9, sensitivity="permissive"))
    await session.flush()
    cache = GuildConfigCache(None, lambda guild_id=None: _loader(session))
    config = await cache.get(9)
    assert config.sensitivity is Sensitivity.PERMISSIVE
    # invalidate is a no-op when there is no cache backend.
    await cache.invalidate(9)
