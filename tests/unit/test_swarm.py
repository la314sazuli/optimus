"""Unit tests for cross-guild swarm correlation.

Uses a tiny in-process fake that implements just enough of the Redis sorted-set
semantics the swarm Lua relies on (ZREMRANGEBYSCORE / ZADD / ZCARD), so the
window-trimming and distinct-guild counting are exercised without a real Redis.
"""

from __future__ import annotations

import pytest

from optimus.services.detection.swarm import SwarmCorrelator


class _FakeRedis:
    """Minimal sorted-set store backing the swarm window script."""

    def __init__(self) -> None:
        self._zsets: dict[str, dict[str, float]] = {}

    async def eval(self, _script: str, _nkeys: int, key: str, *args: object) -> int:
        now = float(args[0])  # type: ignore[arg-type]
        window = float(args[1])  # type: ignore[arg-type]
        guild = str(args[2])
        members = self._zsets.setdefault(key, {})
        cutoff = now - window
        for member in [m for m, score in members.items() if score < cutoff]:
            del members[member]
        members[guild] = now
        return len(members)


async def test_below_threshold_not_swarming() -> None:
    correlator = SwarmCorrelator(_FakeRedis(), min_guilds=3, window_seconds=300)
    obs1 = await correlator.observe(phash=0xABC, guild_id=1)
    obs2 = await correlator.observe(phash=0xABC, guild_id=2)
    assert obs1.distinct_guilds == 1
    assert not obs1.is_swarming
    assert obs2.distinct_guilds == 2
    assert not obs2.is_swarming


async def test_reaches_threshold_across_distinct_guilds() -> None:
    correlator = SwarmCorrelator(_FakeRedis(), min_guilds=3, window_seconds=300)
    await correlator.observe(phash=0xABC, guild_id=1)
    await correlator.observe(phash=0xABC, guild_id=2)
    obs = await correlator.observe(phash=0xABC, guild_id=3)
    assert obs.distinct_guilds == 3
    assert obs.is_swarming


async def test_same_guild_repeats_do_not_swarm() -> None:
    correlator = SwarmCorrelator(_FakeRedis(), min_guilds=3, window_seconds=300)
    obs = None
    for _ in range(5):
        obs = await correlator.observe(phash=0xABC, guild_id=1)
    assert obs is not None
    assert obs.distinct_guilds == 1
    assert not obs.is_swarming


async def test_distinct_phashes_tracked_independently() -> None:
    correlator = SwarmCorrelator(_FakeRedis(), min_guilds=2, window_seconds=300)
    a = await correlator.observe(phash=0x1, guild_id=1)
    b = await correlator.observe(phash=0x2, guild_id=2)
    assert a.distinct_guilds == 1
    assert b.distinct_guilds == 1


def test_min_guilds_must_be_positive() -> None:
    with pytest.raises(ValueError, match="min_guilds"):
        SwarmCorrelator(_FakeRedis(), min_guilds=0)


def test_window_seconds_property() -> None:
    correlator = SwarmCorrelator(_FakeRedis(), window_seconds=120)
    assert correlator.window_seconds == 120


class _ExplodingRedis:
    """``eval`` always raises, simulating a Redis outage during observation."""

    def __init__(self) -> None:
        self.calls = 0

    async def eval(self, *_args: object) -> int:
        self.calls += 1
        raise ConnectionError("redis down")


async def test_observe_fails_safe_on_redis_error() -> None:
    from optimus.services.detection.swarm import REDIS_SWARM_FALLBACK

    redis = _ExplodingRedis()
    correlator = SwarmCorrelator(redis, min_guilds=3, window_seconds=300)
    before = REDIS_SWARM_FALLBACK._value.get()

    obs = await correlator.observe(phash=0xABC, guild_id=1)

    # Degrades to non-swarming rather than propagating the Redis error, so the
    # caller (detection worker) keeps the per-image verdict and never naks.
    assert obs.distinct_guilds == 0
    assert not obs.is_swarming
    assert redis.calls == 1
    assert REDIS_SWARM_FALLBACK._value.get() == before + 1
