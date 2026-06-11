"""Cross-guild swarm correlation via a Redis sorted-set sliding window.

When the same scam image (keyed by phash) is seen in at least ``min_guilds``
distinct guilds inside ``window_seconds``, the campaign is "swarming": the
verdict's confidence is escalated one band and a ``swarm_alert`` is emitted. The
window is a Redis sorted set of ``guild_id`` members scored by timestamp; stale
members are trimmed on each observation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

_SWARM_PREFIX = "optimus:swarm"

# Atomic observe-and-count: trim the window, add this guild, count distinct
# guilds remaining, and refresh the key TTL. Returns the distinct-guild count.
# KEYS[1]=window key. ARGV: now, window_seconds, guild_id.
_SWARM_SCRIPT = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local guild = ARGV[3]
redis.call('ZREMRANGEBYSCORE', key, '-inf', now - window)
redis.call('ZADD', key, now, guild)
redis.call('EXPIRE', key, window + 1)
return redis.call('ZCARD', key)
"""


@dataclass(frozen=True, slots=True)
class SwarmObservation:
    """The result of recording one cross-guild observation of a phash."""

    distinct_guilds: int
    is_swarming: bool


class SwarmCorrelator:
    """Records phash observations across guilds and flags swarming campaigns."""

    def __init__(
        self,
        redis: object,
        *,
        min_guilds: int = 3,
        window_seconds: int = 300,
        prefix: str = _SWARM_PREFIX,
    ) -> None:
        if min_guilds < 1:
            raise ValueError("min_guilds must be >= 1")
        self._redis = redis
        self._min = min_guilds
        self._window = window_seconds
        self._prefix = prefix

    @property
    def window_seconds(self) -> int:
        """The sliding-window width in seconds."""
        return self._window

    def _key(self, phash: int) -> str:
        return f"{self._prefix}:{phash}"

    async def observe(self, phash: int, guild_id: int) -> SwarmObservation:
        """Record that ``guild_id`` just saw ``phash``; return the swarm state."""
        now = time.time()
        count = await self._redis.eval(  # type: ignore[attr-defined]
            _SWARM_SCRIPT,
            1,
            self._key(phash),
            now,
            self._window,
            guild_id,
        )
        distinct = int(count)
        return SwarmObservation(distinct_guilds=distinct, is_swarming=distinct >= self._min)
