"""Cross-guild swarm correlation via a Redis sorted-set sliding window.

When the same scam image (keyed by phash) is seen in at least ``min_guilds``
distinct guilds inside ``window_seconds``, the campaign is "swarming": the
verdict's confidence is escalated one band and a ``swarm_alert`` is emitted. The
window is a Redis sorted set of ``guild_id`` members scored by timestamp; stale
members are trimmed on each observation.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from prometheus_client import Counter

from optimus.core.logging import get_logger

_log = get_logger(__name__)

#: Incremented when a swarm ``observe`` cannot reach Redis and degrades to a
#: non-swarming result. A non-zero rate means cross-guild correlation is
#: temporarily off (campaigns won't be escalated) — but per-image verdicts are
#: unaffected and messages are not stalled.
REDIS_SWARM_FALLBACK = Counter(
    "optimus_swarm_redis_fallback_total",
    "Swarm observations that failed safe (non-swarming) after a Redis error.",
)

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
        op_timeout: float = 3.0,
    ) -> None:
        if min_guilds < 1:
            raise ValueError("min_guilds must be >= 1")
        if op_timeout <= 0:
            raise ValueError("op_timeout must be > 0")
        self._redis = redis
        self._min = min_guilds
        self._window = window_seconds
        self._prefix = prefix
        self._op_timeout = op_timeout
        self._degraded = False

    @property
    def window_seconds(self) -> int:
        """The sliding-window width in seconds."""
        return self._window

    def _key(self, phash: int) -> str:
        return f"{self._prefix}:{phash}"

    async def observe(self, phash: int, guild_id: int) -> SwarmObservation:
        """Record that ``guild_id`` just saw ``phash``; return the swarm state.

        Fails safe on a Redis error: returns a non-swarming observation rather
        than propagating, so a Redis outage degrades swarm correlation to off
        (the per-image verdict still stands) instead of nak'ing — and thereby
        stalling — the very SCAM/AMBIGUOUS messages this enrichment runs on. The
        failure is counted via :data:`REDIS_SWARM_FALLBACK` so the degradation
        is observable.
        """
        now = time.time()
        try:
            count = await asyncio.wait_for(
                self._redis.eval(  # type: ignore[attr-defined]
                    _SWARM_SCRIPT,
                    1,
                    self._key(phash),
                    now,
                    self._window,
                    guild_id,
                ),
                timeout=self._op_timeout,
            )
        except Exception:  # includes asyncio.TimeoutError (TimeoutError subclass)
            REDIS_SWARM_FALLBACK.inc()
            if not self._degraded:
                self._degraded = True
                _log.warning("swarm_redis_fallback", phash=phash)
            return SwarmObservation(distinct_guilds=0, is_swarming=False)
        self._degraded = False
        distinct = int(count)
        return SwarmObservation(distinct_guilds=distinct, is_swarming=distinct >= self._min)
