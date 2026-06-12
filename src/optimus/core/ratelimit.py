"""Token-bucket rate limiting backed by Redis, with an in-memory fallback."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from prometheus_client import Counter

from optimus.core.logging import get_logger

if TYPE_CHECKING:
    from optimus.core.config import Settings

_log = get_logger(__name__)

#: Incremented once per ``acquire`` that could not reach Redis and was served by
#: the in-memory fallback instead. A non-zero rate here means replicas are no
#: longer sharing a bucket, so effective limits are temporarily per-process.
REDIS_RATELIMIT_FALLBACK = Counter(
    "optimus_ratelimit_redis_fallback_total",
    "Rate-limit acquisitions served by the in-memory fallback after a Redis error.",
)


@dataclass(frozen=True, slots=True)
class RateLimit:
    """A token-bucket configuration.

    ``capacity`` tokens accrue at ``refill_rate`` tokens/second up to the cap.
    """

    capacity: float
    refill_rate: float

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if self.refill_rate <= 0:
            raise ValueError("refill_rate must be positive")


class RateLimiter(Protocol):
    """Common interface for rate limiters."""

    async def acquire(self, key: str, limit: RateLimit, cost: float = 1.0) -> bool:
        """Try to consume ``cost`` tokens for ``key``; return whether allowed."""
        ...


# Atomic token-bucket in Lua. KEYS[1]=bucket key.
# ARGV: capacity, refill_rate, cost, now (seconds, float).
# Returns 1 if allowed, 0 otherwise.
_BUCKET_SCRIPT = """
local capacity = tonumber(ARGV[1])
local refill = tonumber(ARGV[2])
local cost = tonumber(ARGV[3])
local now = tonumber(ARGV[4])

local data = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(data[1])
local ts = tonumber(data[2])
if tokens == nil then
  tokens = capacity
  ts = now
end

local elapsed = now - ts
if elapsed < 0 then elapsed = 0 end
tokens = math.min(capacity, tokens + elapsed * refill)

local allowed = 0
if tokens >= cost then
  allowed = 1
  tokens = tokens - cost
end

redis.call('HSET', KEYS[1], 'tokens', tokens, 'ts', now)
local ttl = math.ceil(capacity / refill) + 1
redis.call('EXPIRE', KEYS[1], ttl)
return allowed
"""


class RedisRateLimiter:
    """Distributed token bucket evaluated atomically by a Redis Lua script.

    The read-modify-write happens entirely inside the Lua script, so a single
    ``EVAL`` round trip is atomic on the Redis server even under concurrent
    acquisitions from many replicas — there is no read-then-write race.

    Graceful degradation: if Redis errors at runtime (connection loss, timeout,
    script error) we **fall back to a process-local in-memory limiter** rather
    than failing open (allowing unlimited traffic) or crashing the request path.
    A per-process bucket still bounds load per replica during the outage; the
    cost is that the shared limit is temporarily multiplied by replica count.
    Each fallback increments :data:`REDIS_RATELIMIT_FALLBACK` and logs once. When
    no ``fallback`` is supplied the Redis error is re-raised (callers that want
    fail-open or fail-closed can decide), so the default remains explicit.
    """

    def __init__(
        self,
        redis: object,
        *,
        prefix: str = "optimus:rl",
        fallback: RateLimiter | None = None,
    ) -> None:
        self._redis = redis
        self._prefix = prefix
        self._fallback = fallback

    async def acquire(self, key: str, limit: RateLimit, cost: float = 1.0) -> bool:
        """Consume ``cost`` tokens atomically in Redis (or via fallback on error)."""
        if cost <= 0:
            raise ValueError("cost must be positive")
        full_key = f"{self._prefix}:{key}"
        try:
            result = await self._redis.eval(  # type: ignore[attr-defined]
                _BUCKET_SCRIPT,
                1,
                full_key,
                limit.capacity,
                limit.refill_rate,
                cost,
                time.time(),
            )
        except Exception:
            if self._fallback is None:
                raise
            REDIS_RATELIMIT_FALLBACK.inc()
            _log.warning("ratelimit_redis_fallback", key=key)
            return await self._fallback.acquire(key, limit, cost)
        return bool(int(result))


@dataclass
class _Bucket:
    tokens: float
    ts: float


@dataclass
class InMemoryRateLimiter:
    """Process-local token bucket, used when Redis is unavailable.

    When ``sweep_interval`` is set (seconds), :meth:`acquire` opportunistically
    runs :meth:`evict_idle` at most once per interval so the bucket map stays
    bounded even though nothing else sweeps the in-memory fallback. The gate is a
    plain timestamp compare on the single event loop, so the sweep never races an
    in-flight ``acquire``. ``None`` (the default) disables auto-sweeping, leaving
    eviction entirely to explicit ``evict_idle`` callers.
    """

    time_source: object = field(default=time.monotonic)
    sweep_interval: float | None = None
    _buckets: dict[str, _Bucket] = field(default_factory=dict)
    _last_sweep: float = field(default=0.0)

    def _now(self) -> float:
        return float(self.time_source())  # type: ignore[operator]

    async def acquire(self, key: str, limit: RateLimit, cost: float = 1.0) -> bool:
        """Consume ``cost`` tokens for ``key`` in process memory."""
        if cost <= 0:
            raise ValueError("cost must be positive")
        now = self._now()
        self._maybe_sweep(now, limit)
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _Bucket(tokens=limit.capacity, ts=now)
            self._buckets[key] = bucket
        elapsed = max(0.0, now - bucket.ts)
        bucket.tokens = min(limit.capacity, bucket.tokens + elapsed * limit.refill_rate)
        bucket.ts = now
        if bucket.tokens >= cost:
            bucket.tokens -= cost
            return True
        return False

    def evict_idle(self, limit: RateLimit) -> int:
        """Drop buckets that have fully refilled, returning how many were freed.

        The bucket map otherwise retains one entry per key ever seen. A caller
        that uses this limiter as a long-lived fallback (Redis unavailable)
        should call this periodically to bound memory under high key churn.
        """
        now = self._now()
        stale = [
            key
            for key, b in self._buckets.items()
            if min(limit.capacity, b.tokens + max(0.0, now - b.ts) * limit.refill_rate)
            >= limit.capacity
        ]
        for key in stale:
            del self._buckets[key]
        return len(stale)

    def _maybe_sweep(self, now: float, limit: RateLimit) -> None:
        """Run a time-gated opportunistic ``evict_idle`` if ``sweep_interval`` is set."""
        if self.sweep_interval is None:
            return
        if self._last_sweep == 0.0:
            # Arm the gate on first use; don't sweep an empty/cold map.
            self._last_sweep = now
            return
        if now - self._last_sweep >= self.sweep_interval:
            self._last_sweep = now
            self.evict_idle(limit)


def build_rate_limiter(
    settings: Settings,
    redis: object | None,
    *,
    sweep_interval: float | None,
) -> RateLimiter:
    """Construct the configured rate limiter.

    Backend is chosen by ``settings.ratelimit_backend`` (defaulting to
    ``memory`` so single-node self-hosters see no change). The ``redis`` backend
    additionally carries an in-memory ``fallback`` so a runtime Redis outage
    degrades to per-process limiting instead of crashing the request path
    (see :class:`RedisRateLimiter`). If ``redis`` mode is requested but no client
    was opened, the in-memory limiter is used directly.

    ``sweep_interval`` is the idle-bucket sweep cadence for whichever in-memory
    limiter ends up live (the standalone memory backend, or the Redis fallback).
    """
    from optimus.core.config import RateLimitBackend

    def _in_memory() -> InMemoryRateLimiter:
        return InMemoryRateLimiter(sweep_interval=sweep_interval)

    if settings.ratelimit_backend is RateLimitBackend.REDIS and redis is not None:
        return RedisRateLimiter(
            redis,
            prefix=settings.ratelimit_redis_prefix,
            fallback=_in_memory(),
        )
    return _in_memory()
