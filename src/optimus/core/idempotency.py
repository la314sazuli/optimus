"""Idempotency guards backed by Redis ``SET NX``."""

from __future__ import annotations

import asyncio

from prometheus_client import Counter

from optimus.core.logging import get_logger

_log = get_logger(__name__)

#: Incremented whenever an ``acquire`` cannot reach Redis and the guard fails
#: open (permits the work) instead of stalling the consumer. A non-zero rate
#: means dedup is temporarily degraded — duplicate processing is possible, but
#: the downstream unique constraint on ``idempotency_key`` still prevents
#: duplicate persisted rows, so the failure mode is wasted work, not bad data.
REDIS_IDEMPOTENCY_FALLBACK = Counter(
    "optimus_idempotency_redis_fallback_total",
    "Idempotency acquisitions that failed open after a Redis error.",
)


def build_key(
    message_id: int | str, attachment_id: int | str, *, prefix: str = "optimus:idem"
) -> str:
    """Build the canonical idempotency key for a message attachment."""
    return f"{prefix}:{message_id}:{attachment_id}"


class IdempotencyGuard:
    """Single-acquire guard using Redis ``SET key value NX EX ttl``.

    :meth:`acquire` returns ``True`` exactly once per key within the TTL window,
    ensuring retries never double-act on the same attachment.

    Graceful degradation: if Redis errors at runtime (connection loss, timeout)
    :meth:`acquire` **fails open** — it returns ``True`` so the caller processes
    the message rather than nak'ing it. Failing closed would turn a Redis blip
    into a redelivery storm that stalls the whole consumer, while dedup is only
    best-effort here: the detection persistence path carries a unique constraint
    on ``idempotency_key`` that still collapses genuine duplicates, so the worst
    case of a fail-open is wasted recompute, not a double-acted message. Each
    fail-open increments :data:`REDIS_IDEMPOTENCY_FALLBACK` and logs once per
    outage so the degradation is observable rather than silent.
    """

    def __init__(
        self, redis: object, *, ttl_seconds: int = 86_400, op_timeout: float = 3.0
    ) -> None:
        if ttl_seconds < 1:
            raise ValueError("ttl_seconds must be >= 1")
        if op_timeout <= 0:
            raise ValueError("op_timeout must be > 0")
        self._redis = redis
        self._ttl = ttl_seconds
        self._op_timeout = op_timeout
        self._degraded = False

    async def acquire(self, key: str, token: str = "1") -> bool:  # noqa: S107 - sentinel value, not a credential
        """Atomically claim ``key``; return whether this caller won the claim.

        On a Redis error — or if the call exceeds ``op_timeout`` — this fails
        open (returns ``True``) so the caller processes the message rather than
        nak'ing it. The explicit ``op_timeout`` is a hard ceiling that does not
        depend on the Redis client honoring its own socket timeout under a pool
        reconnect: a dead Redis must not pin a consumer's in-flight slot.
        """
        try:
            result = await asyncio.wait_for(
                self._redis.set(key, token, nx=True, ex=self._ttl),  # type: ignore[attr-defined]
                timeout=self._op_timeout,
            )
        except Exception:  # includes asyncio.TimeoutError (TimeoutError subclass)
            REDIS_IDEMPOTENCY_FALLBACK.inc()
            if not self._degraded:
                self._degraded = True
                _log.warning("idempotency_redis_fallback", key=key)
            return True
        self._degraded = False
        return result is True or result == "OK"

    async def seen(self, key: str) -> bool:
        """Whether ``key`` has already been claimed."""
        exists = await self._redis.exists(key)  # type: ignore[attr-defined]
        return bool(exists)

    async def release(self, key: str) -> None:
        """Release a claim (e.g. to allow reprocessing after a failure)."""
        await self._redis.delete(key)  # type: ignore[attr-defined]
