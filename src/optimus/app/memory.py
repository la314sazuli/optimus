"""A tiny in-process key/value store with the slice of the Redis async API that
simple mode needs.

Simple mode reuses the *real* Redis-backed helpers — :class:`IdempotencyGuard`,
the moderation :class:`Cooldown`, and :class:`GuildConfigCache` — so their dedup
and cooldown semantics are exercised exactly as in production, just against this
process-local store instead of a Redis server. Only the handful of commands those
helpers issue are implemented: ``set`` (with ``nx``/``ex``), ``get``, ``exists``,
``delete``, and ``ping``. TTLs are honoured lazily on access (a key past its expiry
reads as absent) *and* swept actively: the idempotency keys this store mostly holds
are written once and never read again, so lazy expiry alone never reclaims them and
the map would grow without bound under sustained unique traffic. An amortised sweep
every :data:`_SWEEP_EVERY` writes drops expired entries so the footprint tracks the
live keyspace, not the lifetime key count.

This is not a Redis emulator: it is single-process, unsynchronised across
processes, and intentionally minimal. The accepted trade-off for zero external
services is that everything lives in memory and a restart clears it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

#: Run an expiry sweep once every this many writes. Amortises the O(n) scan to
#: O(1) per write while keeping the map bounded to roughly the live keyspace.
_SWEEP_EVERY = 1000


@dataclass
class _Entry:
    value: str
    expires_at: float | None


class MemoryStore:
    """Process-local stand-in for an async Redis client (subset of commands)."""

    def __init__(self, *, time_source: object = time.monotonic) -> None:
        self._data: dict[str, _Entry] = {}
        self._now = time_source
        self._writes_since_sweep = 0

    def _clock(self) -> float:
        return float(self._now())  # type: ignore[operator]

    def _live(self, key: str) -> _Entry | None:
        entry = self._data.get(key)
        if entry is None:
            return None
        if entry.expires_at is not None and entry.expires_at <= self._clock():
            del self._data[key]
            return None
        return entry

    def _sweep_expired(self) -> None:
        """Drop every entry whose TTL has passed (active reclaim, not lazy)."""
        now = self._clock()
        expired = [
            key
            for key, entry in self._data.items()
            if entry.expires_at is not None and entry.expires_at <= now
        ]
        for key in expired:
            del self._data[key]

    async def set(
        self,
        key: str,
        value: str,
        *,
        nx: bool = False,
        ex: int | None = None,
    ) -> bool | None:
        """``SET key value [NX] [EX ex]``; returns ``True`` on write, ``None`` if NX lost."""
        if nx and self._live(key) is not None:
            return None
        expires_at = self._clock() + ex if ex is not None else None
        self._data[key] = _Entry(value=value, expires_at=expires_at)
        self._writes_since_sweep += 1
        if self._writes_since_sweep >= _SWEEP_EVERY:
            self._writes_since_sweep = 0
            self._sweep_expired()
        return True

    async def get(self, key: str) -> str | None:
        """``GET key`` honouring lazy TTL expiry."""
        entry = self._live(key)
        return entry.value if entry is not None else None

    async def exists(self, key: str) -> int:
        """``EXISTS key`` -> 1 or 0."""
        return 1 if self._live(key) is not None else 0

    async def delete(self, key: str) -> int:
        """``DEL key`` -> number of keys removed."""
        if self._live(key) is not None:
            del self._data[key]
            return 1
        return 0

    async def ping(self) -> bool:
        """``PING`` — always ready (the store is local and always up)."""
        return True
