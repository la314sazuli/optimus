"""A Redis-backed per-key cooldown (e.g. one DM warning per user per hour)."""

from __future__ import annotations


class Cooldown:
    """Grants a key at most once per ``window_seconds`` using ``SET NX EX``."""

    def __init__(
        self, redis: object, *, window_seconds: int = 3600, prefix: str = "optimus:cooldown"
    ) -> None:
        if window_seconds < 1:
            raise ValueError("window_seconds must be >= 1")
        self._redis = redis
        self._window = window_seconds
        self._prefix = prefix

    def _key(self, key: str) -> str:
        return f"{self._prefix}:{key}"

    async def acquire(self, key: str) -> bool:
        """Return ``True`` if ``key`` is allowed now (and start its cooldown)."""
        result = await self._redis.set(  # type: ignore[attr-defined]
            self._key(key), "1", nx=True, ex=self._window
        )
        return result is True or result == "OK"
