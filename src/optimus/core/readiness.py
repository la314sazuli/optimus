"""Readiness-probe factories for the dependencies services share.

Each factory returns an async predicate suitable for
:meth:`optimus.core.health.HealthServer.add_readiness_check`. Probes never
raise: a failed dependency resolves to ``False`` so ``/readyz`` reports 503
rather than crashing the probe handler.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from typing import Any, Protocol, runtime_checkable

ReadinessCheck = Callable[[], Awaitable[bool]]


@runtime_checkable
class _Executable(Protocol):
    def execute(self, statement: Any) -> Awaitable[Any]: ...


#: A zero-arg factory yielding a context manager over a DB session (e.g. the
#: ``session_scope`` helper). Kept structural so this module avoids a hard
#: SQLAlchemy import.
DbSessionScope = Callable[[], AbstractAsyncContextManager[_Executable]]


@runtime_checkable
class _Pingable(Protocol):
    def ping(self) -> Awaitable[Any]: ...


@runtime_checkable
class _Shard(Protocol):
    @property
    def is_alive(self) -> bool: ...

    @property
    def is_connected(self) -> bool: ...


@runtime_checkable
class _Sharded(Protocol):
    @property
    def shards(self) -> Any: ...


def redis_check(redis: object | None) -> ReadinessCheck:
    """Return a probe that is ready when ``redis`` answers ``PING``.

    A ``None`` client (Redis optional at boot) is treated as not ready.
    """

    async def _check() -> bool:
        if not isinstance(redis, _Pingable):
            return False
        try:
            await redis.ping()
        except Exception:
            return False
        return True

    return _check


def nats_check(nc: object) -> ReadinessCheck:
    """Return a probe that is ready while the NATS client reports connected."""

    async def _check() -> bool:
        return bool(getattr(nc, "is_connected", False))

    return _check


def shards_check(bot: object) -> ReadinessCheck:
    """Return a probe that is ready when every shard this replica runs is connected.

    Reads ``bot.shards`` (hikari exposes a mapping of shard id to shard) and
    requires every shard to be both alive and connected. Fail-closed: a bot with
    no shards yet (still starting), an unexpected shape, or any access error
    resolves to ``False`` so ``/readyz`` reports 503 until the gateway is up.
    This matches the existing readiness design where an unready dependency keeps
    the replica out of rotation rather than crashing the probe.
    """

    async def _check() -> bool:
        if not isinstance(bot, _Sharded):
            return False
        try:
            shards = bot.shards
            if not shards:
                return False
            return all(
                isinstance(s, _Shard) and s.is_alive and s.is_connected for s in shards.values()
            )
        except Exception:
            return False

    return _check


def db_check(scope: DbSessionScope) -> ReadinessCheck:
    """Return a probe that is ready when the database answers ``SELECT 1``.

    Use for services whose serving path depends on the database (e.g.
    interactions). Any failure resolves to ``False`` so ``/readyz`` reports 503.
    """
    from sqlalchemy import text

    async def _check() -> bool:
        try:
            async with scope() as session:
                await session.execute(text("SELECT 1"))
        except Exception:
            return False
        return True

    return _check
