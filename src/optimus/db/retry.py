"""Retry helper for SQLite's transient ``database is locked`` error.

Under WAL with ``busy_timeout=0``, a second concurrent writer fails
immediately with ``database is locked``. That collision is transient by
nature — the winning writer holds the lock for milliseconds — so callers
retry it with a short backoff. Every other ``OperationalError`` (broken
migration, missing table, …) is genuine and must surface unchanged.

This module is the single home for that pattern; services must not
re-implement it locally.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from sqlalchemy.exc import OperationalError

from optimus.core.backoff import BackoffPolicy, retry_async

#: Tuned for WAL writer collisions: total worst-case wait is well under 2s.
SQLITE_LOCK_RETRY = BackoffPolicy(base=0.05, multiplier=2.0, max_delay=0.5, max_attempts=5)


class NonRetryableDbError(Exception):
    """Sentinel: an ``OperationalError`` that is not a lock error.

    Raised from inside a retried closure to stop the retry loop immediately
    while letting the original exception surface unchanged to the caller.
    """


def is_sqlite_lock_error(exc: OperationalError) -> bool:
    """Whether ``exc`` is SQLite's transient ``database is locked`` error."""
    return "database is locked" in str(exc.orig).lower()


async def retry_sqlite_lock[T](
    operation: Callable[[], Awaitable[T]],
    *,
    policy: BackoffPolicy = SQLITE_LOCK_RETRY,
    on_retry: Callable[[], None] | None = None,
) -> T:
    """Run ``operation``, retrying only SQLite lock collisions.

    ``on_retry`` is invoked once per lock collision before the backoff sleep
    (for logging/metrics). Non-lock ``OperationalError``s and all other
    exceptions propagate unchanged on the first occurrence.
    """

    async def attempt() -> T:
        try:
            return await operation()
        except OperationalError as exc:
            if not is_sqlite_lock_error(exc):
                raise NonRetryableDbError from exc
            if on_retry is not None:
                on_retry()
            raise

    try:
        return await retry_async(attempt, policy, retry_on=(OperationalError,))
    except NonRetryableDbError as exc:
        assert exc.__cause__ is not None
        raise exc.__cause__ from None
