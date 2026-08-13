"""Tests for the shared SQLite lock-retry helper."""

from __future__ import annotations

import pytest
from sqlalchemy.exc import OperationalError

from optimus.core.backoff import BackoffPolicy
from optimus.db.retry import is_sqlite_lock_error, retry_sqlite_lock

_FAST = BackoffPolicy(base=0.001, multiplier=1.0, max_delay=0.001, max_attempts=3)


def _lock_error() -> OperationalError:
    return OperationalError("INSERT", {}, Exception("database is locked"))


def _other_error() -> OperationalError:
    return OperationalError("SELECT", {}, Exception("no such table: guild_hashes"))


def test_is_sqlite_lock_error() -> None:
    assert is_sqlite_lock_error(_lock_error())
    assert not is_sqlite_lock_error(_other_error())


@pytest.mark.asyncio
async def test_retries_lock_error_until_success() -> None:
    calls = 0

    async def op() -> int:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise _lock_error()
        return 42

    assert await retry_sqlite_lock(op, policy=_FAST) == 42
    assert calls == 3


@pytest.mark.asyncio
async def test_non_lock_operational_error_is_not_retried() -> None:
    calls = 0

    async def op() -> int:
        nonlocal calls
        calls += 1
        raise _other_error()

    with pytest.raises(OperationalError, match="no such table"):
        await retry_sqlite_lock(op, policy=_FAST)
    assert calls == 1


@pytest.mark.asyncio
async def test_exhausted_retries_reraise_lock_error() -> None:
    calls = 0

    async def op() -> int:
        nonlocal calls
        calls += 1
        raise _lock_error()

    with pytest.raises(OperationalError, match="database is locked"):
        await retry_sqlite_lock(op, policy=_FAST)
    assert calls == _FAST.max_attempts


@pytest.mark.asyncio
async def test_on_retry_fires_once_per_collision() -> None:
    calls = 0
    notified = 0

    async def op() -> int:
        nonlocal calls
        calls += 1
        if calls < 2:
            raise _lock_error()
        return 1

    def on_retry() -> None:
        nonlocal notified
        notified += 1

    await retry_sqlite_lock(op, policy=_FAST, on_retry=on_retry)
    assert notified == 1
