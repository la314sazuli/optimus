"""Regression coverage for InteractionService._run's transient-lock retry.

Production hit ``sqlite3.OperationalError: database is locked`` inside
``DetectionService._persist``'s savepoint-guarded flush, surfaced to the user
as a failed "Review as scam" interaction with no retry. ``_run`` is the single
choke point every command/button interaction passes through, so the retry
belongs there rather than in each individual write path.

This suite fakes ``self._scope()`` directly (an ``AbstractAsyncContextManager``
factory) rather than standing up a real aiosqlite engine -- the retry logic
itself is indifferent to *why* SQLite reported a lock, so exercising it
against a controlled sequence of raised exceptions is both faster and a more
precise test of the retry/backoff/give-up state machine than trying to force
a real lock contention race deterministically.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from sqlalchemy.exc import OperationalError

from optimus.core.config import get_settings
from optimus.services.interactions.service import InteractionResponse, InteractionService


class _FakeSession:
    """Stand-in for AsyncSession -- _run only threads it through to DbDeps."""


def _lock_error() -> OperationalError:
    # OperationalError(statement, params, orig) -- `orig` is what
    # _is_sqlite_lock_error inspects, matching aiosqlite's real shape.
    return OperationalError("INSERT ...", (), Exception("database is locked"))


def _other_operational_error() -> OperationalError:
    return OperationalError("SELECT ...", (), Exception("no such table: detections"))


def _make_service(*, scope_factory: Any) -> InteractionService:
    return InteractionService(
        scope=scope_factory,
        rate_limiter=None,  # type: ignore[arg-type]  # unused: call() never touches deps here
        settings=get_settings(),
    )


def _scope_factory() -> Any:
    @asynccontextmanager
    async def _scope() -> AsyncIterator[_FakeSession]:
        yield _FakeSession()

    return _scope


async def test_run_retries_transient_lock_error_and_succeeds() -> None:
    """A lock error on the first attempt(s) should not fail the interaction."""
    attempts = 0

    async def call(_deps: Any) -> InteractionResponse:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise _lock_error()
        return InteractionResponse("ok")

    service = _make_service(scope_factory=_scope_factory())
    result = await service._run(call)

    assert result.i18n_key == "ok"
    assert attempts == 3


async def test_run_gives_up_after_exhausting_retry_budget() -> None:
    """A persistently locked database should still surface as a failure."""
    attempts = 0

    async def call(_deps: Any) -> InteractionResponse:
        nonlocal attempts
        attempts += 1
        raise _lock_error()

    service = _make_service(scope_factory=_scope_factory())
    with pytest.raises(OperationalError, match="database is locked"):
        await service._run(call)

    assert attempts == service._LOCK_RETRY_BACKOFF.max_attempts


async def test_run_does_not_retry_non_lock_operational_error() -> None:
    """Only the specific SQLite lock message is transient -- anything else
    (e.g. a broken migration) must fail immediately on the first attempt."""
    attempts = 0

    async def call(_deps: Any) -> InteractionResponse:
        nonlocal attempts
        attempts += 1
        raise _other_operational_error()

    service = _make_service(scope_factory=_scope_factory())
    with pytest.raises(OperationalError, match="no such table"):
        await service._run(call)

    assert attempts == 1


async def test_run_propagates_non_operational_errors_immediately() -> None:
    """Unrelated exceptions (handler bugs, etc.) must not be swallowed or
    retried -- only OperationalError participates in the retry loop."""
    attempts = 0

    async def call(_deps: Any) -> InteractionResponse:
        nonlocal attempts
        attempts += 1
        raise ValueError("boom")

    service = _make_service(scope_factory=_scope_factory())
    with pytest.raises(ValueError, match="boom"):
        await service._run(call)

    assert attempts == 1


async def test_pool_diagnostics_reports_real_pool_stats(tmp_path: Any) -> None:
    """Against a real file-backed SQLite engine, diagnostics should report the
    actual AsyncAdaptedQueuePool stats rather than falling back to an error."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from optimus.db.engine import create_engine
    from optimus.services.interactions.service import _pool_diagnostics

    db_path = tmp_path / "diag.db"
    engine = create_engine(f"sqlite+aiosqlite:///{db_path}")
    try:
        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with factory() as session:
            # A freshly opened AsyncSession is lazy -- it doesn't check out a
            # connection from the pool until it actually runs something, so
            # execute a no-op query first to get a realistic "mid-request"
            # snapshot matching what _run's retry path observes.
            await session.execute(text("SELECT 1"))
            diagnostics = _pool_diagnostics(session)
            assert diagnostics["pool_class"] == "AsyncAdaptedQueuePool"
            assert diagnostics["checkedout"] == 1  # this session's own connection
            assert isinstance(diagnostics["checkedin"], int)
    finally:
        await engine.dispose()


async def test_pool_diagnostics_falls_back_gracefully_for_fake_session() -> None:
    """A session-like object with no real ``get_bind()`` must not raise --
    diagnostics logging must never mask or replace the real error being
    reported alongside it."""
    from optimus.services.interactions.service import _pool_diagnostics

    diagnostics = _pool_diagnostics(_FakeSession())
    assert "pool_diagnostics_error" in diagnostics
