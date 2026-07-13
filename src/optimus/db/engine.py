"""Async engine and session factory helpers."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from optimus.core.config import Settings, get_settings

#: A zero-arg factory yielding a transactional :class:`AsyncSession` scope.
SessionScope = Callable[[], AbstractAsyncContextManager[AsyncSession]]


def create_engine(
    url: str | None = None, *, echo: bool = False, settings: Settings | None = None
) -> AsyncEngine:
    """Create an async engine for ``url`` (defaults to configured database URL).

    For pooled backends (Postgres) the QueuePool is sized from settings so the
    connection footprint is tunable per replica; SQLite (used in tests) has no
    server-side pool and is left on SQLAlchemy's defaults.
    """
    settings = settings or get_settings()
    target = url or settings.database_url
    kwargs: dict[str, Any] = {"echo": echo, "future": True}
    if not target.startswith("sqlite"):
        kwargs.update(
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_recycle=settings.db_pool_recycle,
            pool_pre_ping=settings.db_pool_pre_ping,
        )
    engine = create_async_engine(target, **kwargs)
    if target.startswith("sqlite"):
        _apply_sqlite_pragmas(engine, busy_timeout_ms=settings.sqlite_busy_timeout_ms)
    return engine


def _apply_sqlite_pragmas(engine: AsyncEngine, *, busy_timeout_ms: int) -> None:
    """Set per-connection SQLite pragmas so concurrent writers don't fail fast.

    Simple mode runs the detection/moderation pipeline and the interaction
    handlers as concurrent writers against a single SQLite file. SQLite's default
    rollback journal takes a database-wide lock and its default ``busy_timeout`` is
    0, so a second writer raises ``database is locked`` immediately. WAL lets
    readers proceed alongside one writer, and the busy timeout makes the brief
    writer-vs-writer overlaps wait-and-retry instead of erroring.

    ``foreign_keys=ON`` is also required: SQLite disables foreign-key enforcement
    per connection by default, which silently turns every ``ondelete="CASCADE"``
    into a no-op. Without it the retention purge and GDPR erasure paths, which
    lean on cascades to remove child rows (appeals/evidence under a detection,
    detections under a guild), would orphan those rows in simple mode.
    """

    @event.listens_for(engine.sync_engine, "connect")
    def _set_pragmas(dbapi_connection: Any, _record: Any) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA foreign_keys=ON")
        finally:
            cursor.close()


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Create a session factory bound to ``engine``."""
    return async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


@asynccontextmanager
async def session_scope(
    factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """Provide a transactional session scope, committing or rolling back."""
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
