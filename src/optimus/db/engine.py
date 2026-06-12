"""Async engine and session factory helpers."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any

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
    return create_async_engine(target, **kwargs)


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
