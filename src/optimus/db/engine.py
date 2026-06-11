"""Async engine and session factory helpers."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from optimus.core.config import get_settings

#: A zero-arg factory yielding a transactional :class:`AsyncSession` scope.
SessionScope = Callable[[], AbstractAsyncContextManager[AsyncSession]]


def create_engine(url: str | None = None, *, echo: bool = False) -> AsyncEngine:
    """Create an async engine for ``url`` (defaults to configured database URL)."""
    return create_async_engine(url or get_settings().database_url, echo=echo, future=True)


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
