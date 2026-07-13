"""Async engine and session factory helpers."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any, Protocol

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from optimus.core.config import Settings, get_settings


class SessionScope(Protocol):
    """A factory yielding a transactional :class:`AsyncSession` scope.

    Called with the ``guild_id`` of the tenant whose rows the scope will touch.
    In multi-tenant mode that id is pushed into the ``optimus.guild_id`` session
    GUC so Postgres row-level security (migration ``0002``) enforces isolation on
    every query in the scope. ``guild_id=None`` (the default) opens an unscoped
    transaction for cross-tenant/global work (readiness probes, the global-hash
    index, deployment-wide enumeration).
    """

    def __call__(
        self, guild_id: int | None = None
    ) -> AbstractAsyncContextManager[AsyncSession]: ...


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
    *,
    guild_id: int | None = None,
    multi_tenant: bool = False,
) -> AsyncIterator[AsyncSession]:
    """Provide a transactional session scope, committing or rolling back.

    When ``multi_tenant`` is set and a ``guild_id`` is given, the transaction's
    ``optimus.guild_id`` GUC is set as its first statement so Postgres RLS scopes
    every subsequent query/insert to that tenant. ``set_config(..., is_local =>
    true)`` ties the setting to this transaction, so it is discarded on commit or
    rollback and never leaks to the next checkout of a pooled connection. In
    single-tenant mode (and on SQLite, which has no ``set_config``) no GUC is set
    and behaviour is unchanged.
    """
    async with factory() as session:
        try:
            if multi_tenant and guild_id is not None:
                await session.execute(
                    text("SELECT set_config('optimus.guild_id', :gid, true)"),
                    {"gid": str(guild_id)},
                )
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def create_session_scope(
    factory: async_sessionmaker[AsyncSession], *, multi_tenant: bool = False
) -> SessionScope:
    """Bind ``factory`` (and the tenancy switch) into a :class:`SessionScope`.

    Every service composes its persistence around the returned callable, so the
    RLS GUC wiring lives in one place: callers just pass the ``guild_id`` of the
    tenant they are about to touch.
    """

    def scope(guild_id: int | None = None) -> AbstractAsyncContextManager[AsyncSession]:
        return session_scope(factory, guild_id=guild_id, multi_tenant=multi_tenant)

    return scope
