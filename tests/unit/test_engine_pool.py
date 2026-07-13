"""Tests for async engine pool configuration from settings."""

from __future__ import annotations

import pytest
from sqlalchemy import func, select

from optimus.core.config import Settings
from optimus.db.engine import create_engine, create_session_factory
from optimus.db.models import Base, Detection, Evidence, Guild, GuildHash


def test_sqlite_engine_ignores_pool_settings() -> None:
    # SQLite has no server-side pool; create_engine must not pass QueuePool
    # kwargs (which would raise) for a sqlite URL.
    engine = create_engine("sqlite+aiosqlite://")
    assert engine.url.get_backend_name() == "sqlite"


def test_postgres_engine_applies_pool_settings() -> None:
    settings = Settings(
        _env_file=None,
        db_pool_size=7,
        db_max_overflow=3,
        db_pool_recycle=60,
    )
    # Build (but never connect) a postgres engine and inspect the sync pool.
    engine = create_engine("postgresql+asyncpg://u:p@localhost:5432/db", settings=settings)
    pool = engine.pool
    assert pool.size() == 7
    assert pool._max_overflow == 3  # type: ignore[attr-defined]
    assert pool._recycle == 60  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_sqlite_enforces_ondelete_cascade() -> None:
    # SQLite disables FK enforcement per connection by default, turning every
    # ``ondelete="CASCADE"`` into a silent no-op. create_engine must set
    # ``PRAGMA foreign_keys=ON`` so deleting a parent purges its children in
    # simple mode (the retention/GDPR paths rely on this).
    engine = create_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = create_session_factory(engine)
    try:
        async with factory() as session:
            session.add(Guild(guild_id=1))
            session.add(
                GuildHash(guild_id=1, hash_id="h1", phash=1, dhash=2, whash=3)
            )
            detection = Detection(
                guild_id=1,
                message_id=10,
                channel_id=20,
                attachment_id=30,
                uploader_id=40,
                verdict="scam",
                idempotency_key="k1",
            )
            session.add(detection)
            await session.flush()
            session.add(
                Evidence(
                    detection_id=detection.id,
                    object_key="obj",
                    expires_at=func.now(),
                )
            )
            await session.commit()

        async with factory() as session:
            await session.delete(await session.get(Guild, 1))
            await session.delete(await session.get(Detection, detection.id))
            await session.commit()

        async with factory() as session:
            guild_hashes = (
                await session.execute(select(func.count()).select_from(GuildHash))
            ).scalar_one()
            evidence = (
                await session.execute(select(func.count()).select_from(Evidence))
            ).scalar_one()
        assert guild_hashes == 0
        assert evidence == 0
    finally:
        await engine.dispose()
