"""Tests for async engine pool configuration from settings."""

from __future__ import annotations

from optimus.core.config import Settings
from optimus.db.engine import create_engine


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
