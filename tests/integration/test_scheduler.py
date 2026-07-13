"""Integration tests for scheduler maintenance jobs against aiosqlite."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio import AsyncSession as _Session

from optimus.core.config import get_settings
from optimus.db.engine import SessionScope, create_engine, create_session_factory, session_scope
from optimus.db.models import Base, Detection, Evidence, Guild, ModAction
from optimus.services.scheduler import tasks


@pytest_asyncio.fixture
async def scope() -> AsyncIterator[SessionScope]:
    """A session-scope factory over one shared in-memory database."""
    engine: AsyncEngine = create_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = create_session_factory(engine)

    @asynccontextmanager
    async def _scope(guild_id: int | None = None) -> AsyncIterator[_Session]:
        async with session_scope(factory) as s:
            yield s

    # The single connection of an in-memory sqlite engine is reused, so the
    # schema persists across scopes for the duration of the fixture.
    yield _scope
    await engine.dispose()


async def _add_guild(scope: SessionScope, guild_id: int, *, retention_days: int = 30) -> None:
    async with scope() as s:
        s.add(Guild(guild_id=guild_id, retention_days=retention_days))


async def _add_detection(scope: SessionScope, guild_id: int, *, key: str, created: datetime) -> int:
    async with scope() as s:
        det = Detection(
            guild_id=guild_id,
            message_id=1,
            channel_id=2,
            attachment_id=3,
            uploader_id=4,
            distances={},
            verdict="scam",
            idempotency_key=key,
            created_at=created,
        )
        s.add(det)
        await s.flush()
        return det.id


async def test_retention_deletes_only_old_rows(scope: SessionScope) -> None:
    now = datetime(2026, 6, 1, tzinfo=UTC)
    await _add_guild(scope, 1, retention_days=30)
    await _add_detection(scope, 1, key="old", created=now - timedelta(days=40))
    await _add_detection(scope, 1, key="new", created=now - timedelta(days=5))

    deleted = await tasks.enforce_retention(scope, default_days=30, now=now)
    assert deleted == 1
    async with scope() as s:
        remaining = (await s.execute(Detection.__table__.select())).fetchall()
    assert len(remaining) == 1


async def test_retention_respects_per_guild_window(scope: SessionScope) -> None:
    now = datetime(2026, 6, 1, tzinfo=UTC)
    await _add_guild(scope, 1, retention_days=7)
    await _add_detection(scope, 1, key="d", created=now - timedelta(days=10))
    deleted = await tasks.enforce_retention(scope, default_days=30, now=now)
    assert deleted == 1


async def test_retention_also_clears_mod_actions(scope: SessionScope) -> None:
    now = datetime(2026, 6, 1, tzinfo=UTC)
    await _add_guild(scope, 1, retention_days=10)
    async with scope() as s:
        s.add(
            ModAction(
                guild_id=1,
                actor_id=0,
                action="delete_ban",
                target="4",
                payload={},
                created_at=now - timedelta(days=20),
            )
        )
    deleted = await tasks.enforce_retention(scope, default_days=30, now=now)
    assert deleted == 1


async def _add_appeal(
    scope: SessionScope, guild_id: int, detection_id: int, *, created: datetime
) -> None:
    from optimus.db.models import Appeal

    async with scope() as s:
        s.add(
            Appeal(
                guild_id=guild_id,
                detection_id=detection_id,
                user_id=7,
                created_at=created,
            )
        )


async def _count(scope: SessionScope, model: object) -> int:
    async with scope() as s:
        rows = (await s.execute(model.__table__.select())).fetchall()  # type: ignore[attr-defined]
    return len(rows)


async def test_purge_disabled_by_default_keeps_everything(scope: SessionScope) -> None:
    now = datetime(2026, 6, 1, tzinfo=UTC)
    await _add_guild(scope, 1)
    await _add_detection(scope, 1, key="ancient", created=now - timedelta(days=999))
    deleted = await tasks.purge_old_data(
        scope, retention_days=None, batch_size=100, pause_seconds=0.0, now=now
    )
    assert deleted == 0
    assert await _count(scope, Detection) == 1


async def test_purge_deletes_only_old_rows(scope: SessionScope) -> None:
    now = datetime(2026, 6, 1, tzinfo=UTC)
    await _add_guild(scope, 1)
    await _add_detection(scope, 1, key="old", created=now - timedelta(days=40))
    await _add_detection(scope, 1, key="new", created=now - timedelta(days=5))
    deleted = await tasks.purge_old_data(
        scope, retention_days=30, batch_size=100, pause_seconds=0.0, now=now
    )
    assert deleted == 1
    assert await _count(scope, Detection) == 1


async def test_purge_respects_batch_size_and_loops_until_done(scope: SessionScope) -> None:
    now = datetime(2026, 6, 1, tzinfo=UTC)
    await _add_guild(scope, 1)
    for i in range(5):
        await _add_detection(scope, 1, key=f"old-{i}", created=now - timedelta(days=40))

    pauses: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        pauses.append(seconds)

    deleted = await tasks.purge_old_data(
        scope,
        retention_days=30,
        batch_size=2,
        pause_seconds=0.5,
        now=now,
        sleep=fake_sleep,
    )
    assert deleted == 5
    assert await _count(scope, Detection) == 0
    # 5 rows / batch 2 -> batches of 2, 2, 1; a pause follows each full batch
    # (the two of size 2), none after the short final batch. Appeals table is
    # empty so it contributes one immediate short batch (no pause).
    assert pauses == [0.5, 0.5]


async def test_purge_clears_appeals_before_detections(scope: SessionScope) -> None:
    from optimus.db.models import Appeal

    now = datetime(2026, 6, 1, tzinfo=UTC)
    await _add_guild(scope, 1)
    det_id = await _add_detection(scope, 1, key="d", created=now - timedelta(days=40))
    await _add_appeal(scope, 1, det_id, created=now - timedelta(days=40))
    deleted = await tasks.purge_old_data(
        scope, retention_days=30, batch_size=100, pause_seconds=0.0, now=now
    )
    # One appeal + one detection.
    assert deleted == 2
    assert await _count(scope, Appeal) == 0
    assert await _count(scope, Detection) == 0


async def test_service_retention_purge_disabled_returns_zero(scope: SessionScope) -> None:
    now = datetime(2026, 6, 1, tzinfo=UTC)
    await _add_guild(scope, 1)
    await _add_detection(scope, 1, key="ancient", created=now - timedelta(days=999))
    svc = _service(scope, _FakeBus())
    # Default settings leave detection_retention_days unset, so the job is a no-op.
    assert await svc._retention_purge() == 0  # type: ignore[attr-defined]
    assert await _count(scope, Detection) == 1


async def test_service_retention_purge_enabled_via_settings(scope: SessionScope) -> None:
    from optimus.services.scheduler.service import SchedulerService

    now = datetime(2026, 6, 1, tzinfo=UTC)
    await _add_guild(scope, 1)
    await _add_detection(scope, 1, key="old", created=now - timedelta(days=40))
    settings = get_settings().model_copy(
        update={
            "detection_retention_days": 30,
            "retention_batch_size": 100,
            "retention_batch_pause_seconds": 0.0,
        }
    )
    svc = SchedulerService(settings, _FakeBus(), scope)  # type: ignore[arg-type]
    purged = await svc._retention_purge()  # type: ignore[attr-defined]
    assert purged == 1
    assert await _count(scope, Detection) == 0


async def test_rollup_counts_previous_hour(scope: SessionScope) -> None:
    # "now" is 10:30; the rolled-up bucket is 09:00-10:00.
    now = datetime(2026, 6, 1, 10, 30, tzinfo=UTC)
    await _add_guild(scope, 1)
    await _add_detection(scope, 1, key="in", created=datetime(2026, 6, 1, 9, 15, tzinfo=UTC))
    await _add_detection(scope, 1, key="out", created=datetime(2026, 6, 1, 10, 15, tzinfo=UTC))

    written = await tasks.roll_up_stats(scope, now=now)
    assert written == 1
    from optimus.db.models import StatsRollup

    async with scope() as s:
        rows = (await s.execute(StatsRollup.__table__.select())).fetchall()
    assert len(rows) == 1
    assert rows[0].detections == 1
    # sqlite returns naive datetimes; compare on the wall-clock fields.
    assert rows[0].bucket_start.replace(tzinfo=UTC) == datetime(2026, 6, 1, 9, 0, tzinfo=UTC)


async def test_evidence_cleanup_deletes_expired(scope: SessionScope) -> None:
    now = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    await _add_guild(scope, 1)
    det_id = await _add_detection(scope, 1, key="e", created=now)
    async with scope() as s:
        s.add(
            Evidence(
                detection_id=det_id,
                object_key="evidence/1/9",
                expires_at=now - timedelta(hours=2),
            )
        )
        s.add(
            Evidence(
                detection_id=det_id,
                object_key="evidence/1/10",
                expires_at=now + timedelta(hours=2),
            )
        )

    deleted_keys: list[str] = []

    async def delete_object(key: str) -> None:
        deleted_keys.append(key)

    removed = await tasks.cleanup_evidence(scope, delete_object=delete_object, now=now)
    assert removed == 1
    assert deleted_keys == ["evidence/1/9"]
    async with scope() as s:
        remaining = (await s.execute(Evidence.__table__.select())).fetchall()
    assert len(remaining) == 1


class _FakeBus:
    """Captures published (subject, event) pairs."""

    def __init__(self) -> None:
        self.published: list[tuple[str, object]] = []

    async def publish(self, subject: str, event: object) -> None:
        self.published.append((subject, event))


def _service(scope: SessionScope, bus: _FakeBus) -> object:
    from optimus.services.scheduler.service import SchedulerService

    return SchedulerService(get_settings(), bus, scope)  # type: ignore[arg-type]


async def test_service_retention_and_rollup_jobs_delegate_to_tasks(
    scope: SessionScope,
) -> None:
    old = datetime(2020, 1, 1, tzinfo=UTC)
    await _add_guild(scope, 1)
    await _add_detection(scope, 1, key="ancient", created=old)

    svc = _service(scope, _FakeBus())
    # _retention purges the ancient detection (default 30-day window).
    assert await svc._retention() == 1  # type: ignore[attr-defined]
    async with scope() as s:
        from optimus.db.models import Detection

        assert (await s.execute(Detection.__table__.select())).fetchall() == []


async def test_service_index_rebuild_publishes_invalidate_event(
    scope: SessionScope,
) -> None:
    from optimus.contracts.events import SUBJECT_INDEX_INVALIDATE, IndexInvalidateEvent

    bus = _FakeBus()
    svc = _service(scope, bus)
    affected = await svc._index_rebuild()  # type: ignore[attr-defined]
    assert affected == 1
    assert len(bus.published) == 1
    subject, event = bus.published[0]
    assert subject == SUBJECT_INDEX_INVALIDATE
    assert isinstance(event, IndexInvalidateEvent)
    assert event.correlation_id == "scheduler"


async def test_service_health_sweep_pings_db(scope: SessionScope) -> None:
    svc = _service(scope, _FakeBus())
    # A successful DB ping reports zero rows affected.
    assert await svc._health_sweep() == 0  # type: ignore[attr-defined]


async def test_service_evidence_job_runs_with_default_noop_deleter(
    scope: SessionScope,
) -> None:
    # No delete_object injected -> the no-op deleter is used; with no expired
    # rows the job simply reports zero.
    svc = _service(scope, _FakeBus())
    assert await svc._evidence() == 0  # type: ignore[attr-defined]


async def test_service_start_launches_loops_and_request_stop_unwinds(
    scope: SessionScope,
) -> None:
    import asyncio

    svc = _service(scope, _FakeBus())
    handles = svc.start()  # type: ignore[attr-defined]
    assert len(handles) == 6
    # Intervals default to minutes, so loops are parked in their first sleep;
    # request_stop wakes them and they exit cooperatively.
    svc.request_stop()  # type: ignore[attr-defined]
    await asyncio.wait_for(asyncio.gather(*handles), timeout=2.0)
    assert all(h.done() for h in handles)
