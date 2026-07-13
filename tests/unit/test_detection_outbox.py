"""Regression tests for detection idempotency-release and the transactional outbox.

These cover the distributed-correctness fixes:

* the idempotency claim is released when the fallible persist/publish fails, so a
  transient DB/broker blip triggers redelivery instead of a silently dropped
  verdict (and is *not* released on success);
* in outbox mode the verdict (and any swarm alert) are persisted alongside the
  ``Detection`` row in one transaction, and nothing is published inline;
* :class:`OutboxRelay` drains staged rows, marks them published, and on a
  publish failure stops the batch (committing prior marks) so the failing row is
  retried in order on the next poll.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio import AsyncSession as _Session

from optimus.contracts.events import (
    SUBJECT_SWARM_ALERT,
    SUBJECT_VERDICT,
    SwarmAlertEvent,
    Verdict,
    VerdictEvent,
)
from optimus.core.config import get_settings
from optimus.db.engine import SessionScope, create_engine, create_session_factory, session_scope
from optimus.db.models import Base, Detection, OutboxEvent
from optimus.db.repositories import OutboxRepository
from optimus.services.detection.relay import OutboxRelay
from optimus.services.detection.service import DetectionService
from optimus.services.detection.worker import DetectionResult


@pytest_asyncio.fixture
async def scope() -> AsyncIterator[SessionScope]:
    engine: AsyncEngine = create_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = create_session_factory(engine)

    @asynccontextmanager
    async def _scope(guild_id: int | None = None) -> AsyncIterator[_Session]:
        async with session_scope(factory, guild_id=guild_id) as s:
            yield s

    yield _scope
    await engine.dispose()


def _verdict(key: str = "idem-1") -> VerdictEvent:
    return VerdictEvent(
        correlation_id="c",
        occurred_at=datetime.now(UTC),
        guild_id=7,
        channel_id=2,
        message_id=3,
        attachment_id=4,
        uploader_id=42,
        idempotency_key=key,
        verdict=Verdict.SCAM,
        confidence=0.9,
    )


def _swarm() -> SwarmAlertEvent:
    return SwarmAlertEvent(
        correlation_id="c",
        occurred_at=datetime.now(UTC),
        phash=123,
        distinct_guilds=3,
        window_seconds=60,
    )


class _StubWorker:
    def __init__(self, result: DetectionResult | None) -> None:
        self._result = result

    async def handle(self, event: object) -> DetectionResult | None:
        return self._result


class _RecordingRelease:
    def __init__(self) -> None:
        self.released: list[str] = []

    async def __call__(self, key: str) -> None:
        self.released.append(key)


class _OkBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, object]] = []

    async def publish(self, subject: str, event: object, *, msg_id: str | None = None) -> None:
        self.published.append((subject, event))


class _BoomBus:
    async def publish(self, subject: str, event: object, *, msg_id: str | None = None) -> None:
        raise RuntimeError("broker down")


def _service(
    scope: SessionScope,
    bus: object,
    *,
    release: object | None = None,
    use_outbox: bool = False,
) -> DetectionService:
    result = DetectionResult(verdict=_verdict())
    return DetectionService(
        get_settings(),
        bus,  # type: ignore[arg-type]
        _StubWorker(result),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        scope,
        idempotency_release=release,  # type: ignore[arg-type]
        use_outbox=use_outbox,
    )


async def test_release_called_when_publish_fails(scope: SessionScope) -> None:
    release = _RecordingRelease()
    svc = _service(scope, _BoomBus(), release=release)
    with pytest.raises(RuntimeError):
        await svc.on_image(_image())
    # The claim is released so JetStream's redelivery re-runs the work.
    assert release.released == ["idem-1"]


async def test_release_not_called_on_success(scope: SessionScope) -> None:
    release = _RecordingRelease()
    bus = _OkBus()
    svc = _service(scope, bus, release=release)
    await svc.on_image(_image())
    assert release.released == []
    assert bus.published[0][0] == SUBJECT_VERDICT


async def test_outbox_mode_persists_verdict_and_alert_atomically(scope: SessionScope) -> None:
    bus = _OkBus()
    result = DetectionResult(verdict=_verdict(), swarm_alert=_swarm())
    svc = DetectionService(
        get_settings(),
        bus,  # type: ignore[arg-type]
        _StubWorker(result),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        scope,
        use_outbox=True,
    )
    await svc.on_image(_image())

    async with scope() as s:
        detections = (await s.execute(Detection.__table__.select())).fetchall()
        outbox = (await s.execute(OutboxEvent.__table__.select())).fetchall()
    assert len(detections) == 1
    subjects = sorted(r.subject for r in outbox)
    assert subjects == sorted([SUBJECT_VERDICT, SUBJECT_SWARM_ALERT])
    # Outbox mode stages rows for the relay; it never publishes inline.
    assert bus.published == []


async def test_relay_drains_and_marks_published(scope: SessionScope) -> None:
    async with scope() as s:
        repo = OutboxRepository(s)
        await repo.enqueue(subject=SUBJECT_VERDICT, payload='{"a":1}', msg_id="k1")
        await repo.enqueue(subject=SUBJECT_SWARM_ALERT, payload='{"b":2}', msg_id=None)

    seen: list[tuple[str, bytes, str | None]] = []

    async def publish(subject: str, payload: bytes, msg_id: str | None = None) -> None:
        seen.append((subject, payload, msg_id))

    relay = OutboxRelay(scope, publish, batch=10)
    published = await relay.drain_once()
    assert published == 2
    assert seen[0] == (SUBJECT_VERDICT, b'{"a":1}', "k1")
    # A second drain finds nothing outstanding.
    assert await relay.drain_once() == 0


async def test_relay_stops_batch_on_publish_failure_and_retries(scope: SessionScope) -> None:
    async with scope() as s:
        repo = OutboxRepository(s)
        await repo.enqueue(subject=SUBJECT_VERDICT, payload="p1", msg_id="k1")
        await repo.enqueue(subject=SUBJECT_VERDICT, payload="p2", msg_id="k2")

    fail = True

    async def publish(subject: str, payload: bytes, msg_id: str | None = None) -> None:
        if fail:
            raise RuntimeError("publish blip")

    relay = OutboxRelay(scope, publish, batch=10)
    assert await relay.drain_once() == 0  # first row fails -> nothing marked
    async with scope() as s:
        remaining = (await s.execute(OutboxEvent.__table__.select())).fetchall()
    assert all(r.published_at is None for r in remaining)

    fail = False
    assert await relay.drain_once() == 2  # both retried in order once healthy


def _image() -> object:
    from optimus.contracts.events import ImageFetchedEvent

    return ImageFetchedEvent(
        correlation_id="c",
        occurred_at=datetime.now(UTC),
        guild_id=7,
        channel_id=2,
        message_id=3,
        attachment_id=4,
        uploader_id=42,
        idempotency_key="idem-1",
        content_type="image/png",
        size_bytes=10,
        sha256="0" * 64,
        data_b64="",
    )
