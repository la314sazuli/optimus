"""Detection service runtime: bus wiring around :class:`DetectionWorker`.

Builds the index manager (rebuilt from Postgres), subscribes to a core-NATS
invalidation subject for incremental index updates, consumes
``image_fetched.v1``, and publishes ``verdict.v1`` (and ``swarm_alert.v1`` when a
campaign is swarming). Detections are persisted for audit/stats.
"""

from __future__ import annotations

import asyncio
import contextlib
from contextlib import AbstractAsyncContextManager

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from optimus.bus import Bus
from optimus.bus.nats import EventBus
from optimus.contracts.events import (
    SUBJECT_IMAGE_FETCHED,
    SUBJECT_INDEX_INVALIDATE,
    SUBJECT_SWARM_ALERT,
    SUBJECT_VERDICT,
    ImageFetchedEvent,
    IndexInvalidateEvent,
)
from optimus.core.config import Sensitivity, Settings, get_settings
from optimus.core.health import HealthServer
from optimus.core.idempotency import IdempotencyGuard
from optimus.core.logging import configure_logging, get_logger
from optimus.core.readiness import db_check, nats_check, redis_check
from optimus.db.engine import (
    SessionScope,
    create_engine,
    create_session_factory,
    session_scope,
)
from optimus.db.models import Detection
from optimus.db.repositories import DetectionRepository, GuildRepository, WhitelistRepository
from optimus.hashing.decoder import DecodeLimits
from optimus.services.detection.index import HashIndex, IndexManager
from optimus.services.detection.matcher import WhitelistEntry
from optimus.services.detection.swarm import SwarmCorrelator
from optimus.services.detection.worker import DetectionResult, DetectionWorker

_log = get_logger(__name__)


class DetectionService:
    """Owns the worker, index manager, and persistence for detection."""

    def __init__(
        self,
        settings: Settings,
        bus: Bus,
        worker: DetectionWorker,
        index_manager: IndexManager,
        session_scope_factory: SessionScope,
    ) -> None:
        self._settings = settings
        self._bus = bus
        self._worker = worker
        self._indexes = index_manager
        self._scope = session_scope_factory

    @property
    def scope(self) -> SessionScope:
        """The DB session-scope factory backing persistence (for readiness)."""
        return self._scope

    async def on_image(self, event: ImageFetchedEvent) -> None:
        """Process a fetched image and publish its verdict (+ swarm alert)."""
        result = await self._worker.handle(event)
        if result is None:
            return
        await self._persist(result)
        await self._bus.publish(
            SUBJECT_VERDICT, result.verdict, msg_id=result.verdict.idempotency_key
        )
        if result.swarm_alert is not None:
            await self._bus.publish(SUBJECT_SWARM_ALERT, result.swarm_alert)

    async def on_invalidate(self, event: IndexInvalidateEvent) -> None:
        """Reload an index in response to a control-plane invalidation."""
        await self._indexes.invalidate(event.guild_id)
        _log.info("index_invalidated", guild_id=event.guild_id)

    async def _persist(self, result: DetectionResult) -> None:
        v = result.verdict
        async with self._scope() as session:
            repo = DetectionRepository(session, v.guild_id)
            if await repo.get_by_idempotency_key(v.idempotency_key) is not None:
                return
            # The insert runs in a savepoint so a unique-constraint loss only
            # rolls back the failed row, not the surrounding transaction.
            # Concurrent redelivery can race two replicas past the read-check;
            # the loser hits the unique key on idempotency_key. The constraint is
            # the authority, so we swallow it as a no-op rather than nak (which
            # would redeliver a message whose row already exists, forever).
            try:
                async with session.begin_nested():
                    await repo.record(
                        Detection(
                            guild_id=v.guild_id,
                            message_id=v.message_id,
                            channel_id=v.channel_id,
                            attachment_id=v.attachment_id,
                            uploader_id=v.uploader_id,
                            distances=dict(v.distances),
                            verdict=v.verdict.value,
                            idempotency_key=v.idempotency_key,
                        )
                    )
            except IntegrityError:
                pass


def build_service(
    settings: Settings,
    bus: Bus,
    redis: object | None,
    *,
    session_scope_factory: SessionScope | None = None,
    enable_swarm: bool = True,
) -> DetectionService:
    """Wire a :class:`DetectionService` from settings and shared clients.

    ``session_scope_factory`` lets the simple-mode composer pass a shared engine's
    scope instead of opening a second one; distributed mode leaves it ``None`` so
    the service owns its own engine exactly as before.

    ``enable_swarm`` gates the cross-guild :class:`SwarmCorrelator`, which needs a
    real Redis (it issues a Lua ``EVAL``). Simple mode passes ``enable_swarm=False``
    so it can share the in-memory store for the idempotency guard without that
    store needing to emulate ``EVAL``; a single-process bot has no fleet-wide swarm
    signal to correlate anyway. Distributed mode keeps it on, unchanged.
    """
    if session_scope_factory is not None:
        scope = session_scope_factory
    else:
        engine = create_engine()
        factory = create_session_factory(engine)

        def scope() -> AbstractAsyncContextManager[AsyncSession]:
            return session_scope(factory)

    index_manager = IndexManager(scope, max_guilds=settings.detection_guild_index_cap)

    guard = IdempotencyGuard(redis) if redis is not None else _NullGuard()
    swarm = (
        SwarmCorrelator(
            redis,
            min_guilds=settings.swarm_min_guilds,
            window_seconds=settings.swarm_window_seconds,
        )
        if redis is not None and enable_swarm
        else None
    )

    async def guild_index(guild_id: int) -> HashIndex:
        return await index_manager.guild_index(guild_id)

    async def global_index() -> HashIndex:
        return await index_manager.global_index()

    async def whitelist(guild_id: int) -> list[WhitelistEntry]:
        async with scope() as session:
            rows = await WhitelistRepository(session, guild_id).list()
            return [WhitelistEntry(phash=r.phash) for r in rows]

    async def sensitivity(guild_id: int) -> Sensitivity:
        async with scope() as session:
            guild = await GuildRepository(session).get(guild_id)
            if guild is None:
                return settings.sensitivity_default
            return Sensitivity(guild.sensitivity)

    limits = DecodeLimits(
        cpu_seconds=settings.decode_cpu_seconds,
        mem_bytes=settings.decode_mem_bytes,
        wall_timeout=settings.decode_timeout_seconds,
        max_image_pixels=settings.max_image_pixels,
        max_frames=settings.max_frames,
    )

    worker = DetectionWorker(
        guild_index=guild_index,
        global_index=global_index,
        whitelist=whitelist,
        sensitivity=sensitivity,
        idempotency_acquire=guard.acquire,
        swarm=swarm,
        limits=limits,
    )
    return DetectionService(settings, bus, worker, index_manager, scope)


class _NullGuard:
    """Fallback idempotency guard that always permits (no Redis available)."""

    async def acquire(self, key: str) -> bool:
        return True


async def _amain() -> None:
    settings = get_settings()
    configure_logging(level=settings.log_level, service_name="optimus-detection")

    bus, nc = await EventBus.connect(settings.nats_url)
    await bus.ensure_stream(duplicate_window=settings.bus_duplicate_window_seconds)
    redis = _open_redis(settings)
    service = build_service(settings, bus, redis)

    health = HealthServer(host=settings.health_host, port=settings.health_port)
    health.add_readiness_check(nats_check(nc), name="nats")
    # Detection's whole job is decode -> match -> *persist*; with Postgres down it
    # can only nak-and-redeliver, never make progress. Gate readiness on the DB so
    # the probe tells the truth (503) during a DB outage instead of reporting ready
    # while every message bounces — matching how interactions already gates on its DB.
    health.add_readiness_check(db_check(service.scope), name="postgres")
    if redis is not None:
        health.add_readiness_check(redis_check(redis), name="redis")
    await health.start()

    async def _invalidate_cb(raw_msg: object) -> None:
        event = IndexInvalidateEvent.model_validate_json(raw_msg.data)  # type: ignore[attr-defined]
        await service.on_invalidate(event)

    sub = await nc.subscribe(SUBJECT_INDEX_INVALIDATE, cb=_invalidate_cb)

    stop = asyncio.Event()
    consume_task = asyncio.create_task(
        bus.consume(
            SUBJECT_IMAGE_FETCHED,
            durable="detection",
            model=ImageFetchedEvent,
            handler=service.on_image,
            batch=settings.detection_fetch_batch,
            max_deliver=settings.detection_max_deliver,
            max_inflight=settings.detection_max_inflight,
            ack_wait=settings.detection_ack_wait_seconds,
            stop_event=stop,
        )
    )
    try:
        await consume_task
    finally:
        health.set_live(False)
        stop.set()
        with contextlib.suppress(Exception):
            await sub.unsubscribe()
        with contextlib.suppress(Exception):
            await nc.drain()
        await health.stop()


def _open_redis(settings: Settings) -> object | None:
    try:
        import redis.asyncio as aioredis

        return aioredis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_timeout=settings.redis_socket_timeout,
            socket_connect_timeout=settings.redis_socket_timeout,
        )
    except Exception:  # pragma: no cover - redis optional at boot
        _log.warning("redis_unavailable_detection")
        return None


def main() -> None:
    """Console entrypoint: ``python -m optimus.services.detection``."""
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
