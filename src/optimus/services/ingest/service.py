"""Ingest service runtime: bus consumer wiring around :class:`IngestWorker`."""

from __future__ import annotations

import asyncio
import contextlib
from functools import partial

from optimus.bus.nats import EventBus
from optimus.contracts.events import (
    SUBJECT_IMAGE_FETCHED,
    SUBJECT_MESSAGE_IMAGE,
    MessageImageEvent,
)
from optimus.core.config import Settings, get_settings
from optimus.core.health import HealthServer
from optimus.core.logging import configure_logging, get_logger
from optimus.core.ratelimit import (
    RateLimit,
    RateLimiter,
    build_rate_limiter,
)
from optimus.core.readiness import nats_check, redis_check
from optimus.ingest.fetcher import FetchedImage, fetch_image
from optimus.services.ingest.worker import IngestWorker, RateLimitedError

_log = get_logger(__name__)


async def _handle(worker: IngestWorker, bus: EventBus, event: MessageImageEvent) -> None:
    """Consumer callback: fetch and (if successful) publish the fetched event."""
    try:
        fetched = await worker.handle(event)
    except RateLimitedError:
        # Re-raise so the bus naks and redelivers later under back-pressure.
        raise
    if fetched is not None:
        await bus.publish(SUBJECT_IMAGE_FETCHED, fetched, msg_id=fetched.idempotency_key)


def build_worker(settings: Settings, redis: object | None) -> IngestWorker:
    """Construct an :class:`IngestWorker` with the configured fetch + limiter."""
    limiter: RateLimiter = build_rate_limiter(
        settings,
        redis,
        # Degraded fallback (no Redis, or a runtime Redis outage):
        # opportunistically sweep idle guild buckets so the process-local map
        # cannot grow without bound.
        sweep_interval=settings.ingest_inmemory_sweep_seconds,
    )
    rate = RateLimit(
        capacity=settings.ingest_fetch_rate_capacity,
        refill_rate=settings.ingest_fetch_rate_refill,
    )
    fetch = partial(
        fetch_image,
        max_bytes=settings.ingest_max_bytes,
        max_redirects=settings.ingest_max_redirects,
    )

    async def _fetch(url: str) -> FetchedImage:
        return await fetch(url)

    return IngestWorker(
        _fetch, limiter, rate=rate, max_inline_bytes=settings.ingest_max_inline_bytes
    )


async def _amain() -> None:
    settings = get_settings()
    configure_logging(level=settings.log_level, service_name="optimus-ingest")

    bus, nc = await EventBus.connect(settings.nats_url)
    # Fail fast if the inline-image cap cannot actually ride through NATS: an
    # oversized publish would otherwise nak/redeliver/drop silently, so the
    # image is never scanned. Better a loud startup error the operator can fix
    # (raise the server max_payload or lower the inline cap) than silent loss.
    bus.validate_inline_capacity(settings.ingest_max_inline_bytes)
    await bus.ensure_stream()

    redis = _open_redis(settings)
    worker = build_worker(settings, redis)

    health = HealthServer(host=settings.health_host, port=settings.health_port)
    health.add_readiness_check(nats_check(nc), name="nats")
    if redis is not None:
        health.add_readiness_check(redis_check(redis), name="redis")
    await health.start()

    stop = asyncio.Event()
    consume_task = asyncio.create_task(
        bus.consume(
            SUBJECT_MESSAGE_IMAGE,
            durable="ingest",
            model=MessageImageEvent,
            handler=partial(_handle, worker, bus),
            stop_event=stop,
        )
    )
    try:
        await consume_task
    finally:
        health.set_live(False)
        stop.set()
        with contextlib.suppress(Exception):
            await nc.drain()
        await health.stop()


def _open_redis(settings: Settings) -> object | None:
    try:
        import redis.asyncio as aioredis

        return aioredis.from_url(settings.redis_url, decode_responses=True)
    except Exception:  # pragma: no cover - redis optional at boot
        _log.warning("redis_unavailable_ingest")
        return None


def main() -> None:
    """Console entrypoint: ``python -m optimus.services.ingest``."""
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
