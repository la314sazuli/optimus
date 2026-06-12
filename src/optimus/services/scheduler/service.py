"""Scheduler runtime: jittered periodic loops with per-run metrics.

Each maintenance job runs on its own interval with additive jitter so loops
don't thunder together. Every run is wrapped in try/except and counted; a
failing job never kills its loop or the others. Shutdown is cooperative via a
shared stop event.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING

from prometheus_client import Counter, Gauge
from sqlalchemy.ext.asyncio import AsyncSession

if TYPE_CHECKING:
    from sqlalchemy.sql.elements import TextClause

from optimus.bus.nats import EventBus
from optimus.contracts.events import SUBJECT_INDEX_INVALIDATE, IndexInvalidateEvent
from optimus.core.config import Settings, get_settings
from optimus.core.health import HealthServer
from optimus.core.logging import configure_logging, get_logger
from optimus.core.readiness import nats_check
from optimus.db.engine import (
    SessionScope,
    create_engine,
    create_session_factory,
    session_scope,
)
from optimus.services.scheduler import tasks

_log = get_logger(__name__)

TASK_RUNS = Counter(
    "optimus_scheduler_task_runs_total",
    "Scheduler task executions.",
    ["task", "outcome"],
)
TASK_AFFECTED = Counter(
    "optimus_scheduler_rows_affected_total",
    "Rows affected by scheduler tasks.",
    ["task"],
)
LAST_RUN = Gauge(
    "optimus_scheduler_last_run_timestamp",
    "Unix timestamp of the last successful run per task.",
    ["task"],
)


def jittered_interval(base: float, fraction: float, rng: random.Random | None = None) -> float:
    """Return ``base`` plus up to ``fraction*base`` of additive jitter."""
    if base <= 0:
        raise ValueError("base must be positive")
    r = rng or random
    return base + r.uniform(0.0, max(0.0, fraction) * base)


async def run_periodic(
    name: str,
    interval: float,
    job: Callable[[], Awaitable[int]],
    *,
    stop: asyncio.Event,
    jitter_fraction: float = 0.1,
    rng: random.Random | None = None,
    time_source: Callable[[], float] | None = None,
) -> None:
    """Run ``job`` every ~``interval`` seconds until ``stop`` is set.

    Each run is isolated: exceptions are logged and counted but never propagate.
    """
    import time

    clock = time_source or time.time
    while not stop.is_set():
        delay = jittered_interval(interval, jitter_fraction, rng)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=delay)
        if stop.is_set():
            break
        try:
            affected = await job()
        except Exception:
            TASK_RUNS.labels(task=name, outcome="error").inc()
            _log.exception("scheduler_task_failed", task=name)
            continue
        TASK_RUNS.labels(task=name, outcome="ok").inc()
        TASK_AFFECTED.labels(task=name).inc(affected)
        LAST_RUN.labels(task=name).set(clock())


class SchedulerService:
    """Wires the maintenance jobs to their loops."""

    def __init__(
        self,
        settings: Settings,
        bus: EventBus,
        scope: SessionScope,
        *,
        delete_object: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self._settings = settings
        self._bus = bus
        self._scope = scope
        self._delete_object = delete_object or _noop_delete
        self._stop = asyncio.Event()

    async def _retention(self) -> int:
        return await tasks.enforce_retention(self._scope, default_days=30)

    async def _evidence(self) -> int:
        return await tasks.cleanup_evidence(self._scope, delete_object=self._delete_object)

    async def _rollups(self) -> int:
        return await tasks.roll_up_stats(self._scope)

    async def _index_rebuild(self) -> int:
        from datetime import UTC, datetime

        await self._bus.publish(
            SUBJECT_INDEX_INVALIDATE,
            IndexInvalidateEvent(correlation_id="scheduler", occurred_at=datetime.now(UTC)),
        )
        return 1

    async def _health_sweep(self) -> int:
        async with self._scope() as session:
            await session.execute(_ping_stmt())
        return 0

    def start(self) -> list[asyncio.Task[None]]:
        """Launch every loop and return the task handles."""
        s = self._settings
        jf = s.scheduler_jitter_fraction
        return [
            asyncio.create_task(
                run_periodic(
                    "retention",
                    s.scheduler_retention_interval,
                    self._retention,
                    stop=self._stop,
                    jitter_fraction=jf,
                )
            ),
            asyncio.create_task(
                run_periodic(
                    "evidence",
                    s.scheduler_evidence_interval,
                    self._evidence,
                    stop=self._stop,
                    jitter_fraction=jf,
                )
            ),
            asyncio.create_task(
                run_periodic(
                    "rollups",
                    s.scheduler_rollup_interval,
                    self._rollups,
                    stop=self._stop,
                    jitter_fraction=jf,
                )
            ),
            asyncio.create_task(
                run_periodic(
                    "index_rebuild",
                    s.scheduler_index_rebuild_interval,
                    self._index_rebuild,
                    stop=self._stop,
                    jitter_fraction=jf,
                )
            ),
            asyncio.create_task(
                run_periodic(
                    "health_sweep",
                    s.scheduler_health_interval,
                    self._health_sweep,
                    stop=self._stop,
                    jitter_fraction=jf,
                )
            ),
        ]

    def request_stop(self) -> None:
        """Signal all loops to finish their current sleep and exit."""
        self._stop.set()


def _ping_stmt() -> TextClause:
    from sqlalchemy import text

    return text("SELECT 1")


async def _noop_delete(_key: str) -> None:
    return None


async def _amain() -> None:  # pragma: no cover - runtime entrypoint
    settings = get_settings()
    configure_logging(level=settings.log_level, service_name="optimus-scheduler")

    bus, nc = await EventBus.connect(settings.nats_url)
    await bus.ensure_stream()

    engine = create_engine()
    factory = create_session_factory(engine)

    def scope() -> AbstractAsyncContextManager[AsyncSession]:
        return session_scope(factory)

    delete_object = _build_evidence_deleter(settings)
    service = SchedulerService(settings, bus, scope, delete_object=delete_object)

    health = HealthServer(host=settings.health_host, port=settings.health_port)
    health.add_readiness_check(nats_check(nc), name="nats")
    await health.start()

    handles = service.start()
    try:
        await asyncio.gather(*handles)
    finally:
        health.set_live(False)
        service.request_stop()
        for h in handles:
            h.cancel()
        with contextlib.suppress(Exception):
            await nc.drain()
        await health.stop()
        await engine.dispose()


def _build_evidence_deleter(  # pragma: no cover - optional backend
    settings: Settings,
) -> Callable[[str], Awaitable[None]] | None:
    if not settings.evidence_enabled:
        return None
    from optimus.evidence.store import S3ObjectStore

    store = S3ObjectStore(
        settings.evidence_bucket,
        endpoint_url=settings.evidence_endpoint_url,
        region=settings.evidence_region,
    )

    async def delete(key: str) -> None:
        await store.delete(key)

    return delete


def main() -> None:  # pragma: no cover
    """Console entrypoint: ``python -m optimus.services.scheduler``."""
    asyncio.run(_amain())


if __name__ == "__main__":  # pragma: no cover
    main()
