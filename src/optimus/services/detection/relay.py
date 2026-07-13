"""Transactional-outbox relay: drain staged rows onto the bus, at-least-once.

Detection persists its ``Detection`` row and the resulting bus messages into the
``outbox`` table in one transaction (see :class:`~optimus.db.models.OutboxEvent`).
This relay publishes those rows and marks them published. Publish carries the
stored ``msg_id`` so JetStream's server-side dedup collapses a row re-published
after a crash/retry, making re-emission safe.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from optimus.core.logging import get_logger
from optimus.db.engine import SessionScope
from optimus.db.repositories import OutboxRepository

_log = get_logger(__name__)

#: Publishes an outbox row: ``(subject, payload_bytes, msg_id)``.
PublishRaw = Callable[[str, bytes, str | None], Awaitable[None]]


class OutboxRelay:
    """Polls the outbox and publishes unpublished rows in id order."""

    def __init__(
        self,
        scope: SessionScope,
        publish: PublishRaw,
        *,
        batch: int = 100,
        poll_seconds: float = 0.5,
    ) -> None:
        self._scope = scope
        self._publish = publish
        self._batch = batch
        self._poll_seconds = poll_seconds

    async def drain_once(self) -> int:
        """Publish one batch of unpublished rows; return how many were published.

        A publish failure stops the batch without raising: rows published so far
        are marked (and committed on scope exit), and the failing row plus the
        rest are retried on the next poll, preserving order.
        """
        async with self._scope() as session:
            repo = OutboxRepository(session)
            rows = await repo.fetch_unpublished(self._batch)
            if not rows:
                return 0
            published: list[int] = []
            for row in rows:
                try:
                    await self._publish(row.subject, row.payload.encode("utf-8"), row.msg_id)
                except Exception:
                    _log.exception("outbox_publish_failed", outbox_id=row.id, subject=row.subject)
                    break
                published.append(row.id)
            await repo.mark_published(published, now=datetime.now(UTC))
            return len(published)

    async def run(self, stop: asyncio.Event) -> None:
        """Drain in a loop until ``stop`` is set, sleeping when the outbox is idle."""
        while not stop.is_set():
            try:
                count = await self.drain_once()
            except Exception:
                _log.exception("outbox_relay_failed")
                count = 0
            # A full batch likely means more is waiting; loop again immediately.
            if count < self._batch:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(stop.wait(), timeout=self._poll_seconds)
