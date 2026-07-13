"""In-process event bus: asyncio queues standing in for NATS JetStream.

This is the transport behind ``OPTIMUS_MODE=simple``. It satisfies the same
:class:`~optimus.bus.Bus` surface as :class:`~optimus.bus.nats.EventBus` but
keeps everything in one process: each ``(subject, durable)`` consumer owns a
bounded :class:`asyncio.Queue`, and :meth:`publish` fans an event out to every
durable subscribed to the subject. It deliberately mirrors three JetStream
behaviours the pipeline relies on:

* **Publish dedup** — a repeated ``Nats-Msg-Id`` (the ``msg_id`` argument) within
  a sliding window is dropped before fan-out, so a handler that re-runs after a
  redelivery cannot fan a duplicate to every consumer.
* **Bounded in-flight** — each consumer processes at most ``max_inflight``
  messages concurrently (reusing ``detection_max_inflight`` semantics); the queue
  is sized to the same bound so a slow handler applies back-pressure to the
  publisher rather than ballooning memory.
* **Redelivery + poison-drop** — a handler that raises is retried up to
  ``max_deliver`` times, then the message is dropped (counted), matching the
  JetStream nak/redeliver/term path.

What it intentionally does **not** do is persist anything: messages live only in
memory, so a restart loses any messages that were queued but not yet processed.
That is an accepted trade-off for the zero-dependency single-process mode; run
``OPTIMUS_MODE=distributed`` (NATS JetStream) when durability matters.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TypeVar

from pydantic import BaseModel

from optimus.bus.nats import (
    MESSAGES_ACKED,
    MESSAGES_DROPPED,
    MESSAGES_INFLIGHT,
    MESSAGES_NAKED,
    MESSAGES_PUBLISHED,
)
from optimus.core.logging import correlation_context, get_logger

_log = get_logger(__name__)

E = TypeVar("E", bound=BaseModel)

#: Default sliding dedup window (seconds), matching the JetStream default.
DEFAULT_DUPLICATE_WINDOW_SECONDS = 2 * 60 * 60


@dataclass
class _Delivery:
    """One enqueued message plus its delivery attempt count."""

    event: BaseModel
    deliveries: int = 0


@dataclass
class _Consumer:
    """A durable consumer bound to one subject."""

    subject: str
    durable: str
    model: type[BaseModel]
    handler: Callable[[BaseModel], Awaitable[None]]
    max_deliver: int
    max_inflight: int
    queue: asyncio.Queue[_Delivery] = field(init=False)

    def __post_init__(self) -> None:
        # Queue depth tracks the in-flight bound so a backed-up handler blocks the
        # publisher (back-pressure) instead of letting the queue grow unbounded.
        self.queue = asyncio.Queue(maxsize=max(1, self.max_inflight))


class InProcessBus:
    """Asyncio-queue event bus satisfying the :class:`~optimus.bus.Bus` protocol.

    Construct one instance and share it across every co-located service. Each
    :meth:`consume` call registers a durable consumer and runs its delivery loop
    until ``stop_event`` is set; :meth:`publish` enqueues to every consumer
    subscribed to the subject.
    """

    def __init__(
        self,
        *,
        duplicate_window: float = DEFAULT_DUPLICATE_WINDOW_SECONDS,
        time_source: Callable[[], float] = time.monotonic,
    ) -> None:
        self._consumers: dict[str, list[_Consumer]] = {}
        self._duplicate_window = duplicate_window
        self._now = time_source
        # subject -> {msg_id: expiry_monotonic}; pruned lazily on publish.
        self._seen: dict[str, dict[str, float]] = {}

    async def publish(self, subject: str, event: BaseModel, *, msg_id: str | None = None) -> None:
        """Validate (by serialization), dedup, and fan ``event`` out to consumers.

        Serializing up front mirrors the NATS bus, which publishes the JSON wire
        form; it also fails fast on an unserializable event rather than deep in a
        consumer. A repeated ``msg_id`` within the dedup window is dropped before
        fan-out so no consumer ever sees the duplicate.
        """
        event.model_dump_json()
        if msg_id is not None and self._is_duplicate(subject, msg_id):
            return
        MESSAGES_PUBLISHED.labels(subject=subject).inc()
        # Snapshot the consumer list: fan-out awaits on a full queue, and a
        # consumer whose loop exits during that await deregisters itself (mutating
        # the live list). Iterating a copy keeps the loop stable and avoids a
        # "list changed size during iteration" mid-fan-out.
        for consumer in list(self._consumers.get(subject, ())):
            await consumer.queue.put(_Delivery(event=event))

    def _is_duplicate(self, subject: str, msg_id: str) -> bool:
        """Whether ``msg_id`` was already seen on ``subject`` within the window."""
        now = self._now()
        seen = self._seen.setdefault(subject, {})
        # Prune expired ids so the map cannot grow without bound under churn.
        expired = [mid for mid, exp in seen.items() if exp <= now]
        for mid in expired:
            del seen[mid]
        if msg_id in seen:
            return True
        seen[msg_id] = now + self._duplicate_window
        return False

    def register(
        self,
        subject: str,
        durable: str,
        model: type[E],
        handler: Callable[[E], Awaitable[None]],
        *,
        max_deliver: int = 5,
        max_inflight: int = 16,
    ) -> _Consumer:
        """Synchronously register a durable consumer and return its handle.

        Registration is split out from :meth:`consume` so a caller that launches
        the delivery loop as a background task can publish immediately afterwards
        without racing the task's first scheduling: register first (sync), then
        ``create_task(run_consumer(...))``. :meth:`consume` itself registers when
        the caller has not done so already, preserving the one-shot await form.
        """
        consumer = _Consumer(
            subject=subject,
            durable=durable,
            model=model,
            handler=handler,  # type: ignore[arg-type]
            max_deliver=max_deliver,
            max_inflight=max(1, max_inflight),
        )
        self._consumers.setdefault(subject, []).append(consumer)
        return consumer

    def _unregister(self, consumer: _Consumer) -> None:
        """Remove a consumer from fan-out; idempotent and safe to call twice.

        Called when a consumer's delivery loop exits so :meth:`publish` stops
        enqueuing to a queue nothing will ever drain. Prunes the subject's list
        (and the subject key itself once empty) to keep the registry bounded
        across repeated consumer start/stop cycles on a shared bus.
        """
        consumers = self._consumers.get(consumer.subject)
        if not consumers:
            return
        try:
            consumers.remove(consumer)
        except ValueError:
            return
        if not consumers:
            del self._consumers[consumer.subject]

    def run(
        self,
        subject: str,
        durable: str,
        model: type[E],
        handler: Callable[[E], Awaitable[None]],
        *,
        max_deliver: int = 5,
        max_inflight: int = 16,
        stop_event: asyncio.Event | None = None,
    ) -> asyncio.Task[None]:
        """Register a consumer (sync) and launch its delivery loop as a task.

        This is the race-free way to start a consumer that may receive a publish
        immediately: the consumer is in ``self._consumers`` the instant this
        returns, before the awaitable loop is first scheduled. Returns the task so
        the caller can cancel/await it on shutdown.
        """
        consumer = self.register(
            subject, durable, model, handler, max_deliver=max_deliver, max_inflight=max_inflight
        )
        return asyncio.create_task(
            self.consume(
                subject,
                durable=durable,
                model=model,
                handler=handler,
                stop_event=stop_event,
                consumer=consumer,
            )
        )

    async def consume(
        self,
        subject: str,
        durable: str,
        model: type[E],
        handler: Callable[[E], Awaitable[None]],
        *,
        batch: int = 16,
        fetch_timeout: float = 5.0,
        max_deliver: int = 5,
        max_inflight: int = 16,
        ack_wait: float = 60.0,
        stop_event: asyncio.Event | None = None,
        consumer: _Consumer | None = None,
    ) -> None:
        """Register a durable consumer and run its delivery loop until stopped.

        ``batch``, ``fetch_timeout`` and ``ack_wait`` exist only to match the
        :class:`~optimus.bus.Bus` signature; the in-process loop has no fetch
        round-trip, so they are inert here. ``max_inflight`` bounds concurrent
        handler execution and ``max_deliver`` bounds redelivery, exactly as in the
        JetStream consumer.

        Pass a ``consumer`` previously obtained from :meth:`register` to run a
        loop for an already-registered consumer (avoiding a registration race when
        the loop is launched as a task); otherwise one is registered here.
        """
        if consumer is None:
            consumer = self.register(
                subject,
                durable,
                model,
                handler,
                max_deliver=max_deliver,
                max_inflight=max_inflight,
            )

        sem = asyncio.Semaphore(consumer.max_inflight)
        tasks: set[asyncio.Task[None]] = set()
        try:
            while stop_event is None or not stop_event.is_set():
                try:
                    delivery = await asyncio.wait_for(consumer.queue.get(), timeout=0.05)
                except TimeoutError:
                    continue
                await sem.acquire()
                task = asyncio.create_task(self._dispatch(consumer, delivery, sem))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
        finally:
            # Deregister before draining so a concurrent publish stops fanning out
            # to this (now-stopping) consumer's queue. Otherwise the consumer stays
            # in self._consumers after its loop exits, leaking the entry and — once
            # its bounded queue fills — blocking publish() forever on a dead queue.
            self._unregister(consumer)
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    async def _dispatch(
        self, consumer: _Consumer, delivery: _Delivery, sem: asyncio.Semaphore
    ) -> None:
        """Run the handler for one delivery, redelivering or dropping on failure.

        The in-flight permit is released *before* any redelivery ``queue.put``.
        The queue is sized to ``max_inflight``, so a requeue can block when the
        queue is full; holding the permit across that put would let every worker
        block on put while the consume loop is stuck on ``sem.acquire()`` and
        cannot drain the queue — a deadlock. The waiting message is not in flight,
        so it must not occupy a permit.
        """
        try:
            requeue = await self._run_handler(consumer, delivery)
        finally:
            sem.release()
        if requeue:
            await consumer.queue.put(delivery)

    async def _run_handler(self, consumer: _Consumer, delivery: _Delivery) -> bool:
        """Invoke the handler once; return whether the delivery should be requeued."""
        delivery.deliveries += 1
        subject = consumer.subject
        MESSAGES_INFLIGHT.labels(subject=subject).inc()
        try:
            cid = getattr(delivery.event, "correlation_id", None)
            with correlation_context(cid):
                try:
                    await consumer.handler(delivery.event)
                except Exception:
                    return self._on_handler_failure(consumer, delivery)
            MESSAGES_ACKED.labels(subject=subject).inc()
            return False
        finally:
            MESSAGES_INFLIGHT.labels(subject=subject).dec()

    def _on_handler_failure(self, consumer: _Consumer, delivery: _Delivery) -> bool:
        """Record a failed delivery; return True to redeliver, False once dropped."""
        subject = consumer.subject
        MESSAGES_NAKED.labels(subject=subject).inc()
        if delivery.deliveries >= consumer.max_deliver:
            MESSAGES_DROPPED.labels(subject=subject, reason="max_deliver").inc()
            _log.warning(
                "bus_message_dropped",
                subject=subject,
                reason="max_deliver",
                deliveries=delivery.deliveries,
            )
            return False
        _log.warning("bus_handler_failed", subject=subject, deliveries=delivery.deliveries)
        return True
