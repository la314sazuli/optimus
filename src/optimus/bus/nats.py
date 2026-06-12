"""NATS JetStream helper: bounded streams, validated publish, pull-consumer loop.

Provides at-least-once delivery with explicit ack/nak. Streams are bounded with
a discard-old policy; dropped messages are counted via a Prometheus metric so
back-pressure is observable rather than silent.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, TypeVar

from prometheus_client import Counter, Gauge
from pydantic import BaseModel, ValidationError

from optimus.contracts.events import EVENT_SUBJECTS, STREAM_EVENTS
from optimus.core.logging import correlation_context, get_logger

if TYPE_CHECKING:
    from nats.aio.msg import Msg
    from nats.js import JetStreamContext

_log = get_logger(__name__)

E = TypeVar("E", bound=BaseModel)

MESSAGES_PUBLISHED = Counter(
    "optimus_bus_messages_published_total",
    "Messages published to the bus.",
    ["subject"],
)
MESSAGES_ACKED = Counter(
    "optimus_bus_messages_acked_total",
    "Messages acknowledged by a consumer.",
    ["subject"],
)
MESSAGES_NAKED = Counter(
    "optimus_bus_messages_naked_total",
    "Messages negatively-acknowledged (will be redelivered).",
    ["subject"],
)
MESSAGES_DROPPED = Counter(
    "optimus_bus_messages_dropped_total",
    "Messages dropped (undecodable / poison).",
    ["subject", "reason"],
)
MESSAGES_INFLIGHT = Gauge(
    "optimus_bus_messages_inflight",
    "Messages currently being processed by a consumer.",
    ["subject"],
)

# Default per-stream bound: keep up to this many messages, oldest discarded.
DEFAULT_MAX_MSGS = 1_000_000
# ...and this many bytes (1 GiB) before discarding oldest.
DEFAULT_MAX_BYTES = 1 * 1024 * 1024 * 1024
# JetStream server-side publish-dedup window. A redelivered/retried publish
# carrying the same ``Nats-Msg-Id`` within this window is collapsed to a single
# stored message, so a flaky publisher cannot fan a duplicate into every
# downstream consumer.
DEFAULT_DUPLICATE_WINDOW_SECONDS = 2 * 60 * 60
# Header NATS uses for server-side message deduplication.
NATS_MSG_ID_HEADER = "Nats-Msg-Id"

# Default consumer back-pressure knobs. ``ack_wait`` must comfortably exceed the
# slowest expected handler so slow processing buffers in JetStream rather than
# tripping a spurious redelivery storm; ``max_ack_pending`` caps how many
# messages the server will let a replica hold unacked at once (the real
# in-flight bound). Sized from the load-harness finding that a 2 vCPU replica
# saturates around ~10 img/s.
DEFAULT_ACK_WAIT_SECONDS = 60
DEFAULT_MAX_ACK_PENDING = 16


class EventBus:
    """Thin wrapper over a JetStream context for publishing and consuming events."""

    def __init__(self, js: JetStreamContext) -> None:
        self._js = js

    @classmethod
    async def connect(cls, url: str) -> tuple[EventBus, Any]:
        """Connect to NATS and return an :class:`EventBus` plus the raw client.

        The caller owns the returned client and is responsible for draining it.
        """
        import nats as _nats

        nc = await _nats.connect(url)
        js = nc.jetstream()
        return cls(js), nc

    async def ensure_stream(
        self,
        name: str = STREAM_EVENTS,
        subjects: tuple[str, ...] = EVENT_SUBJECTS,
        *,
        max_msgs: int = DEFAULT_MAX_MSGS,
        max_bytes: int = DEFAULT_MAX_BYTES,
        duplicate_window: float = DEFAULT_DUPLICATE_WINDOW_SECONDS,
    ) -> None:
        """Create or update a bounded stream with a discard-old retention policy.

        ``duplicate_window`` enables JetStream server-side publish dedup: a
        message published with a ``Nats-Msg-Id`` seen within the window is stored
        once, so a retried publish never fans a duplicate to consumers.
        """
        from nats.js.api import DiscardPolicy, RetentionPolicy, StreamConfig

        config = StreamConfig(
            name=name,
            subjects=list(subjects),
            retention=RetentionPolicy.LIMITS,
            discard=DiscardPolicy.OLD,
            max_msgs=max_msgs,
            max_bytes=max_bytes,
            duplicate_window=duplicate_window,
        )
        try:
            await self._js.update_stream(config=config)
        except Exception:
            await self._js.add_stream(config=config)

    async def publish(self, subject: str, event: BaseModel, *, msg_id: str | None = None) -> None:
        """Validate (by serialization) and publish ``event`` to ``subject``.

        When ``msg_id`` is given it is sent as the ``Nats-Msg-Id`` header so the
        server collapses duplicate publishes (e.g. a handler that re-runs after a
        redelivery) within the stream's duplicate window. The id is namespaced by
        subject so the same business key on two subjects never cross-dedups.
        """
        payload = event.model_dump_json().encode("utf-8")
        headers = {NATS_MSG_ID_HEADER: f"{subject}:{msg_id}"} if msg_id is not None else None
        await self._js.publish(subject, payload, headers=headers)
        MESSAGES_PUBLISHED.labels(subject=subject).inc()

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
        max_inflight: int = DEFAULT_MAX_ACK_PENDING,
        ack_wait: float = DEFAULT_ACK_WAIT_SECONDS,
        stop_event: asyncio.Event | None = None,
    ) -> None:
        """Run a pull-consumer loop with explicit ack/nak and poison-drop.

        Undecodable messages are terminated (dropped) and counted; handler
        failures are nak'd for redelivery up to ``max_deliver``.

        Back-pressure is bounded two ways. ``max_inflight`` caps concurrently
        processing messages per replica (also set as the consumer's
        ``max_ack_pending`` so the server stops handing out more), and the
        ``fetch`` batch is clamped to the spare in-flight budget so a slow
        replica leaves messages buffered in JetStream instead of pulling them
        into memory. ``ack_wait`` is how long the server waits for an ack before
        redelivering; it must exceed the slowest handler so slow processing does
        not trigger a spurious redelivery storm.
        """
        from nats.js.api import AckPolicy, ConsumerConfig

        max_inflight = max(1, max_inflight)
        sub = await self._js.pull_subscribe(
            subject,
            durable=durable,
            config=ConsumerConfig(
                durable_name=durable,
                ack_policy=AckPolicy.EXPLICIT,
                max_deliver=max_deliver,
                ack_wait=ack_wait,
                max_ack_pending=max_inflight,
            ),
        )

        sem = asyncio.Semaphore(max_inflight)
        tasks: set[asyncio.Task[None]] = set()
        while stop_event is None or not stop_event.is_set():
            # Only pull as many as we can currently process, so a slow replica
            # buffers in JetStream rather than ballooning its own memory.
            want = min(batch, max_inflight - len(tasks))
            if want <= 0:
                await asyncio.sleep(0)
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                tasks -= done
                continue
            try:
                msgs = await sub.fetch(want, timeout=fetch_timeout)
            except TimeoutError:
                continue
            except Exception:  # connection hiccup; let caller's supervisor restart
                _log.warning("bus_fetch_failed", subject=subject)
                await asyncio.sleep(0.5)
                continue
            for msg in msgs:
                await sem.acquire()
                task = asyncio.create_task(self._dispatch(subject, model, handler, msg, sem))
                tasks.add(task)
                task.add_done_callback(tasks.discard)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _dispatch(
        self,
        subject: str,
        model: type[E],
        handler: Callable[[E], Awaitable[None]],
        msg: Msg,
        sem: asyncio.Semaphore | None = None,
    ) -> None:
        try:
            try:
                event = model.model_validate_json(msg.data)
            except ValidationError:
                MESSAGES_DROPPED.labels(subject=subject, reason="decode").inc()
                _log.warning("bus_message_dropped", subject=subject, reason="decode")
                await msg.term()
                return

            MESSAGES_INFLIGHT.labels(subject=subject).inc()
            try:
                cid = getattr(event, "correlation_id", None)
                with correlation_context(cid):
                    try:
                        await handler(event)
                    except Exception:
                        MESSAGES_NAKED.labels(subject=subject).inc()
                        _log.exception("bus_handler_failed", subject=subject)
                        await msg.nak()
                        return
                MESSAGES_ACKED.labels(subject=subject).inc()
                await msg.ack()
            finally:
                MESSAGES_INFLIGHT.labels(subject=subject).dec()
        finally:
            if sem is not None:
                sem.release()
