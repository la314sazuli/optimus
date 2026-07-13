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

from optimus.contracts.events import (
    EVENT_SUBJECTS,
    STREAM_EVENTS,
    SUBJECT_DEAD_LETTER,
    DeadLetterEvent,
)
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
MESSAGES_DEAD_LETTERED = Counter(
    "optimus_bus_messages_dead_lettered_total",
    "Messages routed to the dead-letter subject after exhausting max_deliver.",
    ["subject"],
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


# A base64-encoded image inflates to ceil(n/3)*4 ~= 4/3 the raw bytes; the
# surrounding JSON envelope (ids, sha256, content-type, timestamps) and the
# ``Nats-Msg-Id`` header add a small fixed overhead. This factor + slack is the
# worst-case wire size a raw inline image of ``n`` bytes occupies on NATS, used
# to validate the configured inline cap against the server's ``max_payload``.
_BASE64_INFLATION = 4 / 3
#: Fixed bytes reserved for the JSON envelope and NATS headers around the
#: base64 image field. Generous: the real envelope is a few hundred bytes.
_ENVELOPE_OVERHEAD_BYTES = 4096


def _num_delivered(msg: Msg) -> int | None:
    """Delivery count for ``msg`` from JetStream metadata, or ``None`` if absent.

    A ``None`` (e.g. a non-JetStream message, or metadata that cannot be parsed)
    means we cannot tell it is exhausted, so the caller nak's rather than
    dead-letters — never dropping a message just because the count is unknown.
    """
    try:
        return int(msg.metadata.num_delivered)
    except Exception:
        return None


def inline_wire_size(raw_bytes: int) -> int:
    """Worst-case NATS wire size for an inline image of ``raw_bytes`` raw bytes.

    Accounts for base64 inflation of the image field plus a fixed allowance for
    the surrounding JSON envelope and the ``Nats-Msg-Id`` header. Used to check
    that ``ingest_max_inline_bytes`` actually fits the server's ``max_payload``
    before publishing, rather than discovering it as a runtime publish failure.
    """
    return int(raw_bytes * _BASE64_INFLATION) + _ENVELOPE_OVERHEAD_BYTES


class PayloadLimitError(RuntimeError):
    """Raised when the configured inline image cap exceeds the NATS max_payload.

    Failing fast at startup is far better than the silent alternative: an
    in-bounds image whose base64 wire form exceeds ``max_payload`` raises
    ``MaxPayloadError`` deep in the publish path, is nak'd and redelivered until
    ``max_deliver`` is exhausted, then dropped — so the image is never scanned
    and nothing surfaces beyond a redelivery/drop metric.
    """


class EventBus:
    """Thin wrapper over a JetStream context for publishing and consuming events."""

    def __init__(self, js: JetStreamContext, *, max_payload: int | None = None) -> None:
        self._js = js
        self._max_payload = max_payload

    @classmethod
    async def connect(cls, url: str) -> tuple[EventBus, Any]:
        """Connect to NATS and return an :class:`EventBus` plus the raw client.

        The caller owns the returned client and is responsible for draining it.
        The server-negotiated ``max_payload`` (from the connection INFO) is
        captured so the publish-size budget can be validated up front.

        ``max_reconnect_attempts=-1`` makes reconnection retry forever. The
        nats-py default (60 attempts x 2s) gives up permanently after ~2 minutes
        of downtime: a NATS outage or restart longer than that would leave a
        consumer disconnected for good — readiness stuck at 503, no processing,
        no recovery without a process restart. A long-lived service must ride out
        an arbitrarily long broker outage and resume on its own, so we retry
        indefinitely (paced by the 2s default ``reconnect_time_wait``).
        """
        import nats as _nats

        nc = await _nats.connect(url, max_reconnect_attempts=-1)
        js = nc.jetstream()
        return cls(js, max_payload=nc.max_payload), nc

    @property
    def max_payload(self) -> int | None:
        """Server-negotiated max publish size in bytes (``None`` if unknown)."""
        return self._max_payload

    def validate_inline_capacity(self, max_inline_bytes: int) -> None:
        """Ensure an inline image at the configured cap fits the server max_payload.

        Raises :class:`PayloadLimitError` if the worst-case wire size of an image
        at ``max_inline_bytes`` would exceed the connected server's
        ``max_payload``. Self-hosters must either raise the NATS ``max_payload``
        (see ``docker-compose.yml``) or lower ``OPTIMUS_INGEST_MAX_INLINE_BYTES``.
        A ``None`` ``max_payload`` (server INFO not seen) skips the check.
        """
        if self._max_payload is None:
            return
        needed = inline_wire_size(max_inline_bytes)
        if needed > self._max_payload:
            raise PayloadLimitError(
                "ingest_max_inline_bytes is too large for the NATS server's "
                f"max_payload: an image of {max_inline_bytes:,} raw bytes needs "
                f"~{needed:,} wire bytes (base64 + envelope) but the server "
                f"accepts only {self._max_payload:,}. Raise the NATS server "
                "max_payload (docker-compose.yml passes --max_payload) to at "
                f"least {needed:,}, or lower OPTIMUS_INGEST_MAX_INLINE_BYTES."
            )

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
        await self.publish_raw(subject, event.model_dump_json().encode("utf-8"), msg_id=msg_id)

    async def publish_raw(self, subject: str, payload: bytes, msg_id: str | None = None) -> None:
        """Publish already-serialized ``payload`` bytes to ``subject``.

        The relay uses this to re-emit an outbox row's stored JSON without
        re-instantiating its model; the ``Nats-Msg-Id`` dedup semantics match
        :meth:`publish`.
        """
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
                task = asyncio.create_task(
                    self._dispatch(subject, model, handler, msg, sem, max_deliver=max_deliver)
                )
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
        *,
        max_deliver: int = 5,
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
                    except Exception as exc:
                        await self._on_handler_failure(subject, msg, max_deliver, exc)
                        return
                MESSAGES_ACKED.labels(subject=subject).inc()
                await msg.ack()
            finally:
                MESSAGES_INFLIGHT.labels(subject=subject).dec()
        finally:
            if sem is not None:
                sem.release()

    async def _on_handler_failure(
        self, subject: str, msg: Msg, max_deliver: int, exc: Exception
    ) -> None:
        """Nak for redelivery, or dead-letter once ``max_deliver`` is exhausted.

        Without this, a message that fails ``max_deliver`` times is dropped by
        JetStream with nothing but a generic nak counter to show for it. On the
        final attempt we route the raw payload to :data:`SUBJECT_DEAD_LETTER`,
        count it, and log an error so the loss is captured and alertable rather
        than silent. If the dead-letter publish itself fails we fall back to a
        nak so the message is not lost on our account.
        """
        delivered = _num_delivered(msg)
        if delivered is not None and delivered >= max_deliver:
            try:
                await self._dead_letter(subject, msg, delivered, exc)
            except Exception:
                MESSAGES_NAKED.labels(subject=subject).inc()
                _log.exception("bus_dead_letter_failed", subject=subject)
                await msg.nak()
                return
            MESSAGES_DEAD_LETTERED.labels(subject=subject).inc()
            _log.error(
                "bus_message_dead_lettered",
                subject=subject,
                delivered=delivered,
                error=type(exc).__name__,
            )
            await msg.term()
            return
        MESSAGES_NAKED.labels(subject=subject).inc()
        _log.exception("bus_handler_failed", subject=subject)
        await msg.nak()

    async def _dead_letter(self, subject: str, msg: Msg, delivered: int, exc: Exception) -> None:
        from datetime import UTC, datetime

        event = DeadLetterEvent(
            correlation_id="dead-letter",
            occurred_at=datetime.now(UTC),
            original_subject=subject,
            payload=msg.data.decode("utf-8", errors="replace"),
            delivered=delivered,
            error=type(exc).__name__,
        )
        await self.publish(SUBJECT_DEAD_LETTER, event)
