"""Priority-aware dispatch for moderation work under REST rate-limit pressure.

During a raid, Discord REST rate limits and the per-guild token bucket throttle
the action path. Without prioritization, a protective action (delete a scam
message, timeout/ban the attacker) can queue behind low-urgency work (appeal
DMs, audit messages, notifications) and arrive too late to matter.

:class:`PriorityDispatcher` is a single-replica, in-process scheduler that runs
submitted coroutines under a bounded concurrency budget, always picking the
highest-priority ready item first and preserving FIFO order within a class.

Three properties make it safe to drop in front of the existing executor:

* **Priority + FIFO.** An :class:`asyncio.PriorityQueue` orders items by
  ``(effective_priority, sequence)``. ``sequence`` is a monotonic counter so
  ties within a class drain first-in-first-out (heapq is not stable on its own).
* **Starvation guard via aging.** A purely strict priority order would let a
  sustained flood of PROTECT work starve NOTIFY/COURTESY indefinitely. Each
  item's *effective* priority improves the longer it waits (one class level per
  ``aging_seconds``), so an old COURTESY item eventually outranks a fresh
  PROTECT one and low-priority work always drains. Aging (rather than a reserved
  worker share) keeps the common case — strict priority — exact, and only bends
  it under sustained pressure, which is precisely when fairness matters.
* **Bounded queue with a fail-safe drop policy.** The queue is capacity-capped.
  When full, COURTESY (and then NOTIFY) submissions are rejected with a
  reason-labeled metric rather than growing memory without bound; **PROTECT is
  never dropped** — protective work is admitted even past the cap, because
  losing it is worse than a transient overshoot.

Multi-replica note: each replica runs its own dispatcher over the work *it*
consumes from JetStream. There is no cross-replica priority coordination and
none is needed — JetStream already balances delivery across replicas, and every
replica independently prioritizing its own queue means the fleet as a whole
favours protective work without any shared state or distributed lock. The Redis
token bucket (see :class:`~optimus.core.ratelimit.RedisRateLimiter`) remains the
only cross-replica throttle; this dispatcher only orders a single replica's
already-admitted work.
"""

from __future__ import annotations

import asyncio
import contextlib
import heapq
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import IntEnum

from prometheus_client import Counter, Gauge, Histogram

from optimus.contracts.events import Action
from optimus.core.logging import get_logger

_log = get_logger(__name__)


class Priority(IntEnum):
    """Dispatch priority classes (lower value = dispatched first).

    ``IntEnum`` so values order directly in the priority queue and aging
    arithmetic (a lower number is more urgent, matching ``heapq`` semantics).
    """

    PROTECT = 0
    NOTIFY = 1
    COURTESY = 2

    @property
    def label(self) -> str:
        """Lowercase metric-label form (``IntEnum.name`` is uppercase)."""
        return self.name.lower()


#: Actions that remove a scam message or punish the attacker. These are the
#: time-critical, protective actions that must never queue behind courtesy work
#: and must never be dropped under load.
_PROTECT_ACTIONS = frozenset(
    {
        Action.DELETE,
        Action.DELETE_TIMEOUT,
        Action.DELETE_KICK,
        Action.DELETE_BAN,
    }
)


def classify_action(action: Action) -> Priority:
    """Map a moderation :class:`Action` to its dispatch :class:`Priority`.

    Protective enforcement is PROTECT; everything else (report-only posting,
    no-op outcomes) is NOTIFY. COURTESY is reserved for explicitly courtesy work
    (e.g. appeal DMs) submitted with that class directly.
    """
    if action in _PROTECT_ACTIONS:
        return Priority.PROTECT
    return Priority.NOTIFY


QUEUE_DEPTH = Gauge(
    "optimus_moderation_priority_queue_depth",
    "Pending moderation dispatch items by priority class.",
    ["priority"],
)
DISPATCH_LATENCY = Histogram(
    "optimus_moderation_dispatch_latency_seconds",
    "Time from submission to start of dispatch, by priority class.",
    ["priority"],
    # Raid-scale waits matter most in the sub-second-to-tens-of-seconds band.
    buckets=(0.005, 0.025, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)
DROPPED = Counter(
    "optimus_moderation_priority_dropped_total",
    "Moderation dispatch submissions rejected, by priority class and reason.",
    ["priority", "reason"],
)


class QueueFullError(RuntimeError):
    """Raised when a submission is rejected because the queue is at capacity."""

    def __init__(self, priority: Priority) -> None:
        super().__init__(f"priority queue full; rejected {priority.name}")
        self.priority = priority


@dataclass(order=True)
class _Item[T]:
    """A queued unit of work, ordered by ``(effective priority, sequence)``.

    Only the sort keys participate in ordering; the coroutine factory, future,
    and bookkeeping fields are excluded so two items never compare by payload.
    """

    sort_key: tuple[int, int]
    priority: Priority = field(compare=False)
    sequence: int = field(compare=False)
    enqueued_at: float = field(compare=False)
    factory: Callable[[], Awaitable[T]] = field(compare=False)
    future: asyncio.Future[T] = field(compare=False)


class PriorityDispatcher[T]:
    """Runs submitted coroutines newest-highest-priority-first under a budget.

    ``concurrency`` worker tasks pull from an in-process priority heap; each
    item is ``(effective_priority, sequence)`` ordered. ``max_queue`` bounds the
    pending heap (PROTECT is admitted past the cap; see module docstring).
    ``aging_seconds`` is the wait time that buys one class level of priority
    boost — the starvation guard.
    """

    def __init__(
        self,
        *,
        concurrency: int = 4,
        max_queue: int = 1000,
        aging_seconds: float = 5.0,
        time_source: Callable[[], float] = time.monotonic,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        if max_queue < 1:
            raise ValueError("max_queue must be >= 1")
        if aging_seconds <= 0:
            raise ValueError("aging_seconds must be positive")
        self._concurrency = concurrency
        self._max_queue = max_queue
        self._aging = aging_seconds
        self._now = time_source
        self._heap: list[_Item[T]] = []
        self._seq = 0
        self._not_empty = asyncio.Condition()
        self._workers: list[asyncio.Task[None]] = []
        self._running = False
        self._depth = dict.fromkeys(Priority, 0)

    def _effective_priority(self, priority: Priority, enqueued_at: float) -> int:
        """Priority adjusted for how long the item has waited (aging guard).

        Each ``aging_seconds`` of waiting subtracts one class level, so a long-
        waiting low-priority item climbs toward (and past) fresh high-priority
        work. Clamped at PROTECT so aging never invents a class more urgent than
        the most urgent real one.
        """
        waited = max(0.0, self._now() - enqueued_at)
        boost = int(waited / self._aging)
        return max(int(Priority.PROTECT), int(priority) - boost)

    async def submit(
        self, priority: Priority, factory: Callable[[], Awaitable[T]]
    ) -> asyncio.Future[T]:
        """Enqueue ``factory`` at ``priority``; return a future for its result.

        ``factory`` is a zero-arg callable returning the coroutine to run, so the
        coroutine is only created when a worker is ready to await it (no
        un-awaited-coroutine warnings while queued). Raises
        :class:`QueueFullError` if the queue is full and ``priority`` is droppable.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[T] = loop.create_future()
        async with self._not_empty:
            if len(self._heap) >= self._max_queue and priority is not Priority.PROTECT:
                DROPPED.labels(priority=priority.label, reason="queue_full").inc()
                _log.warning(
                    "moderation_dispatch_dropped",
                    priority=priority.name,
                    reason="queue_full",
                    depth=len(self._heap),
                )
                raise QueueFullError(priority)
            now = self._now()
            self._seq += 1
            item = _Item(
                sort_key=(int(priority), self._seq),
                priority=priority,
                sequence=self._seq,
                enqueued_at=now,
                factory=factory,
                future=future,
            )
            heapq.heappush(self._heap, item)
            self._depth[priority] += 1
            QUEUE_DEPTH.labels(priority=priority.label).set(self._depth[priority])
            self._not_empty.notify()
        return future

    def _rescore(self) -> None:
        """Recompute aging-adjusted sort keys and re-heapify in place.

        Called under the condition lock before popping so the chosen item
        reflects current wait times. O(n) but n is the bounded pending set.
        """
        for item in self._heap:
            item.sort_key = (
                self._effective_priority(item.priority, item.enqueued_at),
                item.sequence,
            )
        heapq.heapify(self._heap)

    async def _take(self) -> _Item[T]:
        async with self._not_empty:
            while not self._heap:
                await self._not_empty.wait()
            self._rescore()
            item = heapq.heappop(self._heap)
            self._depth[item.priority] -= 1
            QUEUE_DEPTH.labels(priority=item.priority.label).set(self._depth[item.priority])
            return item

    async def _worker(self) -> None:
        while self._running:
            try:
                item = await self._take()
            except asyncio.CancelledError:
                return
            waited = max(0.0, self._now() - item.enqueued_at)
            DISPATCH_LATENCY.labels(priority=item.priority.label).observe(waited)
            try:
                result = await item.factory()
            except asyncio.CancelledError:
                if not item.future.done():
                    item.future.cancel()
                return
            except Exception as exc:
                if not item.future.done():
                    item.future.set_exception(exc)
            else:
                if not item.future.done():
                    item.future.set_result(result)

    async def start(self) -> None:
        """Spin up the worker pool."""
        if self._running:
            return
        self._running = True
        self._workers = [
            asyncio.create_task(self._worker(), name=f"mod-dispatch-{i}")
            for i in range(self._concurrency)
        ]

    async def stop(self) -> None:
        """Cancel workers and fail any still-queued futures."""
        self._running = False
        for task in self._workers:
            task.cancel()
        for task in self._workers:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._workers = []
        async with self._not_empty:
            while self._heap:
                item = heapq.heappop(self._heap)
                self._depth[item.priority] -= 1
                QUEUE_DEPTH.labels(priority=item.priority.label).set(self._depth[item.priority])
                if not item.future.done():
                    item.future.cancel()

    @property
    def depth(self) -> int:
        """Total pending items across all priority classes."""
        return len(self._heap)

    def depth_by(self, priority: Priority) -> int:
        """Pending items in one priority class."""
        return self._depth[priority]
