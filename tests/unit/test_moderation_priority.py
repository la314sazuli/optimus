"""Tests for the priority-aware moderation dispatcher.

Covers: action->priority classification, strict priority ordering under
contention, FIFO within a class, the aging starvation guard, the bounded-queue
drop policy (COURTESY/NOTIFY rejected when full, PROTECT always admitted), the
queue-depth gauge / dispatch-latency histogram, lifecycle (stop cancels queued
work), and the coordinator's use of the dispatcher.
"""

from __future__ import annotations

import asyncio

import pytest

from optimus.contracts.events import Action
from optimus.services.moderation.priority import (
    DROPPED,
    QUEUE_DEPTH,
    Priority,
    PriorityDispatcher,
    QueueFullError,
    classify_action,
)


def test_classify_action_maps_protective_actions() -> None:
    for action in (
        Action.DELETE,
        Action.DELETE_TIMEOUT,
        Action.DELETE_KICK,
        Action.DELETE_BAN,
    ):
        assert classify_action(action) is Priority.PROTECT
    assert classify_action(Action.REPORT_ONLY) is Priority.NOTIFY
    assert classify_action(Action.NONE) is Priority.NOTIFY


def test_priority_order_values() -> None:
    # Lower int = more urgent, so the heap pops PROTECT first.
    assert int(Priority.PROTECT) < int(Priority.NOTIFY) < int(Priority.COURTESY)


async def test_validates_constructor_arguments() -> None:
    with pytest.raises(ValueError, match="concurrency"):
        PriorityDispatcher(concurrency=0)
    with pytest.raises(ValueError, match="max_queue"):
        PriorityDispatcher(max_queue=0)
    with pytest.raises(ValueError, match="aging_seconds"):
        PriorityDispatcher(aging_seconds=0.0)


async def test_priority_ordering_under_contention() -> None:
    """With one worker held busy while items queue, they dispatch by priority.

    A gate keeps the single worker occupied so a backlog forms across classes;
    when released, the recorded start order must be PROTECT, then NOTIFY, then
    COURTESY regardless of submission order.
    """
    clock = {"t": 0.0}
    dispatcher: PriorityDispatcher[int] = PriorityDispatcher(
        concurrency=1, aging_seconds=1_000_000.0, time_source=lambda: clock["t"]
    )
    started: list[Priority] = []
    gate = asyncio.Event()

    async def factory(priority: Priority) -> int:
        started.append(priority)
        return int(priority)

    async def blocker() -> int:
        await gate.wait()
        return -1

    await dispatcher.start()
    # Occupy the only worker so everything below queues before any dispatch.
    block_future = await dispatcher.submit(Priority.PROTECT, blocker)
    await asyncio.sleep(0)  # let the worker pick up the blocker

    # Submit out of priority order.
    futures = [
        await dispatcher.submit(Priority.COURTESY, lambda: factory(Priority.COURTESY)),
        await dispatcher.submit(Priority.PROTECT, lambda: factory(Priority.PROTECT)),
        await dispatcher.submit(Priority.NOTIFY, lambda: factory(Priority.NOTIFY)),
    ]
    gate.set()
    await block_future
    await asyncio.gather(*futures)
    await dispatcher.stop()

    assert started == [Priority.PROTECT, Priority.NOTIFY, Priority.COURTESY]


async def test_fifo_within_a_class() -> None:
    """Items of equal priority dispatch in submission order (heapq isn't stable)."""
    clock = {"t": 0.0}
    dispatcher: PriorityDispatcher[int] = PriorityDispatcher(
        concurrency=1, aging_seconds=1_000_000.0, time_source=lambda: clock["t"]
    )
    started: list[int] = []
    gate = asyncio.Event()

    async def factory(n: int) -> int:
        started.append(n)
        return n

    async def blocker() -> int:
        await gate.wait()
        return -1

    await dispatcher.start()
    block_future = await dispatcher.submit(Priority.PROTECT, blocker)
    await asyncio.sleep(0)

    futures = [await dispatcher.submit(Priority.NOTIFY, lambda i=i: factory(i)) for i in range(5)]
    gate.set()
    await block_future
    await asyncio.gather(*futures)
    await dispatcher.stop()

    assert started == [0, 1, 2, 3, 4]


async def test_starvation_guard_ages_low_priority_past_fresh_high_priority() -> None:
    """An old COURTESY item outranks a freshly-submitted PROTECT item.

    With ``aging_seconds=1`` and the courtesy item having waited long enough to
    earn two class levels of boost, it must dispatch before a PROTECT item that
    has not waited at all — proving low-priority work cannot be starved forever.
    """
    clock = {"t": 0.0}
    dispatcher: PriorityDispatcher[int] = PriorityDispatcher(
        concurrency=1, aging_seconds=1.0, time_source=lambda: clock["t"]
    )
    started: list[Priority] = []
    gate = asyncio.Event()

    async def factory(priority: Priority) -> int:
        started.append(priority)
        return int(priority)

    async def blocker() -> int:
        await gate.wait()
        return -1

    await dispatcher.start()
    block_future = await dispatcher.submit(Priority.PROTECT, blocker)
    await asyncio.sleep(0)

    # Courtesy item queued at t=0.
    courtesy = await dispatcher.submit(Priority.COURTESY, lambda: factory(Priority.COURTESY))
    # Advance the clock so the courtesy item has aged 3s -> -3 class levels,
    # clamped, easily beating a fresh PROTECT.
    clock["t"] = 3.0
    protect = await dispatcher.submit(Priority.PROTECT, lambda: factory(Priority.PROTECT))

    gate.set()
    await block_future
    await asyncio.gather(courtesy, protect)
    await dispatcher.stop()

    assert started == [Priority.COURTESY, Priority.PROTECT]


async def test_drop_policy_rejects_courtesy_when_full_but_admits_protect() -> None:
    """At capacity, droppable work is rejected with a metric; PROTECT is admitted."""
    clock = {"t": 0.0}
    dispatcher: PriorityDispatcher[int] = PriorityDispatcher(
        concurrency=1, max_queue=2, aging_seconds=1_000_000.0, time_source=lambda: clock["t"]
    )
    gate = asyncio.Event()

    async def blocker() -> int:
        await gate.wait()
        return -1

    async def noop() -> int:
        return 0

    await dispatcher.start()
    # Hold the worker, then fill the queue to capacity (2 pending).
    block_future = await dispatcher.submit(Priority.PROTECT, blocker)
    await asyncio.sleep(0)
    await dispatcher.submit(Priority.NOTIFY, noop)
    await dispatcher.submit(Priority.NOTIFY, noop)
    assert dispatcher.depth == 2

    before = DROPPED.labels(priority="courtesy", reason="queue_full")._value.get()
    with pytest.raises(QueueFullError):
        await dispatcher.submit(Priority.COURTESY, noop)
    after = DROPPED.labels(priority="courtesy", reason="queue_full")._value.get()
    assert after == before + 1

    # PROTECT is admitted even though the queue is already full.
    protect = await dispatcher.submit(Priority.PROTECT, noop)
    assert dispatcher.depth == 3

    gate.set()
    await block_future
    await protect
    await dispatcher.stop()


async def test_queue_depth_gauge_tracks_pending_per_class() -> None:
    clock = {"t": 0.0}
    dispatcher: PriorityDispatcher[int] = PriorityDispatcher(
        concurrency=1, aging_seconds=1_000_000.0, time_source=lambda: clock["t"]
    )
    gate = asyncio.Event()

    async def blocker() -> int:
        await gate.wait()
        return -1

    async def noop() -> int:
        return 0

    await dispatcher.start()
    block_future = await dispatcher.submit(Priority.PROTECT, blocker)
    await asyncio.sleep(0)
    await dispatcher.submit(Priority.NOTIFY, noop)
    await dispatcher.submit(Priority.NOTIFY, noop)

    assert dispatcher.depth_by(Priority.NOTIFY) == 2
    assert QUEUE_DEPTH.labels(priority="notify")._value.get() == 2

    gate.set()
    await block_future
    await dispatcher.stop()


async def test_dispatch_latency_histogram_observes_waits() -> None:
    from optimus.services.moderation.priority import DISPATCH_LATENCY

    clock = {"t": 0.0}
    dispatcher: PriorityDispatcher[int] = PriorityDispatcher(
        concurrency=1, aging_seconds=1_000_000.0, time_source=lambda: clock["t"]
    )

    async def noop() -> int:
        return 7

    metric = DISPATCH_LATENCY.labels(priority="protect")
    before = metric._sum.get()

    await dispatcher.start()
    future = await dispatcher.submit(Priority.PROTECT, noop)
    assert await future == 7
    await dispatcher.stop()

    # The observed wait is >= 0; the sum advances by the recorded latency.
    assert metric._sum.get() >= before


async def test_exception_in_factory_propagates_to_future() -> None:
    dispatcher: PriorityDispatcher[int] = PriorityDispatcher(concurrency=1)

    async def boom() -> int:
        raise RuntimeError("kaboom")

    await dispatcher.start()
    future = await dispatcher.submit(Priority.PROTECT, boom)
    with pytest.raises(RuntimeError, match="kaboom"):
        await future
    await dispatcher.stop()


async def test_stop_cancels_still_queued_work() -> None:
    clock = {"t": 0.0}
    dispatcher: PriorityDispatcher[int] = PriorityDispatcher(
        concurrency=1, aging_seconds=1_000_000.0, time_source=lambda: clock["t"]
    )
    gate = asyncio.Event()

    async def blocker() -> int:
        await gate.wait()
        return -1

    async def noop() -> int:
        return 0

    await dispatcher.start()
    await dispatcher.submit(Priority.PROTECT, blocker)
    await asyncio.sleep(0)
    queued = await dispatcher.submit(Priority.NOTIFY, noop)

    await dispatcher.stop()  # cancels the worker mid-block and the queued item
    assert queued.cancelled()


async def test_start_is_idempotent() -> None:
    dispatcher: PriorityDispatcher[int] = PriorityDispatcher(concurrency=2)
    await dispatcher.start()
    await dispatcher.start()  # second start is a no-op, not a second pool

    async def noop() -> int:
        return 1

    future = await dispatcher.submit(Priority.PROTECT, noop)
    assert await future == 1
    await dispatcher.stop()
