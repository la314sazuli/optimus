"""Behavioural tests for :class:`~optimus.bus.inprocess.InProcessBus`.

These prove the in-process bus mirrors the three JetStream behaviours simple
mode depends on, without any NATS:

* **publish dedup** — a repeated ``msg_id`` within the window is dropped before
  fan-out (and expires from the window on an injected clock);
* **fan-out** — one publish reaches every durable subscribed to the subject;
* **bounded in-flight** — a slow handler never runs more than ``max_inflight``
  concurrently even under a burst;
* **redelivery + poison-drop** — a handler that raises is retried up to
  ``max_deliver`` and then dropped, while a success is delivered exactly once.

The consume loop runs as a background task driven by a ``stop_event``; each test
publishes, waits for the expected handler calls, then stops the loop.
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel
from pydantic_core import PydanticSerializationError

from optimus.bus.inprocess import InProcessBus


class _Evt(BaseModel):
    correlation_id: str = "c"
    n: int = 0


async def _run_until(predicate, within: float = 5.0) -> None:  # type: ignore[no-untyped-def]
    """Poll ``predicate`` until true or ``within`` seconds elapse.

    The budget is deliberately generous: these tests share the loop with the
    whole suite, and the consumer drains its queue on a 50 ms poll, so on a
    loaded CI host the first delivery can legitimately lag a second or more. A
    tight budget here produced a rare ``condition not met within timeout`` flake
    (handler simply hadn't been scheduled yet, not a real hang). The wait is a
    *liveness* guard against a genuine stall, not a latency assertion, so a wide
    bound trades nothing for determinism.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + within
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition not met within timeout")


async def _consume_in_task(bus: InProcessBus, stop: asyncio.Event, **kwargs) -> asyncio.Task[None]:  # type: ignore[no-untyped-def]
    task = asyncio.create_task(bus.consume(stop_event=stop, **kwargs))
    # Yield so the consumer registers before the first publish.
    await asyncio.sleep(0)
    return task


async def _drain(stop: asyncio.Event, task: asyncio.Task[None]) -> None:
    stop.set()
    # Wide bound: the consume loop notices the stop on its 50 ms poll and then
    # drains in-flight handlers, which under suite-wide load can take a beat.
    await asyncio.wait_for(task, timeout=5.0)


class _Clock:
    """A manually advanced monotonic clock for dedup-window assertions."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


async def test_publish_fans_out_to_every_consumer() -> None:
    bus = InProcessBus()
    a: list[int] = []
    b: list[int] = []

    async def handler_a(evt: _Evt) -> None:
        a.append(evt.n)

    async def handler_b(evt: _Evt) -> None:
        b.append(evt.n)

    stop = asyncio.Event()
    t1 = await _consume_in_task(bus, stop, subject="s", durable="a", model=_Evt, handler=handler_a)
    t2 = await _consume_in_task(bus, stop, subject="s", durable="b", model=_Evt, handler=handler_b)

    await bus.publish("s", _Evt(n=1))
    await _run_until(lambda: a == [1] and b == [1])

    await _drain(stop, t1)
    await _drain(stop, t2)


async def test_duplicate_msg_id_dropped_within_window() -> None:
    clock = _Clock()
    bus = InProcessBus(duplicate_window=100.0, time_source=clock)
    seen: list[int] = []

    async def handler(evt: _Evt) -> None:
        seen.append(evt.n)

    stop = asyncio.Event()
    task = await _consume_in_task(bus, stop, subject="s", durable="d", model=_Evt, handler=handler)

    await bus.publish("s", _Evt(n=1), msg_id="dup")
    await bus.publish("s", _Evt(n=2), msg_id="dup")  # dropped: same id, same window
    await _run_until(lambda: seen == [1])
    # Give a duplicate a chance to (wrongly) arrive before asserting it did not.
    await asyncio.sleep(0.02)
    assert seen == [1]

    # Past the window the same id is accepted again.
    clock.t = 200.0
    await bus.publish("s", _Evt(n=3), msg_id="dup")
    await _run_until(lambda: seen == [1, 3])

    await _drain(stop, task)


async def test_distinct_msg_ids_both_delivered() -> None:
    bus = InProcessBus()
    seen: list[int] = []

    async def handler(evt: _Evt) -> None:
        seen.append(evt.n)

    stop = asyncio.Event()
    task = await _consume_in_task(bus, stop, subject="s", durable="d", model=_Evt, handler=handler)

    await bus.publish("s", _Evt(n=1), msg_id="a")
    await bus.publish("s", _Evt(n=2), msg_id="b")
    await _run_until(lambda: sorted(seen) == [1, 2])

    await _drain(stop, task)


async def test_inflight_bounded_by_max_inflight() -> None:
    bus = InProcessBus()
    concurrent = 0
    peak = 0
    release = asyncio.Event()

    async def handler(_evt: _Evt) -> None:
        nonlocal concurrent, peak
        concurrent += 1
        peak = max(peak, concurrent)
        await release.wait()
        concurrent -= 1

    stop = asyncio.Event()
    task = await _consume_in_task(
        bus, stop, subject="s", durable="d", model=_Evt, handler=handler, max_inflight=2
    )

    # Publish a burst larger than the bound. The queue is sized to the bound, so
    # the extra publishes block until a slot frees — start them concurrently.
    publishers = [asyncio.create_task(bus.publish("s", _Evt(n=i))) for i in range(6)]
    await _run_until(lambda: concurrent == 2)
    await asyncio.sleep(0.02)
    assert peak == 2  # never more than max_inflight ran at once

    release.set()
    await asyncio.gather(*publishers)
    await _drain(stop, task)
    assert peak == 2


async def test_handler_failure_redelivers_then_drops() -> None:
    bus = InProcessBus()
    attempts = 0

    async def handler(_evt: _Evt) -> None:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("boom")

    stop = asyncio.Event()
    task = await _consume_in_task(
        bus, stop, subject="s", durable="d", model=_Evt, handler=handler, max_deliver=3
    )

    await bus.publish("s", _Evt(n=1))
    await _run_until(lambda: attempts == 3)
    # After max_deliver the message is dropped — no further attempts.
    await asyncio.sleep(0.05)
    assert attempts == 3

    await _drain(stop, task)


async def test_successful_handler_delivered_once() -> None:
    bus = InProcessBus()
    count = 0

    async def handler(_evt: _Evt) -> None:
        nonlocal count
        count += 1

    stop = asyncio.Event()
    task = await _consume_in_task(bus, stop, subject="s", durable="d", model=_Evt, handler=handler)

    await bus.publish("s", _Evt(n=1))
    await _run_until(lambda: count == 1)
    await asyncio.sleep(0.02)
    assert count == 1

    await _drain(stop, task)


async def test_unserializable_event_rejected_at_publish() -> None:
    bus = InProcessBus()

    class _Bad(BaseModel):
        value: object = object()  # not JSON-serializable

    with pytest.raises(PydanticSerializationError):
        await bus.publish("s", _Bad())


# --- consumer deregistration on shutdown (leak / dead-queue hazard) ----------


async def test_consumer_deregisters_when_loop_exits() -> None:
    """A stopped consumer leaves the registry, so fan-out forgets it.

    Regression: ``consume`` never removed its consumer on exit, so each stopped
    consumer stayed in ``_consumers`` forever — a leak on a shared/long-lived
    bus, and worse, ``publish`` kept fanning out to its bounded queue.
    """
    bus = InProcessBus()

    async def handler(_evt: _Evt) -> None:
        return None

    stop = asyncio.Event()
    task = bus.run("s", durable="d", model=_Evt, handler=handler, stop_event=stop)
    await asyncio.sleep(0)
    assert len(bus._consumers.get("s", [])) == 1
    await _drain(stop, task)
    # Once the loop exits the subject is fully pruned, not left holding a corpse.
    assert "s" not in bus._consumers


async def test_repeated_start_stop_does_not_leak_consumers() -> None:
    """Many consumer lifecycles on one shared bus leave no registry residue."""
    bus = InProcessBus()

    async def handler(_evt: _Evt) -> None:
        return None

    for _ in range(20):
        stop = asyncio.Event()
        task = bus.run("s", durable="d", model=_Evt, handler=handler, stop_event=stop)
        await asyncio.sleep(0)
        await _drain(stop, task)

    assert bus._consumers == {}


async def test_publish_after_consumer_stop_does_not_block_publisher() -> None:
    """A publish after a consumer stops must not wedge on the dead queue.

    Drives the exact hazard the leak created: a consumer with a 1-slot queue
    whose handler blocks, then stops while a delivery is parked; once it is
    deregistered, later publishes to the subject return immediately (no live
    consumer, nothing to fan out to) instead of blocking forever on the full,
    abandoned queue.
    """
    bus = InProcessBus()
    release = asyncio.Event()

    async def blocked(_evt: _Evt) -> None:
        await release.wait()

    stop = asyncio.Event()
    task = bus.run("s", durable="d", model=_Evt, handler=blocked, max_inflight=1, stop_event=stop)
    await asyncio.sleep(0)
    await bus.publish("s", _Evt(n=1))  # taken in-flight, blocks on release
    await asyncio.sleep(0.02)

    release.set()
    await _drain(stop, task)

    # Consumer gone: a publish to the now-empty subject must complete promptly.
    await asyncio.wait_for(bus.publish("s", _Evt(n=2)), timeout=1.0)
    await asyncio.wait_for(bus.publish("s", _Evt(n=3)), timeout=1.0)


async def test_redelivery_does_not_deadlock_on_full_queue() -> None:
    """A failing handler's redelivery must not wedge when the queue is full.

    Regression: ``_dispatch`` requeued the failed delivery while still holding its
    in-flight permit. The queue is sized to ``max_inflight``, so a redelivery
    ``put`` blocks when the queue is full; holding the permit across that put let
    a racing publisher steal the single slot the consume loop's ``get`` freed,
    leaving every worker blocked on ``put``, the loop blocked on ``sem.acquire``,
    and nothing able to drain — a permanent deadlock. Releasing the permit before
    the requeue keeps the loop able to acquire and drain. The one-slot queue plus
    concurrent publishers below reproduce the slot-steal race.
    """
    for _ in range(30):
        bus = InProcessBus()
        attempts: list[int] = []
        done = asyncio.Event()

        async def handler(
            _evt: _Evt, counter: list[int] = attempts, signal: asyncio.Event = done
        ) -> None:
            counter.append(1)
            if len(counter) >= 5:
                signal.set()
                return
            raise RuntimeError("boom")

        stop = asyncio.Event()
        task = bus.run(
            "s",
            durable="d",
            model=_Evt,
            handler=handler,
            max_inflight=1,
            max_deliver=100,
            stop_event=stop,
        )
        await asyncio.sleep(0)
        publishers = [asyncio.create_task(bus.publish("s", _Evt(n=i))) for i in range(4)]
        await asyncio.wait_for(asyncio.gather(*publishers), timeout=2.0)
        await asyncio.wait_for(done.wait(), timeout=2.0)
        await _drain(stop, task)


async def test_concurrent_publish_during_consumer_shutdown_settles() -> None:
    """Publishes racing a consumer stop never hang and exactly-once still holds.

    Stress: repeatedly start a consumer, fire concurrent publishes, and stop it
    mid-flight. The deregister-then-drain ordering must keep both the publishers
    and the consumer task from wedging across many cycles.
    """
    for _ in range(50):
        bus = InProcessBus()
        seen: list[int] = []

        async def handler(evt: _Evt, sink: list[int] = seen) -> None:
            sink.append(evt.n)

        stop = asyncio.Event()
        task = bus.run("s", durable="d", model=_Evt, handler=handler, stop_event=stop)
        await asyncio.sleep(0)
        publishers = [asyncio.create_task(bus.publish("s", _Evt(n=i))) for i in range(8)]
        stop.set()  # race the publishes against shutdown
        await asyncio.wait_for(asyncio.gather(*publishers), timeout=1.0)
        await asyncio.wait_for(task, timeout=1.0)
        # No delivery is ever seen more than once (no duplication under the race).
        assert len(seen) == len(set(seen))
        assert bus._consumers == {}
