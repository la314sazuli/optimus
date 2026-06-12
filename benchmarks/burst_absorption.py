"""Model JetStream back-pressure under a sustained image burst on one replica.

Mission experiment 2. A single detection replica sustains ~10 img/s; a raid can
offer 50-100 img/s for minutes. This harness drives the *real*
:meth:`~optimus.bus.nats.EventBus.consume` pull-consumer loop with a faithful
fake JetStream that reproduces the two behaviours that decide whether a burst is
absorbed safely or melts down:

* **Pull-fetch delivery.** A message is only "delivered" (and only then starts
  its ``ack_wait`` redelivery timer) when the consumer ``fetch()``-es it. The
  loop clamps each fetch to the spare in-flight budget, so surplus messages stay
  buffered *un-delivered* in the stream — and an un-delivered message cannot
  time out. The fake enforces exactly this: only fetched-but-unacked messages
  are eligible for an ``ack_wait`` redelivery.

* **ack_wait redelivery.** If a delivered message is not acked within
  ``ack_wait`` it is redelivered (up to ``max_deliver``). The fake re-queues such
  messages and counts the redelivery, so the harness can flag whether the
  default ``ack_wait`` (60 s) is comfortably above the time a fetched message
  spends in-handler at burst depth.

We measure, over a 60 s offered burst at 100 img/s into a replica that processes
~10 img/s: peak concurrent in-flight (must stay <= ``max_inflight``), peak
buffered backlog (the surplus that must live in JetStream, not replica RAM),
redelivery count (must be ~0 — no spurious redelivery storm), and the drain time
after the burst ends.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field

from pydantic import BaseModel

from optimus.bus.nats import EventBus


class _BurstModel(BaseModel):
    """Trivial event matching the consume loop's JSON validation.

    The burst test exercises only the bus back-pressure machinery, not detection
    logic, so any valid model the consumer can ``model_validate_json`` suffices.
    """

    correlation_id: str = "burst"


_MODEL_JSON = _BurstModel().model_dump_json().encode("utf-8")


class _SimMsg:
    """A simulated JetStream message recording ack/nak/term and delivery count."""

    __slots__ = ("acked", "data", "delivered_at", "deliveries", "naked", "termed")

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.acked = False
        self.naked = False
        self.termed = False
        self.deliveries = 0
        self.delivered_at: float | None = None

    async def ack(self) -> None:
        self.acked = True

    async def nak(self) -> None:
        self.naked = True

    async def term(self) -> None:
        self.termed = True


@dataclass
class _BurstStats:
    """Observed back-pressure behaviour over the run."""

    offered: int = 0
    processed: int = 0
    redeliveries: int = 0
    peak_inflight: int = 0
    peak_backlog: int = 0
    drain_seconds: float = 0.0
    samples_inflight: list[int] = field(default_factory=list)
    samples_backlog: list[int] = field(default_factory=list)


class _BurstJetStream:
    """Fake JetStream: time-driven arrivals, pull delivery, ack_wait redelivery.

    The simulation clock is the asyncio loop time. Arrivals are injected by the
    driver (see :func:`run_burst`); ``fetch`` hands out up to ``batch`` buffered
    messages, stamping each delivered message so the ack_wait sweep can redeliver
    it if it goes unacked. Un-delivered (still-buffered) messages are never
    eligible for redelivery, mirroring real pull-consumer semantics.
    """

    def __init__(self, *, ack_wait: float, max_deliver: int, stats: _BurstStats) -> None:
        self._ready: list[_SimMsg] = []  # buffered, awaiting delivery
        self._in_flight: list[_SimMsg] = []  # delivered, awaiting ack
        self._ack_wait = ack_wait
        self._max_deliver = max_deliver
        self._stats = stats
        self._loop = asyncio.get_event_loop()

    def offer(self, msg: _SimMsg) -> None:
        """Inject a new arrival into the buffered backlog."""
        self._ready.append(msg)
        self._stats.offered += 1

    def backlog(self) -> int:
        """Messages buffered but not yet delivered (must live in JetStream)."""
        return len(self._ready)

    async def pull_subscribe(self, subject: str, durable: str, config: object) -> _BurstJetStream:
        return self

    async def fetch(self, batch: int, timeout: float = 5.0) -> list[_SimMsg]:  # noqa: ASYNC109
        self._sweep_ack_wait()
        if not self._ready:
            await asyncio.sleep(0.001)
            raise TimeoutError
        take = self._ready[:batch]
        self._ready = self._ready[batch:]
        now = self._loop.time()
        for m in take:
            m.deliveries += 1
            m.delivered_at = now
            self._in_flight.append(m)
        return take

    def _sweep_ack_wait(self) -> None:
        """Redeliver any delivered-but-unacked message past its ack_wait deadline."""
        now = self._loop.time()
        still: list[_SimMsg] = []
        for m in self._in_flight:
            if m.acked or m.termed:
                continue
            if m.delivered_at is not None and (now - m.delivered_at) >= self._ack_wait:
                if m.deliveries < self._max_deliver:
                    self._stats.redeliveries += 1
                    m.delivered_at = None
                    self._ready.append(m)  # back to buffered for re-delivery
                else:
                    m.termed = True  # exhausted; JetStream would drop it
            else:
                still.append(m)
        self._in_flight = still


async def run_burst(
    *,
    offered_rate: float,
    duration_s: float,
    capacity_rate: float,
    max_inflight: int,
    fetch_batch: int,
    ack_wait: float,
    max_deliver: int,
    time_scale: float,
) -> _BurstStats:
    """Drive one replica's consume loop under a sustained offered burst.

    ``capacity_rate`` images/sec is modelled as a per-image handler delay of
    ``max_inflight / capacity_rate`` seconds (so ``max_inflight`` concurrent
    handlers clear ``capacity_rate`` images/sec). ``time_scale`` compresses wall
    time (e.g. 0.1 runs a 60 s scenario in ~6 s) by scaling every duration and
    the ack_wait together, preserving their ratios — the property under test.
    """
    stats = _BurstStats()
    js = _BurstJetStream(ack_wait=ack_wait * time_scale, max_deliver=max_deliver, stats=stats)
    bus = EventBus(js)  # type: ignore[arg-type]

    per_image_delay = (max_inflight / capacity_rate) * time_scale
    inflight = 0
    loop = asyncio.get_event_loop()

    async def handler(_event: object) -> None:
        nonlocal inflight
        inflight += 1
        stats.peak_inflight = max(stats.peak_inflight, inflight)
        try:
            await asyncio.sleep(per_image_delay)
            stats.processed += 1
        finally:
            inflight -= 1

    # Minimal valid payload for the model the consumer validates against.
    payload = _MODEL_JSON

    stop = asyncio.Event()
    consume_task = asyncio.create_task(
        bus.consume(
            "events.image_fetched.v1",
            durable="detection",
            model=_BurstModel,
            handler=handler,
            batch=fetch_batch,
            fetch_timeout=0.05,
            max_deliver=max_deliver,
            max_inflight=max_inflight,
            ack_wait=ack_wait * time_scale,
            stop_event=stop,
        )
    )

    # Arrival driver: inject Poisson-ish steady arrivals for `duration_s`.
    interval = (1.0 / offered_rate) * time_scale
    burst_end = loop.time() + duration_s * time_scale
    sampler_stop = asyncio.Event()

    async def sampler() -> None:
        while not sampler_stop.is_set():
            stats.samples_inflight.append(inflight)
            b = js.backlog()
            stats.samples_backlog.append(b)
            stats.peak_backlog = max(stats.peak_backlog, b)
            await asyncio.sleep(0.02 * time_scale)

    sampler_task = asyncio.create_task(sampler())

    while loop.time() < burst_end:
        js.offer(_SimMsg(payload))
        await asyncio.sleep(interval)

    # Let the replica drain the backlog after arrivals stop.
    drain_start = loop.time()
    while js.backlog() > 0 or inflight > 0:
        await asyncio.sleep(0.01 * time_scale)
        if loop.time() - drain_start > 120 * time_scale:  # safety bound
            break
    drain_wall = (loop.time() - drain_start) / time_scale

    sampler_stop.set()
    stop.set()
    await asyncio.gather(sampler_task, consume_task, return_exceptions=True)
    stats.drain_seconds = drain_wall
    return stats


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="benchmarks.burst_absorption",
        description="JetStream back-pressure under a sustained image burst (one replica).",
    )
    p.add_argument("--offered-rate", type=float, default=100.0, help="offered img/s (default 100)")
    p.add_argument("--duration", type=float, default=60.0, help="burst seconds (default 60)")
    p.add_argument("--capacity", type=float, default=10.0, help="replica img/s (default 10)")
    p.add_argument("--max-inflight", type=int, default=10, help="detection_max_inflight (def 10)")
    p.add_argument("--fetch-batch", type=int, default=16, help="detection_fetch_batch (def 16)")
    p.add_argument("--ack-wait", type=float, default=60.0, help="ack_wait seconds (default 60)")
    p.add_argument("--max-deliver", type=int, default=5, help="max_deliver (default 5)")
    p.add_argument(
        "--time-scale",
        type=float,
        default=0.05,
        help="compress wall time by this factor, preserving ratios (default 0.05)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint: ``python -m benchmarks.burst_absorption``."""
    args = _parse_args(argv)
    stats = asyncio.run(
        run_burst(
            offered_rate=args.offered_rate,
            duration_s=args.duration,
            capacity_rate=args.capacity,
            max_inflight=args.max_inflight,
            fetch_batch=args.fetch_batch,
            ack_wait=args.ack_wait,
            max_deliver=args.max_deliver,
            time_scale=args.time_scale,
        )
    )
    drain = stats.drain_seconds
    expected_backlog = (args.offered_rate - args.capacity) * args.duration
    expected_drain = expected_backlog / args.capacity
    print("# Burst absorption (one detection replica)")
    print()
    print(
        f"offered            : {args.offered_rate:.0f} img/s for {args.duration:.0f}s "
        f"= {args.offered_rate * args.duration:.0f} images"
    )
    print(
        f"replica capacity   : {args.capacity:.0f} img/s "
        f"(max_inflight={args.max_inflight}, ack_wait={args.ack_wait:.0f}s)"
    )
    print(f"images offered     : {stats.offered}")
    print(f"images processed   : {stats.processed}")
    print(f"peak in-flight     : {stats.peak_inflight}  (ceiling = {args.max_inflight})")
    print(f"peak backlog (JS)  : {stats.peak_backlog}  (~expected surplus {expected_backlog:.0f})")
    print(f"redeliveries       : {stats.redeliveries}  (want 0 -> no redelivery storm)")
    print(f"drain after burst  : {drain:.1f}s  (~expected {expected_drain:.0f}s)")
    print()
    inflight_ok = stats.peak_inflight <= args.max_inflight
    no_storm = stats.redeliveries == 0
    print(f"in-flight bounded  : {'PASS' if inflight_ok else 'FAIL'}")
    print(f"no redelivery storm: {'PASS' if no_storm else 'FAIL'}")


if __name__ == "__main__":
    main()
