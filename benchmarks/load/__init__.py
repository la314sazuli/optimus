"""Throughput load-test harness for the detection pipeline.

Where the :mod:`benchmarks` (accuracy) harness measures *what* the detection
pipeline decides, this package measures *how fast* one detection replica can
decide it. It pushes a configurable number of synthetic images concurrently
through the real :class:`~optimus.services.detection.worker.DetectionWorker`
(the same decode-subprocess + perceptual-hash + BK-tree + ensemble code path
production runs, including the ``asyncio.to_thread`` decode/hash offload) and
reports sustained images/sec, p50/p95/p99 end-to-end latency, and peak RSS.

No external network or live NATS/Redis/Postgres is required: the worker's
injected index/whitelist/sensitivity/idempotency hooks are satisfied by simple
in-process fakes (:mod:`benchmarks.load.fakes`) seeded from the same corpus the
accuracy harness uses, and queue arrival is simulated with asyncio. See
``docs/performance-notes.md`` ("Throughput baseline").
"""

from __future__ import annotations
