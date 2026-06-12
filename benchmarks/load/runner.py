"""Drive synthetic images through the real detection worker and measure throughput.

The runner builds a pool of :class:`~optimus.contracts.events.ImageFetchedEvent`
payloads from the synthetic corpus (PNG-encoded, base64 inline, exactly as the
ingest service hands them to detection), then fans them through a single
:class:`~optimus.services.detection.worker.DetectionWorker` with a bounded number
of concurrent in-flight images. Each image's wall-clock end-to-end latency is
recorded; sustained images/sec is the image count divided by the wall time of the
whole run (so it reflects real overlap, not the sum of per-image latencies).

Concurrency is bounded by the number of worker coroutines rather than a real
queue: ``concurrency`` coroutines pull from a shared in-memory job cursor, which
models a saturated arrival rate (the queue is always non-empty) — the worst case
for a throughput baseline.
"""

from __future__ import annotations

import asyncio
import base64
import io
import resource
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from benchmarks.corpus import Corpus, CorpusImage
from benchmarks.load.fakes import InMemoryIdempotency, StaticIndexes
from optimus.contracts.events import ImageFetchedEvent, Verdict
from optimus.services.detection.worker import DetectionWorker

GUILD_ID = 1


def _png_bytes(img: CorpusImage) -> bytes:
    """Encode a corpus image as PNG bytes (the wire form ingest produces)."""
    buf = io.BytesIO()
    img.image.save(buf, format="PNG")
    return buf.getvalue()


def build_events(corpus: Corpus, count: int) -> list[ImageFetchedEvent]:
    """Build ``count`` image-fetched events cycling through the corpus images.

    Each event carries a distinct ``idempotency_key`` so none is skipped as a
    duplicate, and the corpus images are PNG-encoded once and reused across
    repeats (encoding is not part of the pipeline cost we measure).
    """
    images = list(corpus.all_images())
    if not images:
        raise ValueError("corpus produced no images")
    encoded = [_png_bytes(img) for img in images]
    occurred = datetime.now(UTC)
    events: list[ImageFetchedEvent] = []
    for i in range(count):
        data = encoded[i % len(encoded)]
        events.append(
            ImageFetchedEvent(
                correlation_id=f"load-{i}",
                occurred_at=occurred,
                guild_id=GUILD_ID,
                channel_id=2,
                message_id=1000 + i,
                attachment_id=3,
                uploader_id=4,
                idempotency_key=f"load-{i}",
                content_type="image/png",
                size_bytes=len(data),
                sha256="0" * 64,
                data_b64=base64.b64encode(data).decode("ascii"),
            )
        )
    return events


def _build_worker(corpus: Corpus) -> DetectionWorker:
    """Wire a real :class:`DetectionWorker` over the in-process fakes."""
    indexes = StaticIndexes(corpus)
    return DetectionWorker(
        guild_index=indexes.guild_index,
        global_index=indexes.global_index,
        whitelist=indexes.whitelist,
        sensitivity=indexes.sensitivity,
        idempotency_acquire=InMemoryIdempotency().acquire,
        swarm=None,
    )


def _percentile(sorted_ms: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile (``pct`` in [0, 100]) over ascending ``sorted_ms``."""
    if not sorted_ms:
        return 0.0
    rank = max(1, min(len(sorted_ms), round(pct / 100.0 * len(sorted_ms))))
    return sorted_ms[rank - 1]


def _peak_rss_bytes() -> int:
    """Process peak resident set size in bytes (``ru_maxrss`` is KiB on Linux)."""
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes. The harness targets Linux sandboxes.
    return maxrss * 1024 if sys.platform != "darwin" else maxrss


@dataclass(frozen=True, slots=True)
class LoadResult:
    """Aggregate throughput/latency metrics for one concurrency level."""

    concurrency: int
    images: int
    wall_seconds: float
    images_per_sec: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    max_ms: float
    peak_rss_mb: float
    non_decisions: int

    @property
    def latencies_ok(self) -> bool:
        """Whether every image produced a usable (non-``None``) worker result."""
        return self.non_decisions == 0


async def run_load(
    corpus: Corpus,
    *,
    concurrency: int,
    images: int,
) -> LoadResult:
    """Push ``images`` synthetic images through the worker at ``concurrency`` in-flight.

    Returns the aggregate :class:`LoadResult`. The worker and event pool are
    built fresh per call so repeated levels do not share idempotency state.
    """
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    if images < 1:
        raise ValueError("images must be >= 1")

    worker = _build_worker(corpus)
    events = build_events(corpus, images)
    latencies_ms: list[float] = [0.0] * images
    next_index = 0
    lock = asyncio.Lock()

    async def claim() -> int | None:
        nonlocal next_index
        async with lock:
            if next_index >= images:
                return None
            idx = next_index
            next_index += 1
            return idx

    async def worker_loop() -> int:
        """Pull jobs until the pool is exhausted; return local non-decision count."""
        local_nd = 0
        while True:
            idx = await claim()
            if idx is None:
                return local_nd
            start = time.perf_counter()
            result = await worker.handle(events[idx])
            latencies_ms[idx] = (time.perf_counter() - start) * 1000.0
            # A duplicate (None) cannot happen here — keys are unique — but a
            # decode failure surfaces as a NON_DECISION verdict, which we count.
            if result is not None and result.verdict.verdict is Verdict.NON_DECISION:
                local_nd += 1

    wall_start = time.perf_counter()
    counts = await asyncio.gather(*(worker_loop() for _ in range(concurrency)))
    wall_seconds = time.perf_counter() - wall_start
    non_decisions = sum(counts)

    ordered = sorted(latencies_ms)
    images_per_sec = images / wall_seconds if wall_seconds > 0 else 0.0
    return LoadResult(
        concurrency=concurrency,
        images=images,
        wall_seconds=wall_seconds,
        images_per_sec=images_per_sec,
        p50_ms=_percentile(ordered, 50),
        p95_ms=_percentile(ordered, 95),
        p99_ms=_percentile(ordered, 99),
        mean_ms=sum(latencies_ms) / images,
        max_ms=ordered[-1],
        peak_rss_mb=_peak_rss_bytes() / (1024 * 1024),
        non_decisions=non_decisions,
    )
