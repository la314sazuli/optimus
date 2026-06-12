"""Process-health sampling for the soak harness.

Every sample captures the signals a slow leak shows up in: resident memory, open
file descriptors, the live asyncio task count, the on-disk SQLite size, the depth
of each bus consumer queue, end-to-end latency percentiles, and error counters
broken out by type. Samples are appended to a CSV so the full run can be
post-processed (slope, drift) rather than judged from a single snapshot.
"""

from __future__ import annotations

import os
import resource
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from prometheus_client import REGISTRY

# The fixed CSV column order. Kept explicit (not derived from a dataclass) so the
# header is stable across refactors and easy to load with any tooling.
CSV_COLUMNS = [
    "elapsed_s",
    "rss_mb",
    "open_fds",
    "asyncio_tasks",
    "sqlite_bytes",
    "sqlite_wal_bytes",
    "memstore_keys",
    "q_message_image",
    "q_image_fetched",
    "q_verdict",
    "images_sent",
    "images_acked",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "errors_total",
    "ingest_rejected",
    "detection_payload_rejected",
    "verdicts_clean",
    "verdicts_scam",
    "verdicts_non_decision",
    "bus_dropped",
    "bus_naked",
]


def rss_bytes() -> int:
    """Process resident set size in bytes (``ru_maxrss`` is KiB on Linux)."""
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return maxrss * 1024 if sys.platform != "darwin" else maxrss


def open_fd_count() -> int:
    """Number of open file descriptors for this process (Linux ``/proc``)."""
    try:
        return len(os.listdir(f"/proc/{os.getpid()}/fd"))
    except OSError:
        return -1


def metric_value(name: str, labels: dict[str, str] | None = None) -> float:
    """Read a single prometheus sample value from the default registry.

    Returns 0.0 when the metric or label combination has not been observed yet
    (a counter that never fired has no child series), which is the right zero for
    a soak delta.
    """
    labels = labels or {}
    for metric in REGISTRY.collect():
        for sample in metric.samples:
            if sample.name != name:
                continue
            if all(sample.labels.get(k) == v for k, v in labels.items()):
                return float(sample.value)
    return 0.0


def percentile(sorted_ms: list[float], pct: float) -> float:
    """Nearest-rank percentile (``pct`` in [0, 100]) over ascending values."""
    if not sorted_ms:
        return 0.0
    rank = max(1, min(len(sorted_ms), round(pct / 100.0 * len(sorted_ms))))
    return sorted_ms[rank - 1]


@dataclass
class Sample:
    """One row of soak telemetry."""

    elapsed_s: float
    rss_mb: float
    open_fds: int
    asyncio_tasks: int
    sqlite_bytes: int
    sqlite_wal_bytes: int
    memstore_keys: int
    q_message_image: int
    q_image_fetched: int
    q_verdict: int
    images_sent: int
    images_acked: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    errors_total: int
    ingest_rejected: float
    detection_payload_rejected: float
    verdicts_clean: float
    verdicts_scam: float
    verdicts_non_decision: float
    bus_dropped: float
    bus_naked: float

    def row(self) -> list[str]:
        return [
            f"{self.elapsed_s:.1f}",
            f"{self.rss_mb:.2f}",
            str(self.open_fds),
            str(self.asyncio_tasks),
            str(self.sqlite_bytes),
            str(self.sqlite_wal_bytes),
            str(self.memstore_keys),
            str(self.q_message_image),
            str(self.q_image_fetched),
            str(self.q_verdict),
            str(self.images_sent),
            str(self.images_acked),
            f"{self.p50_ms:.3f}",
            f"{self.p95_ms:.3f}",
            f"{self.p99_ms:.3f}",
            str(self.errors_total),
            f"{self.ingest_rejected:.0f}",
            f"{self.detection_payload_rejected:.0f}",
            f"{self.verdicts_clean:.0f}",
            f"{self.verdicts_scam:.0f}",
            f"{self.verdicts_non_decision:.0f}",
            f"{self.bus_dropped:.0f}",
            f"{self.bus_naked:.0f}",
        ]


@dataclass
class CsvWriter:
    """Append-only CSV writer that flushes each row (crash-safe partial output)."""

    path: Path
    _fh: TextIO | None = field(init=False, default=None)

    def __enter__(self) -> CsvWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        self._fh.write(",".join(CSV_COLUMNS) + "\n")
        self._fh.flush()
        return self

    def write(self, sample: Sample) -> None:
        assert self._fh is not None
        self._fh.write(",".join(sample.row()) + "\n")
        self._fh.flush()

    def __exit__(self, *exc: object) -> None:
        if self._fh is not None:
            self._fh.close()


def linear_slope(xs: Iterable[float], ys: Iterable[float]) -> float:
    """Ordinary least-squares slope of ``ys`` over ``xs`` (0.0 if degenerate)."""
    xs_l = list(xs)
    ys_l = list(ys)
    n = len(xs_l)
    if n < 2:
        return 0.0
    mean_x = sum(xs_l) / n
    mean_y = sum(ys_l) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs_l, ys_l, strict=True))
    den = sum((x - mean_x) ** 2 for x in xs_l)
    return num / den if den else 0.0
