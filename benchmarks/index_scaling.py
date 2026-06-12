"""Measure how the detection HashIndex (multi-index hashing) scales with entry count.

The index-size axis is otherwise untested: the throughput harness
(:mod:`benchmarks.load`) seeds the index from the small synthetic corpus, so it
never characterizes a single guild whose hash set has grown to tens or hundreds
of thousands of entries. This script fills that gap.

For each requested size it builds a :class:`~optimus.services.detection.index.HashIndex`
from ``N`` synthetic 64-bit hash sets (a configurable fraction of which carry a
mirror sibling, so the index holds the same mix of original + flipped entries the
production builder produces), then:

* records process RSS before and after the build (so the delta attributes memory
  to the index, isolating it from interpreter/library baseline),
* times the cold build (the per-replica warm-up cost paid on boot and on every
  ``scheduler_index_rebuild_interval`` rebuild),
* runs a pool of ``candidates()`` queries at the production candidate radius and
  reports p50/p95/p99 latency. Half the queries target a hash known to be near a
  stored entry (the matched-image path); half are random (the clean-image path,
  the common case), so the percentiles reflect a realistic mix.

Synthetic hashes are drawn from a seeded PRNG so runs are reproducible. They are
uniformly random 64-bit values. For multi-index hashing this is the *worst case*:
uniform substrings spread evenly across the per-table buckets, maximizing the
candidate set each query must verify. Real scam-campaign hashes cluster
(near-duplicate re-shares) into fewer buckets, so these latencies are
conservative upper bounds.
"""

from __future__ import annotations

import argparse
import gc
import resource
import sys
import time
from dataclasses import dataclass

import numpy as np

from optimus.services.detection.index import HashIndex, KnownHash
from optimus.services.detection.matcher import DEFAULT_CANDIDATE_RADIUS


def _rss_bytes() -> int:
    """Process RSS in bytes (``ru_maxrss`` is KiB on Linux, bytes on macOS)."""
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return maxrss * 1024 if sys.platform != "darwin" else maxrss


def _rand64(rng: np.random.Generator) -> int:
    """Draw a uniform unsigned 64-bit integer."""
    return int(rng.integers(0, 1 << 64, dtype=np.uint64))


def _near(rng: np.random.Generator, value: int, flips: int) -> int:
    """Return ``value`` with ``flips`` random bits toggled (Hamming distance flips)."""
    out = value
    for pos in rng.choice(64, size=flips, replace=False):
        out ^= 1 << int(pos)
    return out


def _make_entries(n: int, *, mirror_fraction: float, seed: int) -> list[KnownHash]:
    """Build ``n`` synthetic KnownHash entries; ``mirror_fraction`` carry a mirror.

    Each entry's four hashes are independent uniform 64-bit values. A mirror
    sibling (when present) is an independent hash set, so it adds a second,
    unrelated node to the index exactly as a real flipped image would.
    """
    rng = np.random.default_rng(seed)
    entries: list[KnownHash] = []
    for i in range(n):
        phash = _rand64(rng)
        mirror = None
        if rng.random() < mirror_fraction:
            mirror = {
                "phash": _rand64(rng),
                "dhash": _rand64(rng),
                "whash": _rand64(rng),
                "ahash": _rand64(rng),
            }
        entries.append(
            KnownHash(
                hash_id=f"h{i}",
                phash=phash,
                dhash=_rand64(rng),
                whash=_rand64(rng),
                ahash=_rand64(rng),
                source="guild",
                campaign_id=f"c{i % 64}",
                mirror=mirror,
            )
        )
    return entries


def _percentile(sorted_vals: list[float], pct: float) -> float:
    """Nearest-rank percentile (``pct`` in [0, 100]) over ascending values."""
    if not sorted_vals:
        return 0.0
    rank = max(1, min(len(sorted_vals), round(pct / 100.0 * len(sorted_vals))))
    return sorted_vals[rank - 1]


@dataclass(frozen=True, slots=True)
class SizeResult:
    """Build cost, memory, and query latency at one index size."""

    size: int
    tree_nodes: int
    build_seconds: float
    index_rss_mb: float
    bytes_per_entry: float
    q_p50_us: float
    q_p95_us: float
    q_p99_us: float
    q_mean_us: float
    q_max_us: float
    mean_candidates: float


def measure_size(
    size: int,
    *,
    queries: int,
    radius: int,
    mirror_fraction: float,
    seed: int,
) -> SizeResult:
    """Build an index of ``size`` entries and measure build/memory/query cost."""
    entries = _make_entries(size, mirror_fraction=mirror_fraction, seed=seed)
    # Query targets: half near a real stored phash (match path), half random
    # (clean path). Built before the RSS baseline so the query pool memory is
    # not attributed to the index.
    qrng = np.random.default_rng(seed + 1)
    stored_phashes = [e.phash for e in entries]
    targets: list[int] = []
    for i in range(queries):
        if i % 2 == 0:
            base = stored_phashes[int(qrng.integers(0, len(stored_phashes)))]
            targets.append(_near(qrng, base, int(qrng.integers(0, radius + 1))))
        else:
            targets.append(_rand64(qrng))

    gc.collect()
    rss_before = _rss_bytes()
    t0 = time.perf_counter()
    index = HashIndex(entries)
    build_seconds = time.perf_counter() - t0
    gc.collect()
    rss_after = _rss_bytes()
    index_rss = max(0, rss_after - rss_before)
    tree_nodes = index.node_count  # indexed node count (originals + mirror siblings)

    latencies_us: list[float] = []
    total_candidates = 0
    for target in targets:
        s = time.perf_counter()
        cands = index.candidates(target, radius)
        latencies_us.append((time.perf_counter() - s) * 1e6)
        total_candidates += len(cands)

    latencies_us.sort()
    # Keep the index alive across the RSS read above; drop it now.
    del index, entries
    return SizeResult(
        size=size,
        tree_nodes=tree_nodes,
        build_seconds=build_seconds,
        index_rss_mb=index_rss / (1024 * 1024),
        bytes_per_entry=index_rss / size if size else 0.0,
        q_p50_us=_percentile(latencies_us, 50),
        q_p95_us=_percentile(latencies_us, 95),
        q_p99_us=_percentile(latencies_us, 99),
        q_mean_us=sum(latencies_us) / len(latencies_us),
        q_max_us=latencies_us[-1],
        mean_candidates=total_candidates / queries,
    )


def _render(results: list[SizeResult], *, radius: int) -> str:
    """Render a Markdown summary table for the sweep."""
    lines = [
        f"# Index scaling (MIH HashIndex), candidate radius {radius}",
        "",
        "| Entries | Tree nodes | Build (s) | Index RSS (MB) | B/entry | "
        "q p50 (us) | q p95 (us) | q p99 (us) | q max (us) | mean cands |",
        "| ------- | ---------- | --------- | -------------- | ------- | "
        "---------- | ---------- | ---------- | ---------- | ---------- |",
    ]
    for r in results:
        lines.append(
            f"| {r.size:,} | {r.tree_nodes:,} | {r.build_seconds:.3f} | "
            f"{r.index_rss_mb:.1f} | {r.bytes_per_entry:.0f} | "
            f"{r.q_p50_us:.1f} | {r.q_p95_us:.1f} | {r.q_p99_us:.1f} | "
            f"{r.q_max_us:.1f} | {r.mean_candidates:.2f} |"
        )
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="benchmarks.index_scaling",
        description="Measure MIH HashIndex build/memory/query cost vs entry count.",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[10_000, 100_000, 500_000],
        help="entry counts to sweep (default: 10000 100000 500000)",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=2000,
        help="candidate() queries per size for the latency sample (default: 2000)",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=DEFAULT_CANDIDATE_RADIUS,
        help=f"phash Hamming query radius (default: {DEFAULT_CANDIDATE_RADIUS})",
    )
    parser.add_argument(
        "--mirror-fraction",
        type=float,
        default=0.5,
        help="fraction of entries carrying a mirror sibling (default: 0.5)",
    )
    parser.add_argument("--seed", type=int, default=1234, help="PRNG seed (default: 1234)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint: ``python -m benchmarks.index_scaling``."""
    args = _parse_args(argv)
    results: list[SizeResult] = []
    for size in args.sizes:
        result = measure_size(
            size,
            queries=args.queries,
            radius=args.radius,
            mirror_fraction=args.mirror_fraction,
            seed=args.seed,
        )
        results.append(result)
        print(
            f"size={result.size:>8,}  nodes={result.tree_nodes:>8,}  "
            f"build={result.build_seconds:6.3f}s  rss={result.index_rss_mb:7.1f}MB  "
            f"p50={result.q_p50_us:7.1f}us  p95={result.q_p95_us:8.1f}us  "
            f"p99={result.q_p99_us:8.1f}us  cands={result.mean_candidates:.2f}",
            flush=True,
        )
    print("\n" + _render(results, radius=args.radius))


if __name__ == "__main__":
    main()
