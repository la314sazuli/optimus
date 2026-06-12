"""In-process fakes satisfying the detection worker's injected hooks.

The :class:`~optimus.services.detection.worker.DetectionWorker` depends on four
async hooks (guild index, global index, whitelist, sensitivity) plus an
idempotency-acquire callable. In production those are backed by Postgres
(indexes), a DB whitelist, a guild-config lookup, and Redis (idempotency). For a
throughput load test we want to exercise the *worker's* CPU/IO cost — decode
subprocess, perceptual hashing, BK-tree candidate gather, ensemble vote — without
standing up any of that infrastructure, so each hook is replaced by a trivial
in-memory stand-in that returns instantly.

The guild index is the one that matters for realism: it is the *real*
:class:`~optimus.services.detection.index.HashIndex`, seeded from the corpus
campaign bases by reusing :func:`benchmarks.harness.build_index`, so the BK-tree
candidate gather and ensemble vote do real work against a populated tree.
"""

from __future__ import annotations

from benchmarks.corpus import Corpus
from benchmarks.harness import build_index
from optimus.core.config import Sensitivity
from optimus.services.detection.index import HashIndex
from optimus.services.detection.matcher import WhitelistEntry


class InMemoryIdempotency:
    """A set-backed stand-in for the Redis ``SET NX`` idempotency guard.

    Returns ``True`` the first time a key is seen and ``False`` thereafter,
    mirroring the production guard's semantics without a Redis round trip. The
    load harness uses unique keys per image so every acquire succeeds and no
    image is skipped as a duplicate.
    """

    def __init__(self) -> None:
        self._seen: set[str] = set()

    async def acquire(self, key: str) -> bool:
        if key in self._seen:
            return False
        self._seen.add(key)
        return True


class StaticIndexes:
    """Holds the corpus-seeded guild index and an empty global index.

    Both lookups are pure (no await work beyond returning a cached object), so
    the worker's index access adds no artificial latency — the measured cost is
    the decode/hash/match work the harness is trying to characterize.
    """

    def __init__(self, corpus: Corpus) -> None:
        # Reuse the accuracy harness's index builder so the load test matches
        # against the exact same BK-tree the eval harness scores against.
        self._guild = build_index(corpus)
        self._global = HashIndex(())

    async def guild_index(self, guild_id: int) -> HashIndex:
        return self._guild

    async def global_index(self) -> HashIndex:
        return self._global

    async def whitelist(self, guild_id: int) -> list[WhitelistEntry]:
        return []

    async def sensitivity(self, guild_id: int) -> Sensitivity:
        return Sensitivity.BALANCED
