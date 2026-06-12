"""Multi-index hashing (MIH) for sub-linear Hamming-radius lookups on 64-bit hashes.

A BK-tree degrades toward a linear scan at large radii on uniform 64-bit hashes:
pairwise Hamming distance is ``Binomial(64, 0.5)`` (mean 32, sd ~4), so a radius
``r`` prune window ``[d-r, d+r]`` around d≈32 covers nearly the whole populated
edge range and the triangle-inequality prune barely fires. At the production
candidate radius 18 this is acute.

Multi-index hashing (Norouzi, Punjani & Fleet, CVPR 2012) sidesteps this. Split
each 64-bit hash into ``m`` disjoint substrings of ``64 // m`` bits each, and keep
one exact-match hash table per substring (substring value -> list of entry ids).
By the pigeonhole principle, two hashes within total Hamming distance ``r`` must
agree to within ``floor(r / m)`` on at least one substring. So a query within
radius ``r`` is answered by, for each of the ``m`` substrings, enumerating the
Hamming ball of radius ``floor(r / m)`` in substring space, probing that
substring's table, unioning the candidate ids, and verifying the true 64-bit
Hamming distance on each candidate. The substring ball is tiny
(``sum(C(b, k) for k in 0..floor(r/m))`` keys for ``b``-bit substrings), so the
per-query work is dominated by candidate verification rather than a tree walk —
and the recall is exact, not approximate.

This module is hash-payload agnostic: it stores arbitrary string ids against
64-bit values and returns matching ids. The detection index layers
``KnownHash`` provenance on top of it.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from functools import cache

from optimus.hashing.perceptual import HASH_BITS, hamming

# Substring count. 64 bits split into m=4 disjoint 16-bit substrings. At the
# production candidate radius 18 this gives a per-substring search radius of
# floor(18/4)=4: each of the 4 tables is probed over a 16-bit Hamming ball of
# radius 4 (sum(C(16,k), k=0..4) = 2517 keys). Measured faster than m=8 at
# radius 18 (see docs/capacity.md): m=8's 8-bit buckets are 256-wide and admit
# far more candidates to verify, outweighing its smaller ball.
DEFAULT_SUBSTRING_COUNT = 4


def _substring_bits(m: int) -> int:
    if m < 1 or HASH_BITS % m != 0:
        raise ValueError(f"m must be a positive divisor of {HASH_BITS}, got {m}")
    return HASH_BITS // m


def split(value: int, m: int) -> list[int]:
    """Split a 64-bit ``value`` into ``m`` disjoint substrings (most-significant first)."""
    bits = _substring_bits(m)
    mask = (1 << bits) - 1
    return [(value >> (bits * (m - 1 - i))) & mask for i in range(m)]


def hamming_ball(center: int, bits: int, radius: int) -> Iterator[int]:
    """Yield every ``bits``-wide value within Hamming distance ``radius`` of ``center``.

    Enumerates by flipping each combination of up to ``radius`` bit positions.
    ``radius`` is clamped to ``bits`` (a wider radius covers the whole space).
    The center itself (radius 0) is always yielded first.
    """
    if radius < 0:
        raise ValueError("radius must be >= 0")
    r = min(radius, bits)
    positions = range(bits)
    # k=0 yields the center; ascending k keeps near matches first.
    for k in range(r + 1):
        yield from _flip_combinations(center, positions, k)


def _flip_combinations(center: int, positions: range, k: int) -> Iterator[int]:
    """Yield ``center`` with each distinct ``k``-subset of ``positions`` flipped."""
    if k == 0:
        yield center
        return
    # Iterative odometer over k ascending bit indices; avoids itertools import
    # churn and is the hot path for ball enumeration.
    idx = list(range(k))
    n = len(positions)
    while True:
        out = center
        for i in idx:
            out ^= 1 << i
        yield out
        j = k - 1
        while j >= 0 and idx[j] == n - k + j:
            j -= 1
        if j < 0:
            return
        idx[j] += 1
        for t in range(j + 1, k):
            idx[t] = idx[t - 1] + 1


@cache
def ball_size(bits: int, radius: int) -> int:
    """Number of distinct values within Hamming distance ``radius`` in ``bits`` space."""
    r = min(max(radius, 0), bits)
    return sum(math.comb(bits, k) for k in range(r + 1))


@cache
def flip_masks(bits: int, radius: int) -> tuple[int, ...]:
    """Return XOR masks enumerating the radius-``radius`` Hamming ball around 0.

    ``hamming_ball(center, bits, radius)`` equals ``center ^ mask`` for each mask.
    The mask set depends only on ``(bits, radius)``, not the center, so it is
    computed once and reused across all queries — turning per-query ball
    enumeration into a flat XOR loop (the query hot path). Cached because the
    production index hits a single ``(16, 4)`` configuration every query.
    """
    return tuple(hamming_ball(0, bits, radius))


class MultiIndexHash:
    """Exact Hamming-radius lookup over 64-bit hashes via multi-index hashing.

    Stores ``(value, id)`` pairs and answers :meth:`query` with all stored ids
    whose value is within a given Hamming radius — identical results to a linear
    scan, but sub-linear in practice for clustered radii. ``m`` substrings are
    fixed at construction.
    """

    __slots__ = ("_bits", "_m", "_size", "_tables", "_values")

    def __init__(self, m: int = DEFAULT_SUBSTRING_COUNT) -> None:
        self._bits = _substring_bits(m)  # validates m
        self._m = m
        # One table per substring: substring value -> list of entry ids.
        self._tables: list[dict[int, list[str]]] = [{} for _ in range(m)]
        # id -> full 64-bit value, for true-distance verification.
        self._values: dict[str, int] = {}
        self._size = 0

    def __len__(self) -> int:
        return self._size

    @property
    def substring_count(self) -> int:
        """Number of substring tables (``m``)."""
        return self._m

    def add(self, value: int, ident: str) -> None:
        """Index ``value`` (unsigned 64-bit) under ``ident``.

        Re-adding an ``ident`` overwrites its previous value (last write wins),
        mirroring the BK-tree payload-collision behavior the index relied on.
        """
        if value < 0 or value >= (1 << HASH_BITS):
            raise ValueError("value must be an unsigned 64-bit integer")
        if ident in self._values:
            self._remove(ident)
        self._values[ident] = value
        for table, sub in zip(self._tables, split(value, self._m), strict=True):
            table.setdefault(sub, []).append(ident)
        self._size += 1

    def _remove(self, ident: str) -> None:
        """Drop a previously-added ``ident`` from all substring tables."""
        old = self._values.pop(ident)
        for table, sub in zip(self._tables, split(old, self._m), strict=True):
            bucket = table.get(sub)
            if bucket is not None:
                bucket.remove(ident)
                if not bucket:
                    del table[sub]
        self._size -= 1

    def query(self, value: int, radius: int) -> list[str]:
        """Return all stored ids whose value is within ``radius`` of ``value``.

        Pigeonhole: a true match within total radius ``r`` agrees to within
        ``floor(r/m)`` on at least one substring. We enumerate each substring's
        ``floor(r/m)`` ball, union the candidate ids, then verify the true 64-bit
        distance. Results are de-duplicated and exact.

        At large radii the per-substring ball can exceed the corpus size (and at
        ``floor(r/m) >= bits`` it covers the whole substring space); a direct
        linear scan is then both cheaper and identical, so we fall back to it.
        The production candidate radius 18 (sub-radius 4 over 16-bit substrings,
        a 2,517-key ball) never trips this — it is a guard for pathological radii.
        """
        if radius < 0:
            raise ValueError("radius must be >= 0")
        sub_radius = radius // self._m
        # m * ball_size probes vs self._size verifications: scan if cheaper. Decide
        # from the (cheap, closed-form) ball_size *before* materializing masks, so a
        # pathological radius never tries to enumerate an astronomically large ball.
        if self._m * ball_size(self._bits, sub_radius) >= self._size:
            return [ident for ident, v in self._values.items() if hamming(v, value) <= radius]
        masks = flip_masks(self._bits, sub_radius)
        query_subs = split(value, self._m)
        values = self._values
        seen: set[str] = set()
        out: list[str] = []
        for table, qsub in zip(self._tables, query_subs, strict=True):
            get = table.get
            for mask in masks:
                bucket = get(qsub ^ mask)
                if bucket is None:
                    continue
                for ident in bucket:
                    if ident in seen:
                        continue
                    seen.add(ident)
                    if hamming(values[ident], value) <= radius:
                        out.append(ident)
        return out

    def values(self) -> Iterator[int]:
        """Yield every stored hash value (unordered)."""
        yield from self._values.values()
