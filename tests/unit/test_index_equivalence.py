"""Equivalence of the MIH-backed HashIndex to a brute-force linear scan.

The detection HashIndex switched from a BK-tree to multi-index hashing. This
must be a *drop-in*: ``candidates(phash, radius)`` has to return exactly the
KnownHash entries (originals *and* mirror siblings) a linear Hamming scan would,
for any corpus and any radius — including the production candidate radius 18.
These property tests are the drop-in proof.
"""

from __future__ import annotations

import random

from hypothesis import given, settings
from hypothesis import strategies as st

from optimus.hashing.perceptual import hamming
from optimus.services.detection.index import HashIndex, KnownHash

_U64 = st.integers(min_value=0, max_value=(1 << 64) - 1)


def _make_entry(i: int, phash: int, *, mirror_phash: int | None) -> KnownHash:
    mirror = None
    if mirror_phash is not None:
        mirror = {"phash": mirror_phash, "dhash": i, "whash": i, "ahash": i}
    return KnownHash(
        hash_id=f"h{i}",
        phash=phash,
        dhash=i,
        whash=i,
        ahash=i,
        source="guild",
        campaign_id=f"c{i}",
        mirror=mirror,
    )


def _linear_candidates(entries: list[KnownHash], query: int, radius: int) -> set[tuple[str, int]]:
    """Brute-force reference: (hash_id, scored_phash) pairs within radius.

    A mirror sibling resolves to the source hash_id but scores its flipped phash,
    so the identity key is (hash_id, the_phash_that_was_within_radius).
    """
    out: set[tuple[str, int]] = set()
    for e in entries:
        if hamming(e.phash, query) <= radius:
            out.add((e.hash_id, e.phash))
        if e.mirror is not None and hamming(e.mirror["phash"], query) <= radius:
            out.add((e.hash_id, e.mirror["phash"]))
    return out


def _index_candidates(index: HashIndex, query: int, radius: int) -> set[tuple[str, int]]:
    return {(k.hash_id, k.phash) for k in index.candidates(query, radius)}


@settings(max_examples=100, deadline=None)
@given(
    phashes=st.lists(_U64, min_size=1, max_size=120, unique=True),
    query=_U64,
    radius=st.integers(min_value=0, max_value=64),
)
def test_candidates_match_linear_scan(phashes: list[int], query: int, radius: int) -> None:
    entries = [_make_entry(i, p, mirror_phash=None) for i, p in enumerate(phashes)]
    index = HashIndex(entries)
    assert _index_candidates(index, query, radius) == _linear_candidates(entries, query, radius)


@settings(max_examples=80, deadline=None)
@given(seed=st.integers(min_value=0, max_value=100_000), radius=st.sampled_from([0, 1, 8, 12, 18]))
def test_candidates_with_mirrors_match_linear_scan(seed: int, radius: int) -> None:
    rng = random.Random(seed)
    entries: list[KnownHash] = []
    for i in range(80):
        phash = rng.getrandbits(64)
        # Half carry a mirror sibling, as the production builder produces.
        mirror_phash = rng.getrandbits(64) if rng.random() < 0.5 else None
        entries.append(_make_entry(i, phash, mirror_phash=mirror_phash))
    index = HashIndex(entries)
    # A handful of queries: some near a stored phash, some random.
    stored = [e.phash for e in entries]
    queries = [rng.choice(stored) for _ in range(4)] + [rng.getrandbits(64) for _ in range(4)]
    for q in queries:
        assert _index_candidates(index, q, radius) == _linear_candidates(entries, q, radius)


def test_len_counts_sources_not_mirrors_with_mih() -> None:
    entries = [
        _make_entry(0, 111, mirror_phash=222),
        _make_entry(1, 333, mirror_phash=None),
    ]
    index = HashIndex(entries)
    assert len(index) == 2
    # node_count includes the one mirror sibling.
    assert index.node_count == 3
