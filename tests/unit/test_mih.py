"""Tests for multi-index hashing (MIH), including equivalence to a linear scan.

MIH replaces the BK-tree as the phash index. Its query results must be
*identical* to a linear Hamming scan at every radius (it is exact, not
approximate), so the matcher's behavior is unchanged. The property test below is
the equivalence proof; the unit tests pin the substring-enumeration edge cases
(radius 0, sub-radius rounding at the production radius 18, full-width balls) and
the add/overwrite/remove bookkeeping the index relies on.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from optimus.hashing.mih import (
    DEFAULT_SUBSTRING_COUNT,
    MultiIndexHash,
    ball_size,
    hamming_ball,
    split,
)
from optimus.hashing.perceptual import hamming

_U64 = st.integers(min_value=0, max_value=(1 << 64) - 1)


def _linear_scan(values: dict[str, int], query: int, radius: int) -> set[str]:
    return {ident for ident, v in values.items() if hamming(v, query) <= radius}


# --- substring split -------------------------------------------------------


def test_split_reconstructs_value() -> None:
    value = 0x0123_4567_89AB_CDEF
    for m in (1, 2, 4, 8):
        bits = 64 // m
        subs = split(value, m)
        assert len(subs) == m
        rebuilt = 0
        for s in subs:
            rebuilt = (rebuilt << bits) | s
        assert rebuilt == value


def test_split_rejects_non_divisor() -> None:
    with pytest.raises(ValueError):
        split(0, 3)
    with pytest.raises(ValueError):
        split(0, 0)


# --- hamming ball ----------------------------------------------------------


def test_hamming_ball_radius_zero_is_center_only() -> None:
    assert list(hamming_ball(0b1010, bits=4, radius=0)) == [0b1010]


def test_hamming_ball_radius_one() -> None:
    got = set(hamming_ball(0b0000, bits=4, radius=1))
    # center + the four single-bit flips
    assert got == {0b0000, 0b0001, 0b0010, 0b0100, 0b1000}


def test_hamming_ball_is_exactly_the_radius_neighborhood() -> None:
    center, bits, radius = 0b10110, 5, 2
    got = list(hamming_ball(center, bits, radius))
    assert len(got) == len(set(got))  # no duplicates
    assert len(got) == ball_size(bits, radius)
    for v in got:
        assert hamming(center, v) <= radius
    # every value within the radius is present
    for v in range(1 << bits):
        if hamming(center, v) <= radius:
            assert v in got


def test_hamming_ball_radius_at_or_above_width_covers_space() -> None:
    bits = 4
    full = set(hamming_ball(0b0101, bits=bits, radius=bits))
    assert full == set(range(1 << bits))
    # clamped: radius beyond the width still just covers the whole space
    assert set(hamming_ball(0b0101, bits=bits, radius=bits + 5)) == full


def test_hamming_ball_negative_radius_raises() -> None:
    with pytest.raises(ValueError):
        list(hamming_ball(0, bits=4, radius=-1))


def test_ball_size_matches_binomial_sum() -> None:
    assert ball_size(16, 0) == 1
    assert ball_size(16, 4) == sum(math.comb(16, k) for k in range(5))
    assert ball_size(8, 2) == 1 + 8 + 28
    # clamps to full space
    assert ball_size(4, 10) == 1 << 4


# --- MultiIndexHash query --------------------------------------------------


def test_default_m_is_four() -> None:
    assert DEFAULT_SUBSTRING_COUNT == 4
    assert MultiIndexHash().substring_count == 4


def test_query_radius_zero_exact_match() -> None:
    idx = MultiIndexHash()
    idx.add(0xDEAD_BEEF_0000_0001, "a")
    idx.add(0xDEAD_BEEF_0000_0002, "b")  # differs in low bits
    assert idx.query(0xDEAD_BEEF_0000_0001, 0) == ["a"]
    # one bit off -> not an exact match
    assert idx.query(0xDEAD_BEEF_0000_0003, 0) == []


def test_query_empty_index() -> None:
    assert MultiIndexHash().query(123, 18) == []


def test_query_negative_radius_raises() -> None:
    idx = MultiIndexHash()
    idx.add(1, "a")
    with pytest.raises(ValueError):
        idx.query(1, -1)


def test_query_dedups_ids_found_in_multiple_substrings() -> None:
    # A value identical to the query in every substring is reachable from all
    # m tables; it must appear exactly once.
    idx = MultiIndexHash()
    idx.add(0, "z")
    assert idx.query(0, 18) == ["z"]


def test_query_admits_radius_18_match_across_substrings() -> None:
    # 18 flips spread so no single 16-bit substring exceeds floor(18/4)=4.
    base = 0
    flipped = 0
    # 5 + 5 + 4 + 4 = 18 bits, but cap each substring at 4 to satisfy pigeonhole
    # on at least one — here we spread 4,4,4,6 to prove verification still works
    # even when one substring is over the sub-radius (another must be within it).
    for chunk, count in zip(range(4), (4, 4, 4, 6), strict=True):
        for bit in range(count):
            flipped ^= 1 << (chunk * 16 + bit)
    assert hamming(base, flipped) == 18
    idx = MultiIndexHash()
    idx.add(base, "x")
    assert idx.query(flipped, 18) == ["x"]
    # one bit further (19) is outside the radius
    nineteen = flipped ^ (1 << (3 * 16 + 6))
    assert hamming(base, nineteen) == 19
    assert idx.query(nineteen, 18) == []


def test_add_overwrites_same_id() -> None:
    idx = MultiIndexHash()
    idx.add(100, "a")
    idx.add(200, "a")  # last write wins
    assert len(idx) == 1
    assert idx.query(200, 0) == ["a"]
    assert idx.query(100, 0) == []


def test_len_and_values() -> None:
    idx = MultiIndexHash()
    items = {f"h{i}": i for i in range(5)}
    for ident, v in items.items():
        idx.add(v, ident)
    assert len(idx) == 5
    assert set(idx.values()) == set(items.values())


def test_add_rejects_out_of_range() -> None:
    idx = MultiIndexHash()
    with pytest.raises(ValueError):
        idx.add(1 << 64, "a")
    with pytest.raises(ValueError):
        idx.add(-1, "a")


# --- equivalence proof (vs linear scan) ------------------------------------


@settings(max_examples=120, deadline=None)
@given(
    values=st.lists(_U64, min_size=1, max_size=150, unique=True),
    query=_U64,
    radius=st.integers(min_value=0, max_value=64),
)
def test_mih_matches_linear_scan_any_radius(values: list[int], query: int, radius: int) -> None:
    idx = MultiIndexHash()
    by_id = {f"h{i}": v for i, v in enumerate(values)}
    for ident, v in by_id.items():
        idx.add(v, ident)
    got = set(idx.query(query, radius))
    assert got == _linear_scan(by_id, query, radius)


@settings(max_examples=80, deadline=None)
@given(
    values=st.lists(_U64, min_size=1, max_size=200, unique=True),
    query=_U64,
)
def test_mih_matches_linear_scan_at_production_radius_18(values: list[int], query: int) -> None:
    idx = MultiIndexHash()
    by_id = {f"h{i}": v for i, v in enumerate(values)}
    for ident, v in by_id.items():
        idx.add(v, ident)
    got = set(idx.query(query, 18))
    assert got == _linear_scan(by_id, query, 18)


@settings(max_examples=40, deadline=None)
@given(m=st.sampled_from([1, 2, 4, 8]), seed=st.integers(min_value=0, max_value=10_000))
def test_mih_equivalence_independent_of_m(m: int, seed: int) -> None:
    import random

    rng = random.Random(seed)
    by_id = {f"h{i}": rng.getrandbits(64) for i in range(60)}
    idx = MultiIndexHash(m=m)
    for ident, v in by_id.items():
        idx.add(v, ident)
    query = rng.getrandbits(64)
    for radius in (0, 1, 8, 18, 32):
        assert set(idx.query(query, radius)) == _linear_scan(by_id, query, radius)
