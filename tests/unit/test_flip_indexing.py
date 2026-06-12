"""Behavioral tests for flip-invariant indexing and the candidate-radius bump.

These exercise the real perceptual pipeline and matcher: a horizontally
mirrored re-share of a known scam must match back to the *same* source
(hash_id/campaign), and the widened candidate radius must admit the crop-band
distances the eval harness identified.
"""

from __future__ import annotations

import numpy as np

from optimus.contracts.events import Verdict
from optimus.core.config import Sensitivity
from optimus.hashing import perceptual
from optimus.services.detection.index import HashIndex, KnownHash
from optimus.services.detection.matcher import DEFAULT_CANDIDATE_RADIUS, match


def _scam_gray(seed: int = 11, h: int = 64, w: int = 96) -> perceptual.FloatArray:
    """A deterministic, structured grayscale image standing in for a scam banner."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 255, w)
    base = np.tile(x, (h, 1))
    base[8:22, 5:45] = 20.0  # a distinctive asymmetric banner block
    base += rng.normal(0, 6, (h, w))
    return np.clip(base, 0, 255)


def _known_with_mirror(gray: perceptual.FloatArray, hash_id: str = "camp-flip") -> KnownHash:
    h = perceptual.compute_all(gray)
    return KnownHash(
        hash_id=hash_id,
        phash=h["phash"],
        dhash=h["dhash"],
        whash=h["whash"],
        ahash=h["ahash"],
        source="guild",
        campaign_id=hash_id,
        mirror=perceptual.compute_all_mirror(gray),
    )


def test_flip_helper_matches_recompute_of_flipped_image() -> None:
    gray = _scam_gray()
    direct = perceptual.compute_all(np.fliplr(gray))
    assert perceptual.compute_all_mirror(gray) == direct


def test_flipped_reshare_matches_back_to_same_source() -> None:
    gray = _scam_gray()
    index = HashIndex([_known_with_mirror(gray)])

    flipped_upload = perceptual.compute_all(perceptual.flip_horizontal(gray))
    outcome = match(
        flipped_upload,
        guild_index=index,
        global_index=HashIndex([]),
        whitelist=[],
        sensitivity=Sensitivity.STRICT,
    )

    assert outcome.verdict is Verdict.SCAM
    # Dedup/ownership: the mirror sibling resolves to the original source.
    assert outcome.matched_hash_id == "camp-flip"
    assert outcome.campaign_id == "camp-flip"


def test_original_still_matches_when_mirror_indexed() -> None:
    gray = _scam_gray()
    index = HashIndex([_known_with_mirror(gray)])
    upload = perceptual.compute_all(gray)
    outcome = match(
        upload,
        guild_index=index,
        global_index=HashIndex([]),
        whitelist=[],
        sensitivity=Sensitivity.STRICT,
    )
    assert outcome.verdict is Verdict.SCAM
    assert outcome.matched_hash_id == "camp-flip"


def test_len_counts_sources_not_mirror_siblings() -> None:
    gray = _scam_gray()
    # Two distinct sources, each contributing a mirror sibling internally.
    index = HashIndex(
        [
            _known_with_mirror(gray, hash_id="a"),
            _known_with_mirror(_scam_gray(seed=99), hash_id="b"),
        ]
    )
    assert len(index) == 2


def test_flip_not_caught_without_mirror() -> None:
    gray = _scam_gray()
    h = perceptual.compute_all(gray)
    # No mirror supplied -> mirrored re-share should not match (control).
    index = HashIndex(
        [
            KnownHash(
                hash_id="camp-flip",
                phash=h["phash"],
                dhash=h["dhash"],
                whash=h["whash"],
                ahash=h["ahash"],
                source="guild",
                campaign_id="camp-flip",
            )
        ]
    )
    flipped_upload = perceptual.compute_all(perceptual.flip_horizontal(gray))
    outcome = match(
        flipped_upload,
        guild_index=index,
        global_index=HashIndex([]),
        whitelist=[],
        sensitivity=Sensitivity.STRICT,
    )
    assert outcome.verdict is Verdict.CLEAN


def test_candidate_radius_default_is_eighteen() -> None:
    assert DEFAULT_CANDIDATE_RADIUS == 18


def test_crop_band_distance_admitted_only_at_new_radius() -> None:
    # A candidate exactly 15 bits from the known phash (the crop band the eval
    # harness flagged): outside the old radius 12, inside the new radius 18.
    base_phash = 0x0F0F_0F0F_0F0F_0F0F
    flip_mask = (1 << 15) - 1  # flip the low 15 bits -> Hamming distance 15
    near = base_phash ^ flip_mask
    assert perceptual.hamming(base_phash, near) == 15

    known = KnownHash(
        hash_id="c",
        phash=base_phash,
        dhash=0,
        whash=0,
        ahash=0,
        source="guild",
    )
    index = HashIndex([known])
    assert index.candidates(near, 12) == []
    assert [k.hash_id for k in index.candidates(near, 18)] == ["c"]
