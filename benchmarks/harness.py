"""Run the real detection pipeline over a synthetic corpus and score it.

The harness exercises the production code paths end to end: it computes the four
perceptual hashes (:mod:`optimus.hashing.perceptual`), seeds a BK-tree
:class:`~optimus.services.detection.index.HashIndex` from the campaign bases,
gathers candidates by phash Hamming radius, and scores each candidate with the
real ensemble (:func:`optimus.hashing.ensemble.compare`).

To measure threshold behavior independently of the three fixed sensitivity
presets, the harness records the *minimum ensemble score* each image achieves
against the index (the score the matcher would act on) and then applies a sweep
of raw match thresholds. A lower score means a closer match, so an image is
"flagged" at threshold ``t`` when its best score ``<= t``. The same scores are
also bucketed against the configured presets so the report can recommend an
operating point relative to the shipped default.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from benchmarks.corpus import Corpus, CorpusImage
from optimus.hashing import perceptual
from optimus.hashing.ensemble import PRESETS, compare
from optimus.services.detection.index import HashIndex, KnownHash
from optimus.services.detection.matcher import DEFAULT_CANDIDATE_RADIUS

# A no-match image gets this sentinel score (worse than any real distance, which
# is bounded by 1.0). Keeps thresholding uniform without special-casing None.
NO_MATCH_SCORE = 2.0


@dataclass(frozen=True, slots=True)
class Scored:
    """The best (lowest) ensemble score an image achieved against the index."""

    image: CorpusImage
    best_score: float
    matched_campaign: str | None


def _hashes(img: CorpusImage) -> dict[str, int]:
    """Compute the four perceptual hashes for a corpus image."""
    rgb = np.asarray(img.image.convert("RGB"), dtype=np.uint8)
    return perceptual.compute_all(perceptual.to_grayscale(rgb))


def build_index(corpus: Corpus) -> HashIndex:
    """Build a BK-tree hash index seeded with each campaign's base image."""
    entries: list[KnownHash] = []
    for base in corpus.bases:
        h = _hashes(base)
        entries.append(
            KnownHash(
                hash_id=base.name,
                phash=h["phash"],
                dhash=h["dhash"],
                whash=h["whash"],
                ahash=h["ahash"],
                source="guild",
                campaign_id=base.campaign,
            )
        )
    return HashIndex(entries)


def score_corpus(
    corpus: Corpus,
    *,
    candidate_radius: int = DEFAULT_CANDIDATE_RADIUS,
) -> list[Scored]:
    """Score every corpus image against an index built from the bases.

    For each image we gather BK-tree candidates within ``candidate_radius`` of
    its phash and keep the lowest ensemble score. Images with no candidate get
    :data:`NO_MATCH_SCORE`. The sensitivity passed to :func:`compare` only
    affects the verdict label, not the numeric ``score``, so any preset works
    here; we use the index's natural geometry and threshold afterward.
    """
    index = build_index(corpus)
    # Any preset yields the same `score`; verdict labels are recomputed later.
    sensitivity = next(iter(PRESETS))
    scored: list[Scored] = []
    for img in corpus.all_images():
        h = _hashes(img)
        best_score = NO_MATCH_SCORE
        best_campaign: str | None = None
        for known in index.candidates(h["phash"], candidate_radius):
            result = compare(h, known.as_dict(), sensitivity)
            if result.score < best_score:
                best_score = result.score
                best_campaign = known.campaign_id
        scored.append(Scored(image=img, best_score=best_score, matched_campaign=best_campaign))
    return scored


@dataclass(frozen=True, slots=True)
class ThresholdResult:
    """Confusion-matrix tallies and derived metrics at one match threshold."""

    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 1.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 1.0

    @property
    def fpr(self) -> float:
        denom = self.fp + self.tn
        return self.fp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def evaluate_threshold(scored: Sequence[Scored], threshold: float) -> ThresholdResult:
    """Tally a confusion matrix at one raw match threshold.

    An image is flagged when its best ensemble score is ``<= threshold``. A
    flagged scam is a true positive; a flagged clean is a false positive.
    """
    tp = fp = tn = fn = 0
    for s in scored:
        flagged = s.best_score <= threshold
        if s.image.is_scam:
            if flagged:
                tp += 1
            else:
                fn += 1
        elif flagged:
            fp += 1
        else:
            tn += 1
    return ThresholdResult(threshold=threshold, tp=tp, fp=fp, tn=tn, fn=fn)


def sweep_thresholds(
    scored: Sequence[Scored], thresholds: Sequence[float]
) -> list[ThresholdResult]:
    """Evaluate every threshold in ``thresholds`` (ascending) over ``scored``."""
    return [evaluate_threshold(scored, t) for t in thresholds]


def default_thresholds(steps: int = 25, hi: float = 0.30) -> list[float]:
    """A dense, ascending threshold grid from 0 to ``hi`` inclusive."""
    return [round(i * hi / steps, 4) for i in range(steps + 1)]


def perturbation_recall(
    scored: Sequence[Scored], threshold: float
) -> dict[str, tuple[int, int]]:
    """Per-perturbation (and base) recall at ``threshold``.

    Returns ``{perturbation: (caught, total)}`` for scam images only. The
    ``"base"`` key covers the indexed originals (which should always match).
    """
    tallies: dict[str, list[int]] = {}
    for s in scored:
        if not s.image.is_scam or s.image.perturbation is None:
            continue
        caught, total = tallies.setdefault(s.image.perturbation, [0, 0])
        total += 1
        if s.best_score <= threshold:
            caught += 1
        tallies[s.image.perturbation] = [caught, total]
    return {k: (v[0], v[1]) for k, v in tallies.items()}


@dataclass(frozen=True, slots=True)
class OperatingPoint:
    """A recommended threshold and the result it yields."""

    result: ThresholdResult
    rationale: str


def recommend_operating_point(results: Sequence[ThresholdResult]) -> OperatingPoint:
    """Pick the highest-recall threshold that keeps a zero false-positive rate.

    For an auto-moderation action, a single wrongly-flagged clean image is far
    costlier than a missed re-share, so we prefer the strongest recall among the
    thresholds that flag no clean images. If every threshold has a false
    positive (it should not, given the corpus), we fall back to the best F1.
    """
    zero_fp = [r for r in results if r.fp == 0]
    if zero_fp:
        best = max(zero_fp, key=lambda r: (r.recall, r.threshold))
        return OperatingPoint(
            result=best,
            rationale=(
                f"highest recall ({best.recall:.3f}) at zero false positives; "
                f"score threshold {best.threshold:.4f}"
            ),
        )
    best = max(results, key=lambda r: (r.f1, -r.threshold))
    return OperatingPoint(
        result=best,
        rationale=f"best F1 ({best.f1:.3f}); no zero-FP threshold exists for this corpus",
    )


@dataclass(frozen=True, slots=True)
class PresetResult:
    """Confusion matrix for one shipped sensitivity preset's match threshold."""

    name: str
    match_threshold: float
    ambiguous_ceiling: float
    flagged: ThresholdResult


def preset_results(scored: Sequence[Scored]) -> list[PresetResult]:
    """Evaluate the shipped presets, flagging at the ambiguous ceiling.

    The production matcher flags both ``SCAM`` (score <= match_threshold) and
    ``AMBIGUOUS`` (score <= ambiguous_ceiling) verdicts, so the operating
    threshold the bot actually acts/reviews on is the ambiguous ceiling.
    """
    out: list[PresetResult] = []
    for sens, preset in PRESETS.items():
        ceiling = preset.ambiguous_ceiling
        out.append(
            PresetResult(
                name=str(sens.value),
                match_threshold=preset.match_threshold,
                ambiguous_ceiling=ceiling,
                flagged=evaluate_threshold(scored, ceiling),
            )
        )
    return out
