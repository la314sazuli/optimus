"""Pure matching logic: whitelist-first, then ensemble vote over candidates.

Separated from the bus/decoder runtime so the decision rules are unit-testable
with plain hash sets. The whitelist always wins: a candidate whose phash is
within the whitelist radius is forced CLEAN regardless of any scam match.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from optimus.contracts.events import Verdict
from optimus.core.config import Sensitivity
from optimus.hashing.ensemble import EnsembleResult, compare
from optimus.hashing.perceptual import hamming
from optimus.services.detection.index import HashIndex, KnownHash

# phash Hamming radius used to gather index candidates before the full vote.
# Raised 12 -> 18 after the eval harness showed heavy border crops land at phash
# distance 12-16 from the base: radius 18 recovers them (crop recall 33% -> 100%,
# overall recall 0.792 -> 0.875) with zero precision/FPR loss. See
# docs/detection-eval.md.
DEFAULT_CANDIDATE_RADIUS = 18
# phash radius within which a whitelist entry suppresses a match.
DEFAULT_WHITELIST_RADIUS = 8


@dataclass(frozen=True, slots=True)
class WhitelistEntry:
    """A guild whitelist phash (always overrides matches)."""

    phash: int


@dataclass(frozen=True, slots=True)
class MatchOutcome:
    """The result of matching one image's hashes against the indexes."""

    verdict: Verdict
    confidence: float
    matched_hash_id: str | None = None
    campaign_id: str | None = None
    distances: dict[str, int] = field(default_factory=dict)
    whitelisted: bool = False

    @property
    def is_ambiguous(self) -> bool:
        """Whether the outcome needs embedding confirmation."""
        return self.verdict is Verdict.AMBIGUOUS


def is_whitelisted(
    phash: int, whitelist: list[WhitelistEntry], *, radius: int = DEFAULT_WHITELIST_RADIUS
) -> bool:
    """Whether ``phash`` is within ``radius`` of any whitelist entry."""
    return any(hamming(phash, w.phash) <= radius for w in whitelist)


def _best_over_indexes(
    candidate: dict[str, int],
    indexes: list[HashIndex],
    sensitivity: Sensitivity,
    *,
    radius: int,
) -> tuple[EnsembleResult | None, KnownHash | None]:
    """Return the strongest ensemble result across all indexes (lowest score)."""
    best: EnsembleResult | None = None
    best_known: KnownHash | None = None
    for index in indexes:
        for known in index.candidates(candidate["phash"], radius):
            result = compare(candidate, known.as_dict(), sensitivity)
            if best is None or result.score < best.score:
                best, best_known = result, known
    return best, best_known


def match(
    candidate: dict[str, int],
    *,
    guild_index: HashIndex,
    global_index: HashIndex,
    whitelist: list[WhitelistEntry],
    sensitivity: Sensitivity = Sensitivity.BALANCED,
    candidate_radius: int = DEFAULT_CANDIDATE_RADIUS,
    whitelist_radius: int = DEFAULT_WHITELIST_RADIUS,
) -> MatchOutcome:
    """Decide a verdict for ``candidate`` (whitelist wins, then ensemble vote)."""
    if is_whitelisted(candidate["phash"], whitelist, radius=whitelist_radius):
        return MatchOutcome(verdict=Verdict.CLEAN, confidence=1.0, whitelisted=True)

    result, known = _best_over_indexes(
        candidate, [guild_index, global_index], sensitivity, radius=candidate_radius
    )
    if result is None or known is None:
        return MatchOutcome(verdict=Verdict.CLEAN, confidence=1.0)

    return MatchOutcome(
        verdict=result.verdict,
        confidence=result.confidence,
        matched_hash_id=known.hash_id if result.verdict is not Verdict.CLEAN else None,
        campaign_id=known.campaign_id if result.verdict is not Verdict.CLEAN else None,
        distances=result.distances,
    )


def escalate_band(verdict: Verdict, confidence: float) -> tuple[Verdict, float]:
    """Escalate a verdict one confidence band (used on swarm correlation).

    CLEAN -> AMBIGUOUS -> SCAM. Confidence is bumped toward certainty; SCAM and
    NON_DECISION are returned unchanged.
    """
    if verdict is Verdict.AMBIGUOUS:
        return Verdict.SCAM, max(confidence, 0.75)
    if verdict is Verdict.CLEAN:
        return Verdict.AMBIGUOUS, max(confidence, 0.5)
    return verdict, confidence
