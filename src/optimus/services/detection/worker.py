"""Detection worker: decode -> hash -> match -> (swarm) -> verdict.

Stateless per-image logic, independent of the bus runtime. Idempotency is
checked first so retries never re-emit. Decode failures yield a NON_DECISION
(the pipeline never acts on an image it could not safely read). The guild
whitelist is consulted before any scam match. Swarm correlation may escalate a
positive verdict one confidence band and produce a ``swarm_alert``.
"""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from prometheus_client import Counter

from optimus.contracts.events import (
    HashSet,
    ImageFetchedEvent,
    SwarmAlertEvent,
    Verdict,
    VerdictEvent,
)
from optimus.core.config import Sensitivity
from optimus.core.logging import get_logger
from optimus.hashing import perceptual
from optimus.hashing.decoder import DecodedImage, DecodeLimits, decode
from optimus.services.detection.index import HashIndex
from optimus.services.detection.matcher import (
    MatchOutcome,
    WhitelistEntry,
    escalate_band,
    match,
)
from optimus.services.detection.swarm import SwarmCorrelator

_log = get_logger(__name__)

VERDICTS_EMITTED = Counter(
    "optimus_detection_verdicts_total",
    "Verdicts emitted by the detection worker.",
    ["verdict"],
)
DUPLICATE_SKIPPED = Counter(
    "optimus_detection_duplicate_skipped_total",
    "Images skipped because their idempotency key was already claimed.",
)


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """The worker's output for one image: a verdict plus optional swarm alert."""

    verdict: VerdictEvent
    swarm_alert: SwarmAlertEvent | None = None


# Async hooks the worker depends on, injected for testability.
GuildIndexFn = Callable[[int], Awaitable[HashIndex]]
GlobalIndexFn = Callable[[], Awaitable[HashIndex]]
WhitelistFn = Callable[[int], Awaitable[list[WhitelistEntry]]]
SensitivityFn = Callable[[int], Awaitable[Sensitivity]]
IdempotencyAcquire = Callable[[str], Awaitable[bool]]


def best_frame_hashes(image: DecodedImage) -> dict[str, int]:
    """Compute hashes for the most distinctive frame (max phash variance proxy).

    For a single-frame image this is just that frame. For animations we hash the
    first sampled frame; callers that need true best-frame matching iterate via
    :func:`all_frame_hashes`.
    """
    return perceptual.compute_all(image.frames[0])


def all_frame_hashes(image: DecodedImage) -> list[dict[str, int]]:
    """Compute the hash set for every sampled frame."""
    return [perceptual.compute_all(frame) for frame in image.frames]


class DetectionWorker:
    """Per-image detection logic with injected index/whitelist/idempotency hooks."""

    def __init__(
        self,
        *,
        guild_index: GuildIndexFn,
        global_index: GlobalIndexFn,
        whitelist: WhitelistFn,
        sensitivity: SensitivityFn,
        idempotency_acquire: IdempotencyAcquire,
        swarm: SwarmCorrelator | None = None,
        limits: DecodeLimits | None = None,
        use_embedding: bool = False,
    ) -> None:
        self._guild_index = guild_index
        self._global_index = global_index
        self._whitelist = whitelist
        self._sensitivity = sensitivity
        self._acquire = idempotency_acquire
        self._swarm = swarm
        self._limits = limits
        self._use_embedding = use_embedding

    async def handle(self, event: ImageFetchedEvent) -> DetectionResult | None:
        """Process one fetched image; return a verdict, or ``None`` if a duplicate."""
        if not await self._acquire(event.idempotency_key):
            DUPLICATE_SKIPPED.inc()
            return None

        data = base64.b64decode(event.data_b64)
        # Decode (subprocess wall-wait) and perceptual hashing (numpy/Python,
        # up to max_frames frames) are both blocking and CPU/IO-bound. Run them
        # off the event loop so the NATS consumer loop and health server stay
        # responsive while one image is processed.
        frames = await asyncio.to_thread(self._decode_and_hash, data)
        if frames is None:
            return DetectionResult(verdict=self._verdict(event, _non_decision()))

        guild_idx = await self._guild_index(event.guild_id)
        global_idx = await self._global_index()
        whitelist = await self._whitelist(event.guild_id)
        sensitivity = await self._sensitivity(event.guild_id)

        outcome = self._best_frame_outcome(frames, guild_idx, global_idx, whitelist, sensitivity)
        primary = frames[0]

        swarm_alert: SwarmAlertEvent | None = None
        if (
            self._swarm is not None
            and not outcome.whitelisted
            and outcome.verdict in (Verdict.SCAM, Verdict.AMBIGUOUS)
        ):
            obs = await self._swarm.observe(primary["phash"], event.guild_id)
            if obs.is_swarming:
                new_verdict, new_conf = escalate_band(outcome.verdict, outcome.confidence)
                outcome = MatchOutcome(
                    verdict=new_verdict,
                    confidence=new_conf,
                    matched_hash_id=outcome.matched_hash_id,
                    campaign_id=outcome.campaign_id,
                    distances=outcome.distances,
                )
                swarm_alert = SwarmAlertEvent(
                    correlation_id=event.correlation_id,
                    occurred_at=datetime.now(UTC),
                    phash=primary["phash"],
                    distinct_guilds=obs.distinct_guilds,
                    window_seconds=self._swarm.window_seconds,
                    sample_guild_ids=[event.guild_id],
                )

        verdict_event = self._verdict(event, outcome, hashes=primary)
        VERDICTS_EMITTED.labels(verdict=verdict_event.verdict.value).inc()
        return DetectionResult(verdict=verdict_event, swarm_alert=swarm_alert)

    def _decode_and_hash(self, data: bytes) -> list[dict[str, int]] | None:
        """Decode image bytes and hash every sampled frame (blocking; off-loop)."""
        image = decode(data, self._limits)
        if image is None:
            return None
        return all_frame_hashes(image)

    def _best_frame_outcome(
        self,
        frames: list[dict[str, int]],
        guild_idx: HashIndex,
        global_idx: HashIndex,
        whitelist: list[WhitelistEntry],
        sensitivity: Sensitivity,
    ) -> MatchOutcome:
        """Match every frame and keep the most incriminating outcome."""
        best: MatchOutcome | None = None
        rank = {Verdict.SCAM: 0, Verdict.AMBIGUOUS: 1, Verdict.CLEAN: 2, Verdict.NON_DECISION: 3}
        for candidate in frames:
            outcome = match(
                candidate,
                guild_index=guild_idx,
                global_index=global_idx,
                whitelist=whitelist,
                sensitivity=sensitivity,
            )
            if outcome.whitelisted:
                return outcome  # whitelist always wins, immediately
            if best is None or rank[outcome.verdict] < rank[best.verdict]:
                best = outcome
        return best if best is not None else MatchOutcome(verdict=Verdict.CLEAN, confidence=1.0)

    def _verdict(
        self,
        event: ImageFetchedEvent,
        outcome: MatchOutcome,
        *,
        hashes: dict[str, int] | None = None,
    ) -> VerdictEvent:
        hash_set = (
            HashSet(
                phash=hashes["phash"],
                dhash=hashes["dhash"],
                whash=hashes["whash"],
                ahash=hashes["ahash"],
            )
            if hashes is not None
            else None
        )
        return VerdictEvent(
            correlation_id=event.correlation_id,
            occurred_at=datetime.now(UTC),
            guild_id=event.guild_id,
            channel_id=event.channel_id,
            message_id=event.message_id,
            attachment_id=event.attachment_id,
            uploader_id=event.uploader_id,
            idempotency_key=event.idempotency_key,
            verdict=outcome.verdict,
            confidence=outcome.confidence,
            hashes=hash_set,
            matched_hash_id=outcome.matched_hash_id,
            distances=outcome.distances,
        )


def _non_decision() -> MatchOutcome:
    return MatchOutcome(verdict=Verdict.NON_DECISION, confidence=0.0)
