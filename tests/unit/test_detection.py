"""Unit tests for detection matching, worker logic, and the index manager."""

from __future__ import annotations

import base64
from datetime import UTC, datetime
from unittest import mock

from optimus.contracts.events import ImageFetchedEvent, Verdict
from optimus.core.config import Sensitivity
from optimus.hashing.decoder import DecodeLimits
from optimus.services.detection.index import HashIndex, KnownHash
from optimus.services.detection.matcher import (
    MatchOutcome,
    WhitelistEntry,
    escalate_band,
    is_whitelisted,
    match,
)
from optimus.services.detection.swarm import SwarmObservation
from optimus.services.detection.worker import DetectionWorker

# A known scam hash set and an exact-match candidate.
KNOWN = KnownHash(
    hash_id="camp-1",
    phash=0xFFFF_FFFF_0000_0000,
    dhash=0x0F0F_0F0F_0F0F_0F0F,
    whash=0x00FF_00FF_00FF_00FF,
    ahash=0x1234_5678_9ABC_DEF0,
    source="guild",
    campaign_id="camp-1",
)
CANDIDATE = KNOWN.as_dict()
EMPTY = HashIndex([])


def _guild_index() -> HashIndex:
    return HashIndex([KNOWN])


# --- matcher ---------------------------------------------------------------


def test_exact_match_is_scam() -> None:
    outcome = match(
        CANDIDATE,
        guild_index=_guild_index(),
        global_index=EMPTY,
        whitelist=[],
        sensitivity=Sensitivity.BALANCED,
    )
    assert outcome.verdict is Verdict.SCAM
    assert outcome.matched_hash_id == "camp-1"
    assert outcome.campaign_id == "camp-1"


def test_no_candidates_is_clean() -> None:
    far = {"phash": 0, "dhash": 0, "whash": 0, "ahash": 0}
    outcome = match(far, guild_index=_guild_index(), global_index=EMPTY, whitelist=[])
    assert outcome.verdict is Verdict.CLEAN
    assert outcome.matched_hash_id is None


def test_whitelist_always_wins() -> None:
    # Candidate matches a scam exactly, but its phash is whitelisted.
    wl = [WhitelistEntry(phash=CANDIDATE["phash"])]
    outcome = match(
        CANDIDATE,
        guild_index=_guild_index(),
        global_index=EMPTY,
        whitelist=wl,
        sensitivity=Sensitivity.STRICT,
    )
    assert outcome.verdict is Verdict.CLEAN
    assert outcome.whitelisted is True


def test_guild_whitelist_overrides_global_match() -> None:
    # Regression: a per-guild whitelist must suppress a hash promoted to the
    # GLOBAL index, not just guild-local entries. The candidate exactly matches
    # a global scam hash, yet the guild's whitelist forces CLEAN.
    wl = [WhitelistEntry(phash=CANDIDATE["phash"])]
    outcome = match(
        CANDIDATE,
        guild_index=EMPTY,
        global_index=_guild_index(),
        whitelist=wl,
        sensitivity=Sensitivity.STRICT,
    )
    assert outcome.verdict is Verdict.CLEAN
    assert outcome.whitelisted is True
    assert outcome.matched_hash_id is None


def test_is_whitelisted_radius() -> None:
    wl = [WhitelistEntry(phash=0x0)]
    assert is_whitelisted(0x1, wl, radius=4)
    assert not is_whitelisted(0xFFFF_FFFF_FFFF_FFFF, wl, radius=4)


def test_escalate_band() -> None:
    assert escalate_band(Verdict.CLEAN, 0.1) == (Verdict.AMBIGUOUS, 0.5)
    assert escalate_band(Verdict.AMBIGUOUS, 0.1) == (Verdict.SCAM, 0.75)
    assert escalate_band(Verdict.SCAM, 0.9) == (Verdict.SCAM, 0.9)
    assert escalate_band(Verdict.NON_DECISION, 0.0) == (Verdict.NON_DECISION, 0.0)


def test_global_index_also_matched() -> None:
    outcome = match(
        CANDIDATE,
        guild_index=EMPTY,
        global_index=_guild_index(),
        whitelist=[],
        sensitivity=Sensitivity.BALANCED,
    )
    assert outcome.verdict is Verdict.SCAM


# --- worker (idempotency, decode, swarm) -----------------------------------


def _event(*, key: str = "k1", data: bytes = b"\x89PNG\r\n\x1a\n") -> ImageFetchedEvent:
    return ImageFetchedEvent(
        correlation_id="c",
        occurred_at=datetime.now(UTC),
        guild_id=1,
        channel_id=2,
        message_id=3,
        attachment_id=4,
        uploader_id=5,
        idempotency_key=key,
        content_type="image/png",
        size_bytes=len(data),
        sha256="0" * 64,
        data_b64=base64.b64encode(data).decode(),
    )


class _OnceGuard:
    """Idempotency hook that grants a key exactly once."""

    def __init__(self) -> None:
        self._seen: set[str] = set()

    async def acquire(self, key: str) -> bool:
        if key in self._seen:
            return False
        self._seen.add(key)
        return True


class _AlwaysSwarm:
    """Swarm stub that reports swarming on every observation."""

    window_seconds = 300

    async def observe(self, phash: int, guild_id: int) -> SwarmObservation:
        return SwarmObservation(distinct_guilds=5, is_swarming=True)


def _worker(
    *,
    guild_index: HashIndex,
    swarm: object | None = None,
    cls: type[DetectionWorker] = DetectionWorker,
) -> DetectionWorker:
    async def gi(_gid: int) -> HashIndex:
        return guild_index

    async def gx() -> HashIndex:
        return EMPTY

    async def wl(_gid: int) -> list[WhitelistEntry]:
        return []

    async def sens(_gid: int) -> Sensitivity:
        return Sensitivity.BALANCED

    guard = _OnceGuard()
    return cls(
        guild_index=gi,
        global_index=gx,
        whitelist=wl,
        sensitivity=sens,
        idempotency_acquire=guard.acquire,
        swarm=swarm,  # type: ignore[arg-type]
    )


async def test_worker_idempotency_skips_duplicate() -> None:
    worker = _worker(guild_index=EMPTY)
    first = await worker.handle(_event(key="dup"))
    assert first is not None
    second = await worker.handle(_event(key="dup"))
    assert second is None


async def test_worker_non_decodable_is_non_decision() -> None:
    worker = _worker(guild_index=EMPTY)
    result = await worker.handle(_event(key="bad", data=b"not-an-image"))
    assert result is not None
    assert result.verdict.verdict is Verdict.NON_DECISION


async def test_worker_malformed_base64_resolves_not_raises() -> None:
    # A corrupt inline payload must resolve as a NON_DECISION (acked), never
    # raise — raising would nak and redeliver the same poison until max_deliver.
    from optimus.services.detection.worker import PAYLOAD_REJECTED

    worker = _worker(guild_index=EMPTY)
    ev = _event(key="poison")
    object.__setattr__(ev, "data_b64", "!!!not base64!!!")
    before = PAYLOAD_REJECTED.labels(reason="decode")._value.get()
    result = await worker.handle(ev)
    assert result is not None
    assert result.verdict.verdict is Verdict.NON_DECISION
    after = PAYLOAD_REJECTED.labels(reason="decode")._value.get()
    assert after - before == 1


def _scam_png() -> bytes:
    import io

    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(7)
    arr = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG")
    return buf.getvalue()


def _worker_with_limits(limits: DecodeLimits) -> DetectionWorker:
    async def gi(_gid: int) -> HashIndex:
        return EMPTY

    async def gx() -> HashIndex:
        return EMPTY

    async def wl(_gid: int) -> list[WhitelistEntry]:
        return []

    async def sens(_gid: int) -> Sensitivity:
        return Sensitivity.BALANCED

    return DetectionWorker(
        guild_index=gi,
        global_index=gx,
        whitelist=wl,
        sensitivity=sens,
        idempotency_acquire=_OnceGuard().acquire,
        limits=limits,
    )


async def test_worker_decompression_bomb_resolves_to_non_decision() -> None:
    # A 100x100 image under a 100-pixel cap: Pillow's MAX_IMAGE_PIXELS guard in
    # the sandboxed child trips before a full decode, and the worker resolves it
    # as a NON_DECISION (acked) rather than raising (which would nak-loop).
    worker = _worker_with_limits(DecodeLimits(max_image_pixels=100))
    result = await worker.handle(_event(key="bomb", data=_scam_png()))
    assert result is not None
    assert result.verdict.verdict is Verdict.NON_DECISION


async def test_worker_decode_timeout_resolves_to_non_decision() -> None:
    # A wall-clock timeout in the decode subprocess must also resolve, not raise.
    worker = _worker_with_limits(DecodeLimits(wall_timeout=0.000001))
    result = await worker.handle(_event(key="slow", data=_scam_png()))
    assert result is not None
    assert result.verdict.verdict is Verdict.NON_DECISION


async def test_worker_normal_image_still_flows() -> None:
    # Regression: a normal in-bounds image decodes and produces a real verdict.
    worker = _worker_with_limits(DecodeLimits())
    result = await worker.handle(_event(key="normal", data=_scam_png()))
    assert result is not None
    assert result.verdict.verdict in (Verdict.CLEAN, Verdict.AMBIGUOUS, Verdict.SCAM)
    assert result.verdict.hashes is not None


async def test_worker_swarm_escalates_and_alerts() -> None:
    from optimus.hashing import perceptual
    from optimus.hashing.decoder import decode

    data = _scam_png()
    decoded = decode(data)
    assert decoded is not None
    hashes = perceptual.compute_all(decoded.frames[0])
    known = KnownHash(
        hash_id="c-swarm",
        phash=hashes["phash"],
        dhash=hashes["dhash"],
        whash=hashes["whash"],
        ahash=hashes["ahash"],
        source="guild",
        campaign_id="c-swarm",
    )
    worker = _worker(guild_index=HashIndex([known]), swarm=_AlwaysSwarm())
    result = await worker.handle(_event(key="swarm", data=data))
    assert result is not None
    assert result.swarm_alert is not None
    assert result.swarm_alert.distinct_guilds == 5
    # An exact match (SCAM) stays SCAM; escalation only lifts a lower band.
    assert result.verdict.verdict is Verdict.SCAM


async def test_worker_offloads_decode_and_hash_to_thread() -> None:
    # The decode subprocess wait and numpy hashing must not run on the event
    # loop; the worker dispatches them via asyncio.to_thread.
    import asyncio
    import threading

    calls: list[object] = []
    real_to_thread = asyncio.to_thread

    async def spy(func: object, /, *args: object, **kwargs: object) -> object:
        calls.append(func)
        return await real_to_thread(func, *args, **kwargs)  # type: ignore[arg-type]

    # Behavioral check: record the thread the decode+hash actually runs on and
    # assert it is *not* the event-loop thread, so a regression that inlined the
    # work on the loop would fail even if it still routed through to_thread.
    loop_thread = threading.get_ident()
    ran_on: list[int] = []

    class _ThreadProbe(DetectionWorker):
        def _decode_and_hash(self, data: bytes):  # type: ignore[override]
            ran_on.append(threading.get_ident())
            return super()._decode_and_hash(data)

    worker = _worker(guild_index=EMPTY, cls=_ThreadProbe)
    with mock.patch.object(asyncio, "to_thread", spy):
        result = await worker.handle(_event(key="offload", data=_scam_png()))
    assert result is not None
    assert calls, "decode+hash work was not offloaded via asyncio.to_thread"
    assert calls[0] == worker._decode_and_hash
    assert ran_on, "decode+hash did not run"
    assert ran_on[0] != loop_thread, "decode+hash ran on the event-loop thread"


async def test_worker_swarm_escalates_ambiguous_to_scam() -> None:
    class _AmbiguousMatcher(DetectionWorker):
        def _best_frame_outcome(self, *a: object, **k: object) -> MatchOutcome:  # type: ignore[override]
            return MatchOutcome(verdict=Verdict.AMBIGUOUS, confidence=0.4)

    async def gi(_gid: int) -> HashIndex:
        return EMPTY

    async def gx() -> HashIndex:
        return EMPTY

    async def wl(_gid: int) -> list[WhitelistEntry]:
        return []

    async def sens(_gid: int) -> Sensitivity:
        return Sensitivity.BALANCED

    worker = _AmbiguousMatcher(
        guild_index=gi,
        global_index=gx,
        whitelist=wl,
        sensitivity=sens,
        idempotency_acquire=_OnceGuard().acquire,
        swarm=_AlwaysSwarm(),  # type: ignore[arg-type]
    )
    result = await worker.handle(_event(key="amb", data=_scam_png()))
    assert result is not None
    assert result.verdict.verdict is Verdict.SCAM
    assert result.swarm_alert is not None
