"""Fetch + hash a Discord attachment for command-driven hash operations.

``/scamhash add image:<attachment>`` and the message-review flow both need to
turn a live Discord attachment URL into the same ``phash``/``dhash``/``whash``
triple (plus mirror hashes) that the passive detection pipeline computes for
messages it observes directly. This module is that missing glue: it reuses the
existing SSRF-hardened fetcher, sandboxed decoder, and perceptual hash
functions verbatim, so a hash added this way is bit-for-bit comparable to one
the live pipeline would have produced from the same image.

Kept free of hikari/aiohttp session lifecycle concerns -- callers inject an
already-configured fetch function, matching the pattern used by
:class:`optimus.services.ingest.worker.IngestWorker`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from optimus.hashing import perceptual
from optimus.hashing.decoder import DecodeLimits, decode
from optimus.hashing.ocr_extract import analyze_image
from optimus.hashing.qr_extract import extract_qr_urls
from optimus.ingest.fetcher import FetchedImage, FetchError
from optimus.ingest.ssrf import SSRFError


class AttachmentHashError(Exception):
    """Raised when an attachment could not be safely fetched, decoded, or hashed."""


@dataclass(frozen=True, slots=True)
class AttachmentHashes:
    """The full hash set for one successfully hashed attachment."""

    attachment_id: int
    url: str
    phash: int
    dhash: int
    whash: int
    ahash: int
    mphash: int
    mdhash: int
    mwhash: int
    mahash: int
    qr_urls: list[str] = field(default_factory=list)
    ocr_lookalikes: list[dict[str, str]] = field(default_factory=list)
    ocr_signals: list[str] = field(default_factory=list)
    ocr_risk_level: str = "none"


#: Matches IngestWorker.FetchFn: an async URL -> FetchedImage fetch, already
#: bound to whatever SSRF-guarded aiohttp session the caller runs.
FetchFn = Callable[[str], Awaitable[FetchedImage]]


async def hash_attachment(
    fetch: FetchFn,
    *,
    attachment_id: int,
    url: str,
    limits: DecodeLimits | None = None,
) -> AttachmentHashes:
    """Fetch ``url`` and compute its full (including mirror) hash set.

    Uses only the first sampled frame (consistent with how a single manually
    supplied image is treated -- unlike the passive pipeline, which scores
    every frame of an animation independently via
    :func:`optimus.services.detection.worker.all_frame_hashes`).

    Raises :class:`AttachmentHashError` for any fetch, decode, or validation
    failure; callers should surface this as a user-facing command error
    rather than letting it propagate as an unhandled exception.
    """
    try:
        fetched = await fetch(url)
    except (SSRFError, FetchError) as exc:
        raise AttachmentHashError(f"could not fetch attachment: {exc}") from exc
    return await asyncio.to_thread(
        _hash_fetched_attachment,
        fetched.data,
        attachment_id=attachment_id,
        url=url,
        limits=limits,
    )


def _hash_fetched_attachment(
    data: bytes,
    *,
    attachment_id: int,
    url: str,
    limits: DecodeLimits | None,
) -> AttachmentHashes:
    """CPU/subprocess image intelligence performed outside the event loop."""
    decoded = decode(data, limits)
    if decoded is None or not decoded.frames:
        raise AttachmentHashError("attachment could not be decoded as a supported image")

    frame = decoded.frames[0]
    direct = perceptual.compute_all(frame)
    mirror = perceptual.compute_all_mirror(frame)
    analysis = analyze_image(data)
    qr_urls = extract_qr_urls(data)
    return AttachmentHashes(
        attachment_id=attachment_id,
        url=url,
        phash=direct["phash"],
        dhash=direct["dhash"],
        whash=direct["whash"],
        ahash=direct["ahash"],
        mphash=mirror["phash"],
        mdhash=mirror["dhash"],
        mwhash=mirror["whash"],
        mahash=mirror["ahash"],
        qr_urls=qr_urls,
        ocr_lookalikes=analysis["lookalikes"],
        ocr_signals=analysis["signals"],
        ocr_risk_level=analysis["risk_level"],
    )
