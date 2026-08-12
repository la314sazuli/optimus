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

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from optimus.hashing import perceptual
from optimus.hashing.decoder import DecodeLimits, decode
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

    decoded = decode(fetched.data, limits)
    if decoded is None or not decoded.frames:
        raise AttachmentHashError("attachment could not be decoded as a supported image")

    frame = decoded.frames[0]
    direct = perceptual.compute_all(frame)
    mirror = perceptual.compute_all_mirror(frame)

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
    )
