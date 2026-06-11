"""Ingest worker: consume ``message_image.v1``, fetch safely, publish bytes.

Each image URL is fetched through the SSRF guard under a per-guild Redis token
bucket. On success the worker publishes ``image_fetched.v1`` carrying the raw
bytes inline as base64 plus a SHA-256 digest. Bytes are never written to disk.
Fetch failures are swallowed (logged + metric) so a bad URL doesn't poison the
stream — the gateway will simply not see a verdict for it.
"""

from __future__ import annotations

import base64
import hashlib
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from prometheus_client import Counter

from optimus.contracts.events import (
    ImageFetchedEvent,
    MessageImageEvent,
)
from optimus.core.idempotency import build_key
from optimus.core.logging import get_logger
from optimus.core.ratelimit import RateLimit, RateLimiter
from optimus.ingest.fetcher import FetchedImage, FetchError
from optimus.ingest.ssrf import SSRFError

_log = get_logger(__name__)

IMAGES_FETCHED = Counter(
    "optimus_ingest_images_fetched_total",
    "Images successfully fetched and validated.",
)
IMAGES_REJECTED = Counter(
    "optimus_ingest_images_rejected_total",
    "Images rejected during fetch/validation.",
    ["reason"],
)
IMAGES_RATE_LIMITED = Counter(
    "optimus_ingest_rate_limited_total",
    "Fetch attempts dropped by the per-guild rate limiter.",
)

FetchFn = Callable[[str], Awaitable[FetchedImage]]


class IngestWorker:
    """Stateless per-message fetch logic, independent of the bus runtime."""

    def __init__(
        self,
        fetch: FetchFn,
        limiter: RateLimiter,
        *,
        rate: RateLimit,
        ratelimit_prefix: str = "optimus:rl:ingest",
    ) -> None:
        self._fetch = fetch
        self._limiter = limiter
        self._rate = rate
        self._prefix = ratelimit_prefix

    async def handle(self, event: MessageImageEvent) -> ImageFetchedEvent | None:
        """Fetch one image and return the resulting event, or ``None`` to skip.

        A ``None`` return means "permanently drop this image" (rejected/blocked);
        the caller acks. Rate-limited messages raise so the caller can nak for a
        later retry.
        """
        key = f"guild:{event.guild_id}"
        if not await self._limiter.acquire(key, self._rate):
            IMAGES_RATE_LIMITED.inc()
            raise RateLimitedError(event.guild_id)

        try:
            fetched = await self._fetch(event.url)
        except (SSRFError, FetchError) as exc:
            reason = "ssrf" if isinstance(exc, SSRFError) else "fetch"
            IMAGES_REJECTED.labels(reason=reason).inc()
            _log.warning("ingest_rejected", reason=reason, guild_id=event.guild_id)
            return None

        digest = hashlib.sha256(fetched.data).hexdigest()
        IMAGES_FETCHED.inc()
        return ImageFetchedEvent(
            correlation_id=event.correlation_id,
            occurred_at=datetime.now(UTC),
            guild_id=event.guild_id,
            channel_id=event.channel_id,
            message_id=event.message_id,
            attachment_id=event.attachment_id,
            uploader_id=event.uploader_id,
            idempotency_key=build_key(event.message_id, event.attachment_id),
            content_type=fetched.content_type,
            size_bytes=len(fetched.data),
            sha256=digest,
            data_b64=base64.b64encode(fetched.data).decode("ascii"),
        )


class RateLimitedError(Exception):
    """Raised when a per-guild fetch budget is exhausted (retry later)."""

    def __init__(self, guild_id: int) -> None:
        super().__init__(f"rate limited for guild {guild_id}")
        self.guild_id = guild_id
