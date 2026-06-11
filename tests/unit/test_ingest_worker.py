"""Unit tests for the ingest worker (rate limiting + fetch outcomes)."""

from __future__ import annotations

import base64
from datetime import UTC, datetime

import pytest

from optimus.contracts.events import MessageImageEvent
from optimus.core.ratelimit import InMemoryRateLimiter, RateLimit
from optimus.ingest.fetcher import FetchedImage, FetchError
from optimus.ingest.ssrf import SSRFError
from optimus.services.ingest.worker import IngestWorker, RateLimitedError

DATA = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


def _event(guild_id: int = 1, url: str = "https://cdn.test/a.png") -> MessageImageEvent:
    return MessageImageEvent(
        correlation_id="c",
        occurred_at=datetime.now(UTC),
        guild_id=guild_id,
        channel_id=2,
        message_id=3,
        attachment_id=4,
        uploader_id=5,
        url=url,
        filename="a.png",
    )


def _worker(fetch, *, capacity: float = 10.0, refill: float = 10.0) -> IngestWorker:  # type: ignore[no-untyped-def]
    return IngestWorker(
        fetch, InMemoryRateLimiter(), rate=RateLimit(capacity=capacity, refill_rate=refill)
    )


async def test_handle_success_publishes_event() -> None:
    async def fetch(url: str) -> FetchedImage:
        return FetchedImage(data=DATA, content_type="image/png", final_url=url)

    worker = _worker(fetch)
    result = await worker.handle(_event())
    assert result is not None
    assert result.content_type == "image/png"
    assert base64.b64decode(result.data_b64) == DATA
    assert result.size_bytes == len(DATA)
    assert len(result.sha256) == 64


async def test_handle_ssrf_returns_none() -> None:
    async def fetch(_url: str) -> FetchedImage:
        raise SSRFError("blocked")

    assert await _worker(fetch).handle(_event()) is None


async def test_handle_fetch_error_returns_none() -> None:
    async def fetch(_url: str) -> FetchedImage:
        raise FetchError("bad")

    assert await _worker(fetch).handle(_event()) is None


async def test_handle_rate_limited_raises() -> None:
    async def fetch(url: str) -> FetchedImage:
        return FetchedImage(data=DATA, content_type="image/png", final_url=url)

    # Capacity 1, no refill within the test window: second call is limited.
    worker = _worker(fetch, capacity=1.0, refill=0.001)
    assert await worker.handle(_event()) is not None
    with pytest.raises(RateLimitedError):
        await worker.handle(_event())


async def test_rate_limit_is_per_guild() -> None:
    async def fetch(url: str) -> FetchedImage:
        return FetchedImage(data=DATA, content_type="image/png", final_url=url)

    worker = _worker(fetch, capacity=1.0, refill=0.001)
    assert await worker.handle(_event(guild_id=1)) is not None
    # A different guild has its own bucket.
    assert await worker.handle(_event(guild_id=2)) is not None
