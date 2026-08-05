"""Tests for the attachment fetch+hash glue used by ``/scamhash add image:`` and
the message-review command."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from optimus.hashing import perceptual as ph
from optimus.ingest.fetcher import FetchedImage, FetchError
from optimus.ingest.ssrf import SSRFError
from optimus.services.interactions.attachment_hash import (
    AttachmentHashError,
    hash_attachment,
)


def _png_bytes(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def _fetch_ok(data: bytes) -> object:
    async def _fetch(url: str) -> FetchedImage:
        return FetchedImage(data=data, content_type="image/png", final_url=url)

    return _fetch


def _fetch_raises(exc: Exception) -> object:
    async def _fetch(url: str) -> FetchedImage:
        raise exc

    return _fetch


@pytest.mark.asyncio
async def test_hash_attachment_matches_direct_perceptual_computation() -> None:
    rng = np.random.default_rng(7)
    rgb = rng.integers(0, 256, (64, 80, 3), dtype=np.uint8)
    data = _png_bytes(rgb)

    result = await hash_attachment(_fetch_ok(data), attachment_id=42, url="https://x/i.png")

    assert result.attachment_id == 42
    assert result.url == "https://x/i.png"
    # Cross-check against the decoder + perceptual pipeline directly, so this
    # test would fail if hash_attachment ever diverged from the exact frame
    # the passive detection pipeline would have hashed.
    from optimus.hashing.decoder import decode

    decoded = decode(data)
    assert decoded is not None
    direct = ph.compute_all(decoded.frames[0])
    mirror = ph.compute_all_mirror(decoded.frames[0])
    assert result.phash == direct["phash"]
    assert result.dhash == direct["dhash"]
    assert result.whash == direct["whash"]
    assert result.ahash == direct["ahash"]
    assert result.mphash == mirror["phash"]
    assert result.mdhash == mirror["dhash"]
    assert result.mwhash == mirror["whash"]
    assert result.mahash == mirror["ahash"]


@pytest.mark.asyncio
async def test_hash_attachment_wraps_fetch_error() -> None:
    with pytest.raises(AttachmentHashError):
        await hash_attachment(
            _fetch_raises(FetchError("too large")), attachment_id=1, url="https://x/i.png"
        )


@pytest.mark.asyncio
async def test_hash_attachment_wraps_ssrf_error() -> None:
    with pytest.raises(AttachmentHashError):
        await hash_attachment(
            _fetch_raises(SSRFError("blocked host")), attachment_id=1, url="https://x/i.png"
        )


@pytest.mark.asyncio
async def test_hash_attachment_rejects_undecodable_data() -> None:
    with pytest.raises(AttachmentHashError):
        await hash_attachment(
            _fetch_ok(b"not an image at all"), attachment_id=1, url="https://x/i.png"
        )
