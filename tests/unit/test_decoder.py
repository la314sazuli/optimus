"""Tests for the sandboxed image decoder."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from optimus.hashing import perceptual as ph
from optimus.hashing.decoder import DecodeLimits, decode


def _png_bytes(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def test_decode_returns_grayscale_frame() -> None:
    rng = np.random.default_rng(3)
    rgb = rng.integers(0, 256, (64, 80, 3), dtype=np.uint8)
    result = decode(_png_bytes(rgb))
    assert result is not None
    assert result.width == 80
    assert result.height == 64
    assert len(result.frames) == 1
    assert result.frames[0].shape == (64, 80)


def test_decode_garbage_is_non_decision() -> None:
    assert decode(b"not an image at all") is None


def test_decode_respects_pixel_cap() -> None:
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    limits = DecodeLimits(max_image_pixels=100)  # 10_000 pixels exceeds cap
    assert decode(_png_bytes(rgb), limits) is None


def test_decode_animated_samples_multiple_frames() -> None:
    frames = [
        Image.fromarray((np.full((32, 32, 3), v, dtype=np.uint8)), mode="RGB")
        for v in (10, 120, 240, 60)
    ]
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:], duration=50, loop=0)
    result = decode(buf.getvalue(), DecodeLimits(max_frames=4))
    assert result is not None
    assert result.frame_count >= 2
    assert len(result.frames) >= 2


def test_decoded_frame_hashes() -> None:
    rng = np.random.default_rng(9)
    rgb = rng.integers(0, 256, (96, 96, 3), dtype=np.uint8)
    result = decode(_png_bytes(rgb))
    assert result is not None
    hashes = ph.compute_all(result.frames[0])
    for v in hashes.values():
        assert 0 <= v < (1 << 64)


@pytest.mark.parametrize("timeout", [0.000001])
def test_decode_timeout_is_non_decision(timeout: float) -> None:
    rgb = np.zeros((512, 512, 3), dtype=np.uint8)
    assert decode(_png_bytes(rgb), DecodeLimits(wall_timeout=timeout)) is None


def _jpeg_bytes(size: tuple[int, int]) -> bytes:
    img = Image.new("RGB", size, color=(50, 80, 120))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def test_decode_succeeds_at_realistic_phone_camera_resolution() -> None:
    """Regression test for a real production failure.

    A 3024x4032 image (a common modern phone camera / Discord screenshot
    resolution) previously failed to decode under the default limits: the
    child's luminance step cast the full-resolution RGB buffer to float64,
    and that intermediate alone (~576MB for this image size) blew past the
    old 512MB RLIMIT_AS before the image could even be hashed -- silently,
    with no indication to the caller *why* it failed. Every attachment on a
    reviewed message failed identically regardless of image content, since
    the failure was structural (memory budget), not content-dependent.

    Fixed by decoding straight to a small thumbnail (see
    ``test_decode_shrinks_large_frames_before_hashing`` for the memory-shape
    contract this relies on) rather than by raising the memory ceiling alone.
    """
    result = decode(_jpeg_bytes((3024, 4032)))
    assert result is not None
    assert result.width == 3024
    assert result.height == 4032


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP", "BMP"])
def test_decode_succeeds_at_configured_pixel_cap(fmt: str) -> None:
    """The true worst case across every supported format: an image right at
    ``max_image_pixels`` (24MP) must still fit within ``mem_bytes``. WebP
    measured as the most memory-hungry format at this size (no draft()
    equivalent, larger decode working set than PNG) and drove the default
    ``mem_bytes`` value -- this is parametrized across formats so a future
    default that's sized against only one format's profile fails loudly.
    """
    img = Image.new("RGB", (6000, 4000), color=(50, 80, 120))  # exactly 24,000,000 px
    buf = io.BytesIO()
    img.save(buf, format=fmt, **({"quality": 90} if fmt in ("JPEG", "WEBP") else {}))
    result = decode(buf.getvalue())
    assert result is not None
    assert result.width == 6000
    assert result.height == 4000


def test_decode_shrinks_large_frames_before_hashing() -> None:
    """Frames are decoded down to at most ``hash_side`` per side regardless of
    source resolution -- perceptual hashing only ever consumes a <=32x32
    reduction (see :mod:`optimus.hashing.perceptual`), so nothing downstream
    needs, or should ever receive, a full-resolution frame. This is the
    behavioral contract that keeps decode memory roughly constant instead of
    scaling with the source image's resolution.
    """
    result = decode(_jpeg_bytes((6000, 4000)), DecodeLimits(hash_side=64))
    assert result is not None
    # Reported dimensions must still reflect the true original size...
    assert result.width == 6000
    assert result.height == 4000
    # ...while the actual pixel data handed to the hash functions is bounded
    # by hash_side on both axes.
    h, w = result.frames[0].shape
    assert h <= 64
    assert w <= 64


def test_decode_does_not_upscale_small_images() -> None:
    """An image already smaller than ``hash_side`` passes through unchanged
    rather than being upscaled -- shrink() only acts when a dimension exceeds
    the target.
    """
    result = decode(_jpeg_bytes((50, 50)))
    assert result is not None
    assert result.frames[0].shape == (50, 50)
