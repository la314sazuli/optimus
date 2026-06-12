"""Synthetic image payloads for the soak traffic lanes.

Four families, all PNG-encoded inline bytes exactly as ingest hands them to
detection:

* **scam** — a fixed high-entropy noise image whose four perceptual hashes are
  registered as a campaign, so a re-upload matches at distance 0.
* **transformed** — the scam image with a light JPEG round-trip / small crop, the
  near-duplicate a real re-poster produces; should still land inside the match
  radius.
* **clean** — independent high-entropy noise, far outside any candidate radius.
* **hostile** — corrupt/truncated bytes, decompression-bomb-shaped dimensions,
  wrong content types, and zero-byte payloads.

The hostile builders deliberately produce inputs that must be *rejected or
resolved as NON_DECISION* without affecting the process.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

SCAM_SEED = 7
_SIZE = 64


def _noise_png(seed: int, size: int = _SIZE) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG")
    return buf.getvalue()


def scam_png() -> bytes:
    """The canonical scam image (its hashes are the registered campaign)."""
    return _noise_png(SCAM_SEED)


def transformed_png(variant: int) -> bytes:
    """A near-duplicate of the scam image: re-encode + a couple of pixel tweaks.

    Keeps the perceptual hashes within match distance while making the bytes (and
    therefore the SHA-256 / idempotency key) distinct, mirroring a re-poster who
    re-saves the same scam.
    """
    rng = np.random.default_rng(SCAM_SEED)
    arr = rng.integers(0, 256, (_SIZE, _SIZE, 3), dtype=np.uint8)
    # Perturb a handful of pixels — far too few to move a perceptual hash, but
    # enough to change the encoded bytes so it is not a byte-identical dup.
    tweak = np.random.default_rng(10_000 + variant)
    for _ in range(3):
        y = int(tweak.integers(0, _SIZE))
        x = int(tweak.integers(0, _SIZE))
        arr[y, x] = (arr[y, x].astype(int) + 7) % 256
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG")
    return buf.getvalue()


def clean_png(variant: int) -> bytes:
    """Independent noise that should never match the campaign."""
    return _noise_png(1_000_000 + variant)


# --- hostile lane ----------------------------------------------------------------


def hostile_truncated() -> tuple[bytes, str]:
    """A valid PNG header followed by a hard cut mid-stream."""
    full = scam_png()
    return full[: len(full) // 3], "image/png"


def hostile_corrupt() -> tuple[bytes, str]:
    """Random bytes wearing a PNG magic number — decoder must reject."""
    rng = np.random.default_rng(424242)
    body = rng.integers(0, 256, 2048, dtype=np.uint8).tobytes()
    return b"\x89PNG\r\n\x1a\n" + body, "image/png"


def hostile_zero_byte() -> tuple[bytes, str]:
    """An empty payload."""
    return b"", "image/png"


def hostile_wrong_content_type() -> tuple[bytes, str]:
    """Plain text masquerading as an image."""
    return b"this is not an image, it is a wall of text " * 8, "text/plain"


def hostile_decompression_bomb() -> tuple[bytes, str]:
    """A PNG declaring enormous dimensions (decompression-bomb shaped).

    Built by hand-writing an IHDR with a huge width/height so the decoder's
    pixel-count guard trips *before* any allocation, rather than encoding a real
    giant image (which would itself OOM the soak driver). The body after IHDR is
    deliberately junk; a well-behaved decoder rejects on the pixel cap or the
    broken stream, never by allocating the declared buffer.
    """
    import struct
    import zlib

    def _chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    sig = b"\x89PNG\r\n\x1a\n"
    # 60000 x 60000 = 3.6e9 pixels, far over max_image_pixels (24e6).
    ihdr = struct.pack(">IIBBBBB", 60000, 60000, 8, 2, 0, 0, 0)
    return sig + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", b"\x00" * 16) + _chunk(b"IEND", b""), (
        "image/png"
    )


HOSTILE_BUILDERS = (
    hostile_truncated,
    hostile_corrupt,
    hostile_zero_byte,
    hostile_wrong_content_type,
    hostile_decompression_bomb,
)
