"""In-house perceptual hashes (aHash, dHash, pHash, wHash) as 64-bit integers.

Each hash reduces an image to a 64-bit fingerprint robust to re-encoding and
minor edits. Implemented directly on numpy arrays (DCT via matrix multiply) and
PyWavelets (Haar) so there is no dependency on an external imagehash library.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

HASH_BITS = 64
_SIDE = 8  # 8x8 = 64 bits

FloatArray = npt.NDArray[np.float64]


def _bits_to_uint64(bits: npt.NDArray[np.bool_]) -> int:
    """Pack a flat boolean array (MSB first) into a 64-bit unsigned int."""
    flat = bits.flatten()
    if flat.size != HASH_BITS:
        raise ValueError(f"expected {HASH_BITS} bits, got {flat.size}")
    value = 0
    for bit in flat:
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    """Hamming distance between two 64-bit hashes."""
    if a < 0 or b < 0 or a >= (1 << HASH_BITS) or b >= (1 << HASH_BITS):
        raise ValueError("hashes must be unsigned 64-bit integers")
    return int((a ^ b).bit_count())


def _dct_matrix(n: int) -> FloatArray:
    """Return the orthonormal DCT-II basis matrix of size ``n``."""
    k = np.arange(n).reshape(-1, 1)
    x = np.arange(n).reshape(1, -1)
    m: FloatArray = np.cos(np.pi * (2 * x + 1) * k / (2 * n))
    m *= np.sqrt(2.0 / n)
    m[0, :] *= 1.0 / np.sqrt(2.0)
    return m


_DCT32 = _dct_matrix(32)


def ahash(gray: FloatArray) -> int:
    """Average hash: bits set where the pixel exceeds the 8x8 mean."""
    reduced = _resize_mean(gray, _SIDE, _SIDE)
    return _bits_to_uint64(reduced > reduced.mean())


def dhash(gray: FloatArray) -> int:
    """Difference hash: bits set where each pixel is brighter than its right neighbor."""
    reduced = _resize_mean(gray, _SIDE, _SIDE + 1)
    diff = reduced[:, 1:] > reduced[:, :-1]
    return _bits_to_uint64(diff)


def phash(gray: FloatArray) -> int:
    """Perceptual hash: low-frequency DCT coefficients vs. their median."""
    reduced = _resize_mean(gray, 32, 32)
    dct = _DCT32 @ reduced @ _DCT32.T
    low = dct[:_SIDE, :_SIDE]
    # Exclude the DC term (0,0) from the median so it doesn't dominate.
    med = np.median(low.flatten()[1:])
    return _bits_to_uint64(low > med)


def whash(gray: FloatArray) -> int:
    """Wavelet hash: Haar approximation coefficients vs. their median."""
    import pywt

    reduced = _resize_mean(gray, 32, 32)
    coeffs = reduced
    # Two levels of Haar DWT -> 8x8 approximation band.
    for _ in range(2):
        coeffs, _details = pywt.dwt2(coeffs, "haar")
    approx = np.asarray(coeffs, dtype=np.float64)[:_SIDE, :_SIDE]
    med = np.median(approx)
    return _bits_to_uint64(approx > med)


def _resize_mean(gray: FloatArray, out_h: int, out_w: int) -> FloatArray:
    """Area-average resize of a 2-D grayscale array to ``out_h`` x ``out_w``.

    Uses block reduction when downsampling by an integer factor, otherwise a
    bilinear-style gather. Deterministic and dependency-free for stability.
    """
    in_h, in_w = gray.shape
    if (in_h, in_w) == (out_h, out_w):
        return gray.astype(np.float64, copy=False)

    row_idx = (np.arange(out_h) * in_h / out_h).astype(int)
    col_idx = (np.arange(out_w) * in_w / out_w).astype(int)
    row_edges = np.append(row_idx, in_h)
    col_edges = np.append(col_idx, in_w)

    out = np.empty((out_h, out_w), dtype=np.float64)
    for i in range(out_h):
        r0, r1 = row_idx[i], max(row_edges[i + 1], row_idx[i] + 1)
        for j in range(out_w):
            c0, c1 = col_idx[j], max(col_edges[j + 1], col_idx[j] + 1)
            out[i, j] = gray[r0:r1, c0:c1].mean()
    return out


def to_grayscale(rgb: npt.NDArray[np.uint8]) -> FloatArray:
    """Convert an HxWx3 uint8 RGB array to a 2-D float64 luminance array."""
    if rgb.ndim == 2:
        return rgb.astype(np.float64)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("expected an HxWx3 RGB array")
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float64)
    gray: FloatArray = rgb[:, :, :3].astype(np.float64) @ weights
    return gray


def compute_all(gray: FloatArray) -> dict[str, int]:
    """Compute all four hashes for a grayscale array."""
    return {
        "ahash": ahash(gray),
        "dhash": dhash(gray),
        "phash": phash(gray),
        "whash": whash(gray),
    }


def flip_horizontal(gray: FloatArray) -> FloatArray:
    """Return ``gray`` mirrored left-to-right.

    Perceptual hashes are not flip-invariant by construction, so the mirror hash
    must be derived from the actually-mirrored pixels rather than a bit
    permutation of the original hash (which area-resize and the DCT/median do not
    preserve cleanly). Indexing ``compute_all(flip_horizontal(gray))`` lets a
    mirrored re-share match its source at zero distance.
    """
    return np.fliplr(gray)


def compute_all_mirror(gray: FloatArray) -> dict[str, int]:
    """Compute the four hashes of the horizontally-mirrored image."""
    return compute_all(flip_horizontal(gray))
