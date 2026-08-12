"""Sandboxed image decoding.

Untrusted image bytes are decoded in a separate, resource-limited subprocess so
a decompression bomb or a malicious decoder cannot exhaust the worker. The child
applies CPU/memory rlimits and a Pillow pixel cap; the parent enforces a wall
clock timeout. Any failure yields a *non-decision* (``None``) — the pipeline
never acts on an image it could not safely decode.

Perceptual hashing only ever consumes a small (<=32x32) reduction of each
frame (see :mod:`optimus.hashing.perceptual`), so the child decodes straight
to a small thumbnail -- via JPEG's native ``draft()`` decode-time downscale
where available, and a post-decode ``thumbnail()`` resize otherwise -- rather
than ever materializing a full-resolution pixel buffer. Memory use is
therefore roughly constant regardless of the source image's resolution.
"""

from __future__ import annotations

import base64
import json
import os
import resource
import subprocess
import sys
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from optimus.core.logging import get_logger

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class DecodeLimits:
    """Resource limits applied to the decode subprocess."""

    cpu_seconds: int = 5
    #: JPEG (the common case for real photos/screenshots) decodes straight to
    #: a small thumbnail via draft() and needs well under 256MB regardless of
    #: source resolution. Formats without a draft-equivalent (PNG/GIF/WebP/
    #: BMP) have no way to avoid a full-resolution decode buffer before the
    #: post-decode thumbnail() resize can run, so this limit is sized for
    #: the worst measured case across supported formats at the configured
    #: max_image_pixels cap instead: WebP measured as the most memory-hungry
    #: format at 24MP, needing ~500-550MB at minimum even with BLAS threading
    #: pinned to 1 (see _CHILD_ENV_OVERRIDES) -- libwebp's own decode working
    #: set, not a numpy/BLAS artifact. 768MB keeps headroom above that floor.
    #: Note: without pinning BLAS/OMP threads to 1, this same number was
    #: intermittently insufficient on hosts with a high detected CPU count
    #: (production hit "OpenBLAS error: Memory allocation still failed after
    #: 10 retries" at this exact limit) -- OpenBLAS sizes its thread pool
    #: against the host's CPU count, not the container's cgroup quota, and
    #: that pool's own allocation can consume a large, host-dependent chunk
    #: of this budget before any image work happens. This never reproduced
    #: in local testing on a 2-vCPU sandbox, which is what let it ship
    #: initially -- the thread pool overhead is invisible until the host has
    #: enough cores to make it large.
    mem_bytes: int = 768 * 1024 * 1024
    wall_timeout: float = 5.0
    max_image_pixels: int = 24_000_000
    max_frames: int = 8
    #: Side length (pixels) the child decodes/resizes each frame down to
    #: before returning it. Perceptual hashing only ever consumes up to a
    #: 32x32 reduction (see optimus.hashing.perceptual), so decoding to 4x
    #: that gives _resize_mean real pixels to area-average from without ever
    #: holding a full-resolution frame in memory.
    hash_side: int = 128


@dataclass(frozen=True, slots=True)
class DecodedImage:
    """A decoded image as sampled grayscale frames."""

    #: One or more HxW float64 luminance frames (>=1; >1 only for animations).
    frames: list[npt.NDArray[np.float64]]
    width: int
    height: int
    frame_count: int


def _apply_rlimits(limits: DecodeLimits) -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_seconds, limits.cpu_seconds + 1))
    resource.setrlimit(resource.RLIMIT_AS, (limits.mem_bytes, limits.mem_bytes))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))


#: OpenBLAS (numpy's backing library for the tiny matrix-vector product in
#: luminance()) sizes its thread pool against the *host's* detected CPU count,
#: not the container's cgroup CPU quota -- on a host with many cores this
#: pool's own allocation can be tens to hundreds of MB before a single pixel
#: is touched, which is pure waste for an operation this small (at most
#: 128x128x3 elements) and gains nothing from parallelism. This surfaced as
#: "OpenBLAS error: Memory allocation still failed after 10 retries" in
#: production even after the mem_bytes budget was sized generously against
#: locally-measured decode costs alone. Forcing every BLAS/OMP thread pool to
#: a single thread removes that variable host-dependent overhead entirely.
_CHILD_ENV_OVERRIDES = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update(_CHILD_ENV_OVERRIDES)
    return env


# Child program: reads JSON {data, max_pixels, max_frames, hash_side} on
# stdin, writes JSON {frames:[base64 float32 LE], width, height, frame_count}
# on stdout. Every frame is shrunk down to at most hash_side x hash_side
# *during* decode (JPEG) or immediately after (all other formats) so peak
# memory never scales with the source image's resolution -- perceptual
# hashing only ever consumes a <=32x32 reduction of each frame, so nothing
# downstream needs (or benefits from) full-resolution pixels.
_CHILD_SOURCE = r"""
import sys, json, base64, io
import numpy as np
from PIL import Image, ImageSequence

def shrink(im, side):
    # draft() only applies to JPEG/MPO and must run before any pixel access;
    # it asks the decoder itself to emit a downscaled DCT output, so a 6000x4000
    # JPEG never exists at full resolution anywhere in this process. Harmless
    # no-op (returns None) for formats/images it can't accelerate.
    im.draft("RGB", (side, side))
    im.load()
    # Safety net for formats draft() can't touch (PNG/GIF/WebP/BMP) and for
    # JPEGs draft() only partially reduced: a further resize down to the exact
    # target. Cheap once draft has already done the heavy lifting; for other
    # formats this is the only reduction, bounded by the max_pixels decode cap
    # enforced by Image.MAX_IMAGE_PIXELS below.
    if im.width > side or im.height > side:
        # thumbnail() mutates and resizes in place -- no defensive copy needed
        # here, since each frame object is either freshly produced by the
        # sequence iterator or the sole reference to a non-animated image.
        # Skipping the copy avoids briefly holding two full decode buffers at
        # once, which matters at this memory budget.
        im.thumbnail((side, side), Image.Resampling.BOX)
    return im

def luminance(im, side):
    small = shrink(im, side)
    # float32 (not float64) for the RGB cast: halves peak memory for this
    # intermediate with no meaningful precision loss for 0-255 luminance
    # weights, and the caller re-serializes to float32 anyway (see the frames
    # output below).
    arr = np.asarray(small.convert("RGB"), dtype=np.float32)
    w = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return arr @ w

req = json.load(sys.stdin)
Image.MAX_IMAGE_PIXELS = int(req["max_pixels"])
max_frames = int(req["max_frames"])
hash_side = int(req["hash_side"])
raw = base64.b64decode(req["data"])
im = Image.open(io.BytesIO(raw))
# Capture the TRUE original dimensions before draft()/thumbnail() touch
# im.size -- draft() rewrites im.size in place to the reduced size the moment
# it runs, so this must happen first and be read from the file header alone.
w, h = im.size

frames = []
total = getattr(im, "n_frames", 1)
if total > 1:
    step = max(1, total // max_frames)
    for idx, frame in enumerate(ImageSequence.Iterator(im)):
        if idx % step != 0:
            continue
        frames.append(luminance(frame, hash_side))
        if len(frames) >= max_frames:
            break
if not frames:
    frames.append(luminance(im, hash_side))

out = {
    "frames": [base64.b64encode(f.astype("<f4").tobytes()).decode() for f in frames],
    "shapes": [list(f.shape) for f in frames],
    "width": int(w),
    "height": int(h),
    "frame_count": int(total),
}
json.dump(out, sys.stdout)
"""


def decode(data: bytes, limits: DecodeLimits | None = None) -> DecodedImage | None:
    """Decode ``data`` in a sandboxed subprocess.

    Returns a :class:`DecodedImage`, or ``None`` on any decode/limit failure
    (a non-decision).
    """
    lim = limits or DecodeLimits()
    request = json.dumps(
        {
            "data": base64.b64encode(data).decode("ascii"),
            "max_pixels": lim.max_image_pixels,
            "max_frames": lim.max_frames,
            "hash_side": lim.hash_side,
        }
    )
    try:
        proc = subprocess.run(  # noqa: S603 - fixed interpreter + inline source, no shell
            [sys.executable, "-c", _CHILD_SOURCE],
            input=request.encode("utf-8"),
            capture_output=True,
            timeout=lim.wall_timeout,
            preexec_fn=lambda: _apply_rlimits(lim),
            env=_child_env(),
            check=True,
        )
    except subprocess.TimeoutExpired:
        _log.warning("decode_timeout")
        return None
    except subprocess.CalledProcessError as exc:
        # Surface the child's stderr (e.g. a numpy MemoryError, a Pillow
        # DecompressionBombError) -- without this, every decode rejection
        # looked identical from the logs alone, which made a systemic issue
        # like an undersized memory limit indistinguishable from routine
        # per-image corruption.
        stderr_tail = (exc.stderr or b"").decode("utf-8", errors="replace")[-500:]
        _log.warning("decode_failed", returncode=exc.returncode, stderr=stderr_tail)
        return None
    except Exception as exc:
        _log.warning("decode_error", reason=str(exc))
        return None

    try:
        payload = json.loads(proc.stdout)
        frames: list[npt.NDArray[np.float64]] = []
        for b64, shape in zip(payload["frames"], payload["shapes"], strict=True):
            buf = base64.b64decode(b64)
            arr = np.frombuffer(buf, dtype="<f4").astype(np.float64).reshape(shape)
            frames.append(arr)
    except Exception as exc:
        _log.warning("decode_unpack_failed", reason=str(exc))
        return None

    if not frames:
        return None
    return DecodedImage(
        frames=frames,
        width=int(payload["width"]),
        height=int(payload["height"]),
        frame_count=int(payload["frame_count"]),
    )
