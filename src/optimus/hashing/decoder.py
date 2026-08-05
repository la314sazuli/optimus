"""Sandboxed image decoding.

Untrusted image bytes are decoded in a separate, resource-limited subprocess so
a decompression bomb or a malicious decoder cannot exhaust the worker. The child
applies CPU/memory rlimits and a Pillow pixel cap; the parent enforces a wall
clock timeout. Any failure yields a *non-decision* (``None``) — the pipeline
never acts on an image it could not safely decode.
"""

from __future__ import annotations

import base64
import json
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
    #: Sized for the worst case at max_image_pixels: decoding a 24MP image needs
    #: a uint8 RGB buffer (~72MB) plus a float32 RGB cast for luminance
    #: (~288MB) plus the resulting float32 luminance frame (~96MB) plus
    #: Pillow's own JPEG decode working set (~100MB) plus Python/numpy/Pillow
    #: import baseline (~100MB) -- roughly 650MB with the float32 luminance
    #: path below, or nearly double that with the float64 path this replaced.
    #: The previous 512MB limit silently failed to decode any image near a
    #: modern phone camera's real resolution. 1.5GB keeps comfortable headroom
    #: above the calculated worst case on either path.
    mem_bytes: int = 1536 * 1024 * 1024
    wall_timeout: float = 5.0
    max_image_pixels: int = 24_000_000
    max_frames: int = 8


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


# Child program: reads JSON {data, max_pixels, max_frames} on stdin, writes JSON
# {frames:[base64 float32 LE], width, height, frame_count} on stdout.
_CHILD_SOURCE = r"""
import sys, json, base64, io
import numpy as np
from PIL import Image, ImageSequence

def luminance(im):
    # float32 (not float64) for the RGB cast: halves peak memory for the
    # largest intermediate in this pipeline with no meaningful precision loss
    # for 0-255 luminance weights, and the caller re-serializes to float32
    # anyway (see the frames output below).
    arr = np.asarray(im.convert("RGB"), dtype=np.float32)
    w = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return arr @ w

req = json.load(sys.stdin)
Image.MAX_IMAGE_PIXELS = int(req["max_pixels"])
max_frames = int(req["max_frames"])
raw = base64.b64decode(req["data"])
im = Image.open(io.BytesIO(raw))
im.load()
w, h = im.size

frames = []
total = getattr(im, "n_frames", 1)
if total > 1:
    step = max(1, total // max_frames)
    for idx, frame in enumerate(ImageSequence.Iterator(im)):
        if idx % step != 0:
            continue
        frames.append(luminance(frame))
        if len(frames) >= max_frames:
            break
if not frames:
    frames.append(luminance(im))

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
        }
    )
    try:
        proc = subprocess.run(  # noqa: S603 - fixed interpreter + inline source, no shell
            [sys.executable, "-c", _CHILD_SOURCE],
            input=request.encode("utf-8"),
            capture_output=True,
            timeout=lim.wall_timeout,
            preexec_fn=lambda: _apply_rlimits(lim),
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
