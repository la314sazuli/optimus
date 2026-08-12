"""QR code extraction from images using OpenCV's built-in detector."""

from __future__ import annotations

import logging

import cv2  # type: ignore[import-not-found]
import numpy as np

_log = logging.getLogger(__name__)


def extract_qr_urls(image_bytes: bytes) -> list[str]:
    """Decode QR codes from raw image bytes. Returns decoded strings (URLs/text)."""
    try:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return []
        detector = cv2.QRCodeDetector()
        data, _points, _ = detector.detectAndDecode(img)
        if data:
            return [data]
        # Try multi-detect for images with several QR codes
        ok, decoded, _, _ = detector.detectAndDecodeMulti(img)
        if ok:
            return [d for d in decoded if d]
    except Exception:
        _log.warning("qr_extract_failed", exc_info=True)
    return []
