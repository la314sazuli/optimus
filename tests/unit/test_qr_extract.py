"""Tests for QR code extraction wrapper."""

import cv2
import numpy as np

from optimus.hashing.qr_extract import extract_qr_urls


def test_no_qr_returns_empty():
    arr = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", arr)
    assert ok
    urls = extract_qr_urls(buf.tobytes())
    assert urls == []


def test_invalid_bytes_returns_empty():
    urls = extract_qr_urls(b"not an image")
    assert urls == []


def test_empty_bytes_returns_empty():
    urls = extract_qr_urls(b"")
    assert urls == []


def test_corrupt_png_returns_empty():
    urls = extract_qr_urls(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    assert urls == []
