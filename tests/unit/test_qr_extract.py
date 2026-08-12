"""Tests for QR code extraction."""

import cv2
import numpy as np

from optimus.hashing.qr_extract import extract_qr_urls


def _make_qr_image(data: str) -> bytes:
    encoder = cv2.QRCodeEncoder.create()
    img = encoder.encode(data)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_extract_single_qr():
    payload = "https://scam.example.com/wallet-drain"
    data = _make_qr_image(payload)
    urls = extract_qr_urls(data)
    assert payload in urls


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
