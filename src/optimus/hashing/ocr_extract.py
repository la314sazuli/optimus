"""OCR text extraction and URL/domain intelligence for scam images."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np

# Official domains of major AI companies — used for lookalike detection.
OFFICIAL_AI_DOMAINS: frozenset[str] = frozenset(
    {
        "perplexity.ai",
        "perplexity.com",
        "openai.com",
        "chatgpt.com",
        "anthropic.com",
        "claude.ai",
        "google.com",
        "deepmind.google",
        "gemini.google.com",
        "midjourney.com",
        "stability.ai",
        "huggingface.co",
        "mistral.ai",
        "x.ai",
        "grok.com",
        "cohere.com",
        "meta.ai",
        "llama.com",
        "replicate.com",
        "together.ai",
    }
)

_URL_RE = re.compile(
    r"https?://[^\s<>'\"]+"
    r"|www\.[^\s<>'\"]+"
    r"|[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.[a-z]{2,}(?:/[^\s<>'\"]*)?",
    re.IGNORECASE,
)


def extract_text(image_bytes: bytes) -> str:
    """Extract text from an image via Tesseract OCR. Returns empty on failure."""
    try:
        import pytesseract

        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return ""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return str(pytesseract.image_to_string(gray)).strip()
    except Exception:
        return ""


def extract_urls(text: str) -> list[str]:
    """Extract URLs from text. Returns deduplicated, ordered list."""
    seen: set[str] = set()
    urls: list[str] = []
    for match in _URL_RE.finditer(text):
        url = match.group().rstrip(".,;:!?)")
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def normalize_domain(url: str) -> str:
    """Extract the bare domain from a URL: lowercase, strip www., punycode."""
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    try:
        netloc = urlparse(url).netloc.lower()
    except Exception:
        return ""
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc.encode("idna").decode("ascii") if netloc else ""


def is_lookalike(domain: str, official: frozenset[str] = OFFICIAL_AI_DOMAINS) -> str | None:
    """Return the official domain this domain is impersonating, or None.

    Detects: exact match (safe), lookalike characters, suspicious
    prefixes/suffixes, and TLD swaps.
    """
    domain = domain.lower().strip()
    if not domain:
        return None

    # Exact match — not a lookalike.
    if domain in official:
        return None

    for real in official:
        if domain.endswith("." + real) or domain == real:
            return None
        real_sld = real.split(".")[0]
        domain_sld = domain.split(".")[0]
        # Suspicious domain's SLD contains the official SLD. Require a
        # minimum length so short SLDs ("x" from x.ai, "meta" from meta.ai)
        # don't match unrelated domains that merely happen to contain them.
        if len(real_sld) >= 5 and real_sld in domain_sld and domain != real:
            return real
        # TLD swap: same SLD, different TLD
        if domain_sld == real_sld and domain != real:
            return real
        # Levenshtein distance <= 2 for short domains
        if _levenshtein(domain, real) <= 2 and len(real) >= 6:
            return real

    return None


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb)))
        prev = curr
    return prev[-1]


def analyze_image(image_bytes: bytes) -> dict[str, Any]:
    """Full analysis: OCR text, URLs, and domain lookalike detection.

    Returns:
        {"text": str, "urls": list[str], "lookalikes": list[dict[str, str]]}
    """
    text = extract_text(image_bytes)
    urls = extract_urls(text)
    lookalikes: list[dict[str, str]] = []
    for url in urls:
        domain = normalize_domain(url)
        if not domain:
            continue
        target = is_lookalike(domain)
        if target:
            lookalikes.append({"domain": domain, "impersonating": target, "url": url})
    return {"text": text, "urls": urls, "lookalikes": lookalikes}
