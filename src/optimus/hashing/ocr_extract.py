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

# Phishing signal patterns — keyword groups common in scam images.
# Each tuple: (category, regex). Matched category is surfaced to moderators.
_PHISHING_SIGNALS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("free_offer", re.compile(r"\b(free|airdrop|giveaway|reward|bonus)\b", re.I)),
    ("claim", re.compile(r"\b(claim|redeem|collect|get\s+your)\b", re.I)),
    ("urgency", re.compile(r"\b(limited|expires?|act\s+now|last\s+chance|hurry)\b", re.I)),
    (
        "credentials",
        re.compile(
            r"\b(login|sign\s+in|password|api\s+key|connect\s+wallet|verify\s+account)\b", re.I
        ),
    ),
    (
        "ai_community",
        re.compile(
            r"\b(sora|gpt-?5|claude\s+beta|free\s+pro|perplexity\s+pro|midjourney\s+free)\b", re.I
        ),
    ),
    ("impersonation", re.compile(r"\b(official|support|admin|staff|team|moderator)\b", re.I)),
)

_URL_RE = re.compile(
    r"https?://[^\s<>'\"]+"
    r"|www\.[^\s<>'\"]+"
    r"|[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.[a-z]{2,}(?:/[^\s<>'\"]*)?",
    re.IGNORECASE,
)

# OCR defang patterns — scammers break URLs to evade detection.
_DEFANG_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"hxxps?://", re.I), "https://"),
    (re.compile(r"\s*\[\.\]\s*"), "."),
    (re.compile(r"\s*\(\.\)\s*"), "."),
    (re.compile(r"\s+dot\s+", re.I), "."),
    (re.compile(r"\s*\.\s*"), "."),
    (re.compile(r"\s+"), ""),
)


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

_MAX_OCR_DIM = 4000  # skip OCR on absurdly large images


def _preprocess_variants(img: np.ndarray) -> list[np.ndarray]:
    """Build 2-3 deterministic preprocessing variants for multi-pass OCR.

    Tesseract accuracy varies with image quality; running multiple
    variants and deduplicating the text catches more real content.
    """
    h, w = img.shape[:2]
    if max(h, w) > _MAX_OCR_DIM:
        scale = _MAX_OCR_DIM / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    # Upscale small images for better OCR.
    if max(gray.shape) < 1000:
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    variants: list[np.ndarray] = [gray]

    # CLAHE contrast enhancement — helps with low-contrast screenshots.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    variants.append(clahe.apply(gray))

    # Otsu binary — helps with text on noisy/photo backgrounds.
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu)

    return variants


# ---------------------------------------------------------------------------
# OCR with confidence filtering
# ---------------------------------------------------------------------------

_MIN_CONFIDENCE = 50  # drop tokens below this confidence


def _ocr_confident_text(img: np.ndarray) -> str:
    """Run OCR with per-word confidence filtering via image_to_data."""
    import pytesseract

    data = pytesseract.image_to_data(
        img, output_type=pytesseract.Output.DICT, config="--oem 3 --psm 6"
    )
    lines: dict[int, list[str]] = {}
    for i, conf in enumerate(data["conf"]):
        if int(conf) >= _MIN_CONFIDENCE and data["text"][i].strip():
            line_num = data["line_num"][i]
            lines.setdefault(line_num, []).append(data["text"][i])
    return "\n".join(" ".join(words) for words in lines.values()).strip()


def _ocr_simple(img: np.ndarray) -> str:
    """Fallback: plain image_to_string without confidence data."""
    import pytesseract

    return str(pytesseract.image_to_string(img, config="--oem 3 --psm 11")).strip()


def extract_text(image_bytes: bytes) -> str:
    """Extract text from an image via multi-pass Tesseract OCR.

    Runs 2-3 preprocessing variants, deduplicates results, and filters
    low-confidence tokens. Returns empty on any failure.
    """
    try:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return ""
        variants = _preprocess_variants(img)
        seen: set[str] = set()
        combined: list[str] = []
        for v in variants:
            try:
                text = _ocr_confident_text(v)
            except Exception:
                text = _ocr_simple(v)
            if text and text not in seen:
                seen.add(text)
                combined.append(text)
        return "\n".join(combined).strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# URL repair — fix OCR-broken defanged URLs
# ---------------------------------------------------------------------------


def _repair_urls(text: str) -> str:
    """Repair common OCR artifacts in URLs that scammers use to evade detection.

    Handles: hxxps://, [.] , (.), dot, and stray spaces in domains.
    """
    result = text
    for pattern, replacement in _DEFANG_REPLACEMENTS:
        result = pattern.sub(replacement, result)
    return result


# ---------------------------------------------------------------------------
# Phishing signal detection
# ---------------------------------------------------------------------------


def find_phishing_signals(text: str) -> list[str]:
    """Scan OCR text for common scam keyword patterns. Returns matched categories."""
    matched: list[str] = []
    seen: set[str] = set()
    for category, pattern in _PHISHING_SIGNALS:
        if pattern.search(text) and category not in seen:
            seen.add(category)
            matched.append(category)
    return matched


# ---------------------------------------------------------------------------
# URL extraction and domain analysis
# ---------------------------------------------------------------------------


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

    if domain in official:
        return None

    for real in official:
        if domain.endswith("." + real) or domain == real:
            return None
        real_sld = real.split(".")[0]
        domain_sld = domain.split(".")[0]
        if len(real_sld) >= 5 and real_sld in domain_sld and domain != real:
            return real
        if domain_sld == real_sld and domain != real:
            return real
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
    """Full analysis: OCR text, URLs, lookalike domains, and phishing signals.

    Returns:
        {"text": str, "urls": list[str], "lookalikes": list[dict[str, str]],
         "signals": list[str]}
    """
    text = extract_text(image_bytes)
    repaired = _repair_urls(text)
    urls = extract_urls(repaired)
    lookalikes: list[dict[str, str]] = []
    for url in urls:
        domain = normalize_domain(url)
        if not domain:
            continue
        target = is_lookalike(domain)
        if target:
            lookalikes.append({"domain": domain, "impersonating": target, "url": url})
    signals = find_phishing_signals(text)
    return {
        "text": text,
        "urls": urls,
        "lookalikes": lookalikes,
        "signals": signals,
    }
