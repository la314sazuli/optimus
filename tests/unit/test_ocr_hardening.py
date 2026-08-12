"""Tests for OCR hardening: preprocessing, URL repair, phishing signals."""

from optimus.hashing.ocr_extract import (
    _preprocess_variants,
    _repair_urls,
    find_phishing_signals,
)


def test_repair_urls_defanged():
    text = "Visit hxxps://openai[.]com/claim now"
    assert "https://openai.com/claim" in _repair_urls(text)


def test_repair_urls_spaces_in_domain():
    text = "Go to open ai . com / login"
    repaired = _repair_urls(text)
    assert "openai.com" in repaired


def test_repair_urls_dot_parentheses():
    text = "perplexity (.) com scam"
    assert "perplexity.com" in _repair_urls(text)


def test_repair_urls_dot_word():
    text = "free openai dot com claim"
    assert "openai.com" in _repair_urls(text)


def test_repair_urls_plain_text_unchanged():
    text = "This is normal text without urls"
    assert _repair_urls(text) == "Thisisnormaltextwithouturls"


def test_phishing_signals_free_offer():
    signals, _, _ = find_phishing_signals("Claim your FREE Pro account now!")
    assert "free_offer" in signals
    assert "claim" in signals


def test_phishing_signals_urgency():
    signals, _, _ = find_phishing_signals("Limited time! Expires in 1 hour. Act now!")
    assert "urgency" in signals


def test_phishing_signals_credentials():
    signals, _, _ = find_phishing_signals("Please verify your account and login")
    assert "credentials" in signals


def test_phishing_signals_ai_community():
    signals, _, _ = find_phishing_signals("Get free Sora access and GPT-5 beta!")
    assert "ai_community" in signals
    assert "free_offer" in signals


def test_phishing_signals_impersonation():
    signals, _, _ = find_phishing_signals("Official support team message")
    assert "impersonation" in signals


def test_phishing_signals_clean_text():
    signals, _, _ = find_phishing_signals("Hello, how are you today?")
    assert signals == []


def test_phishing_signals_multiple_categories():
    signals, _, _ = find_phishing_signals(
        "FREE airdrop! Limited time! Connect wallet to claim your reward. Official admin support."
    )
    assert "free_offer" in signals
    assert "urgency" in signals
    assert "credentials" in signals
    assert "impersonation" in signals


def test_phishing_signals_empty_text():
    signals, score, level = find_phishing_signals("")
    assert signals == []
    assert score == 0
    assert level == "none"


def test_preprocess_variants_returns_multiple():
    import numpy as np

    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    variants = _preprocess_variants(img)
    assert len(variants) >= 2
    for v in variants:
        assert v.ndim == 2


def test_preprocess_variants_upscales_small():
    import numpy as np

    img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    variants = _preprocess_variants(img)
    for v in variants:
        assert v.shape[0] >= 400  # upscaled 2x


def test_preprocess_variants_downscales_huge():
    import numpy as np

    img = np.random.randint(0, 255, (5000, 5000, 3), dtype=np.uint8)
    variants = _preprocess_variants(img)
    for v in variants:
        assert max(v.shape) <= 4000


def test_preprocess_grayscale_input():
    import numpy as np

    gray = np.random.randint(0, 255, (500, 500), dtype=np.uint8)
    variants = _preprocess_variants(gray)
    assert len(variants) >= 2
