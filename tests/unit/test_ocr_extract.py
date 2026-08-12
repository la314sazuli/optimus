"""Tests for OCR text extraction, URL extraction, and domain lookalike detection."""

from optimus.hashing.ocr_extract import (
    extract_urls,
    is_lookalike,
    normalize_domain,
)


def test_extract_urls_from_text():
    text = "Claim free Pro at https://perplexity-claim.com/offer now!"
    urls = extract_urls(text)
    assert len(urls) == 1
    assert (
        "perplexity-claim.com/offer" in urls[0] or "https://perplexity-claim.com/offer" in urls[0]
    )


def test_extract_urls_dedupes():
    text = "Visit openai.com and openai.com today"
    urls = extract_urls(text)
    assert len(urls) == 1


def test_extract_urls_handles_empty():
    assert extract_urls("") == []
    assert extract_urls("no urls here") == []


def test_normalize_domain_strips_www():
    assert normalize_domain("www.openai.com") == "openai.com"


def test_normalize_domain_lowercases():
    assert normalize_domain("OpenAI.COM/path") == "openai.com"


def test_normalize_domain_handles_bare_domain():
    assert normalize_domain("perplexity.ai") == "perplexity.ai"


def test_lookalike_exact_match_is_safe():
    assert is_lookalike("openai.com") is None
    assert is_lookalike("perplexity.ai") is None


def test_lookalike_subdomain_of_official_is_safe():
    assert is_lookalike("api.openai.com") is None


def test_lookalike_detects_prefix():
    assert is_lookalike("openai-claim.com") == "openai.com"


def test_lookalike_detects_tld_swap():
    result = is_lookalike("openai.ru")
    assert result is not None
    assert "openai" in result


def test_lookalike_detects_close_typos():
    assert is_lookalike("openai.con") == "openai.com"


def test_lookalike_returns_none_for_unrelated():
    assert is_lookalike("example.com") is None
    assert is_lookalike("github.com") is None


def test_lookalike_empty_domain():
    assert is_lookalike("") is None
