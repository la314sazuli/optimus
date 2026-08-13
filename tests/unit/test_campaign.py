"""Tests for the scam campaign detection module."""

from __future__ import annotations

from optimus.hashing.campaign import (
    CAMPAIGN_THRESHOLD,
    campaign_color,
    find_campaign,
    new_campaign_id,
    summarize_campaigns,
)


def test_find_campaign_matches_similar_phash():
    # phash values within threshold are the same campaign
    existing = [("abc123", 0x0000000000000000)]
    assert find_campaign(0x0000000000000001, existing) == "abc123"


def test_find_campaign_returns_none_for_different_phash():
    # Completely different hashes don't match
    existing = [("abc123", 0xFFFFFFFFFFFFFFFF)]
    assert find_campaign(0x0000000000000000, existing) is None


def test_find_campaign_at_exact_threshold():
    # At exactly the threshold distance, it still matches
    base = 0x0000000000000000
    # Set exactly CAMPAIGN_THRESHOLD bits
    target = base | ((1 << CAMPAIGN_THRESHOLD) - 1)
    existing = [("camp1", base)]
    assert find_campaign(target, existing) == "camp1"


def test_find_campaign_just_above_threshold():
    # One bit above threshold does not match
    base = 0x0000000000000000
    target = base | ((1 << (CAMPAIGN_THRESHOLD + 1)) - 1)
    existing = [("camp1", base)]
    assert find_campaign(target, existing) is None


def test_find_campaign_empty_existing():
    assert find_campaign(0x12345678, []) is None


def test_new_campaign_id_is_unique():
    ids = {new_campaign_id() for _ in range(100)}
    assert len(ids) == 100


def test_campaign_color_small():
    assert campaign_color(2) == 0x95A5A6  # gray
    assert campaign_color(3) == 0x95A5A6


def test_campaign_color_medium():
    assert campaign_color(4) == 0xF1C40F  # yellow
    assert campaign_color(9) == 0xF1C40F


def test_campaign_color_large():
    assert campaign_color(10) == 0xE74C3C  # red
    assert campaign_color(50) == 0xE74C3C


def test_summarize_campaigns():
    campaigns = [("abc12345", 3), ("def67890", 7)]
    lines = summarize_campaigns(campaigns)
    assert len(lines) == 2
    assert "abc12345" in lines[0]
    assert "3 variant(s)" in lines[0]
    assert "def67890" in lines[1]
    assert "7 variant(s)" in lines[1]


def test_summarize_campaigns_empty():
    assert summarize_campaigns([]) == []
