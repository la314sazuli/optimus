"""Scam campaign detection: cluster visually similar hashes.

A campaign is a group of scam images that are visually similar (small Hamming
distance between perceptual hashes) — typically recompressed, resized, or
watermarked variants of the same source image. Detecting campaigns lets
modators see coordinated attacks rather than isolated incidents.
"""

from __future__ import annotations

import secrets

from optimus.hashing.perceptual import hamming

#: Maximum Hamming distance for two phash values to be considered the same campaign.
CAMPAIGN_THRESHOLD = 8


def new_campaign_id() -> str:
    """Generate a short, URL-safe campaign identifier."""
    return secrets.token_hex(8)


def find_campaign(
    new_phash: int,
    existing: list[tuple[str, int]],
    *,
    threshold: int = CAMPAIGN_THRESHOLD,
) -> str | None:
    """Return the campaign_id of the first matching hash, if any.

    ``existing`` is a list of ``(campaign_id, phash)`` pairs for active hashes
    already in the guild. Returns ``None`` when no existing hash is within
    ``threshold`` bits of ``new_phash``, meaning this image starts a new campaign.
    """
    for campaign_id, existing_phash in existing:
        if hamming(new_phash, existing_phash) <= threshold:
            return campaign_id
    return None


def campaign_color(member_count: int) -> int:
    """Embed color for a campaign based on its member count."""
    if member_count >= 10:
        return 0xE74C3C  # red — large campaign
    if member_count >= 4:
        return 0xF1C40F  # yellow — medium campaign
    return 0x95A5A6  # gray — small campaign


def summarize_campaigns(campaigns: list[tuple[str, int]]) -> list[str]:
    """Format campaigns as display lines."""
    lines = []
    for campaign_id, count in campaigns:
        lines.append(f"**Campaign `{campaign_id}`** — {count} variant(s)")
    return lines
