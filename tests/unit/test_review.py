"""Unit tests for the review custom_id scheme and report content."""

from __future__ import annotations

import pytest

from optimus.services.moderation import review as review_mod
from optimus.services.moderation.review import (
    REVIEW_BUTTONS,
    ReportData,
    ReviewAction,
    build_action_rows,
    build_embed,
    decode_custom_id,
    encode_custom_id,
    report_fields,
    report_title,
)


@pytest.mark.parametrize("action", list(ReviewAction))
def test_custom_id_roundtrips(action: ReviewAction) -> None:
    cid = encode_custom_id(action, 12345)
    assert cid.startswith("om:v1:")
    parsed = decode_custom_id(cid)
    assert parsed is not None
    assert parsed.action is action
    assert parsed.detection_id == 12345


def test_decode_rejects_foreign_custom_id() -> None:
    assert decode_custom_id("other:thing:confirm_scam:1") is None
    assert decode_custom_id("om:v1:confirm_scam") is None
    assert decode_custom_id("om:v1:not_an_action:1") is None
    assert decode_custom_id("om:v1:confirm_scam:notint") is None


def test_all_buttons_are_offered() -> None:
    assert set(REVIEW_BUTTONS) == set(ReviewAction)


def test_build_action_rows_default_layout() -> None:
    rows = build_action_rows(42)
    sizes = [len(row.components) for row in rows]  # type: ignore[attr-defined]
    assert sum(sizes) == len(REVIEW_BUTTONS)
    assert all(0 < n <= 5 for n in sizes)


@pytest.mark.parametrize("count", [1, 4, 5, 6, 10, 11])
def test_build_action_rows_never_emits_empty_or_overfull_row(
    monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    # A button count that is a multiple of 5 previously produced a trailing
    # empty action row, which Discord rejects.
    buttons = tuple((list(ReviewAction) * 3)[:count])
    monkeypatch.setattr(review_mod, "REVIEW_BUTTONS", buttons)
    rows = build_action_rows(1)
    sizes = [len(row.components) for row in rows]  # type: ignore[attr-defined]
    assert sum(sizes) == count
    assert all(0 < n <= 5 for n in sizes)


def test_report_title_and_fields() -> None:
    data = ReportData(
        detection_id=9,
        guild_id=1,
        channel_id=2,
        message_id=3,
        uploader_id=42,
        verdict="scam",
        confidence=0.91,
        action_taken="delete_ban",
        matched_hash_id="camp-7",
        swarm_guilds=4,
        evidence_url="https://example/x",
    )
    assert "#9" in report_title(data)
    field_names = [name for name, _ in report_fields(data)]
    assert "Uploader" in field_names
    assert "Matched hash" in field_names
    assert "Swarm" in field_names
    assert "Evidence" in field_names


def test_report_fields_omit_optional_when_absent() -> None:
    data = ReportData(
        detection_id=1,
        guild_id=1,
        channel_id=2,
        message_id=3,
        uploader_id=4,
        verdict="ambiguous",
        confidence=0.6,
        action_taken="report_only",
    )
    names = [name for name, _ in report_fields(data)]
    assert "Matched hash" not in names
    assert "Swarm" not in names
    assert "Evidence" not in names


def test_build_embed_renders_title_and_one_field_per_report_field() -> None:
    import hikari

    data = ReportData(
        detection_id=9,
        guild_id=1,
        channel_id=2,
        message_id=3,
        uploader_id=42,
        verdict="scam",
        confidence=0.91,
        action_taken="delete_ban",
        matched_hash_id="camp-7",
    )
    embed = build_embed(data)
    assert isinstance(embed, hikari.Embed)
    assert embed.title is not None and "#9" in embed.title
    expected = report_fields(data)
    assert len(embed.fields) == len(expected)
    rendered = {(f.name, f.value) for f in embed.fields}
    assert rendered == set(expected)
    assert all(f.is_inline for f in embed.fields)
