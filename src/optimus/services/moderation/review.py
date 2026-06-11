"""Mod-review channel: custom_id scheme, report content, and provisioning.

The interactive button ``custom_id`` scheme is ``om:v1:<action>:<detection_id>``.
Encoding/decoding and the report's textual content are kept pure so they are
unit-testable; the hikari embed/action-row construction and the channel
provisioning REST calls live behind thin adapters at the bottom of the module.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, cast

from optimus.i18n import translate

CUSTOM_ID_PREFIX = "om:v1"


class ReviewAction(StrEnum):
    """The moderator actions offered as buttons on a report."""

    CONFIRM_SCAM = "confirm_scam"
    FALSE_POSITIVE = "false_positive"
    BAN_UPLOADER = "ban_uploader"
    UNBAN = "unban"
    WHITELIST_IMAGE = "whitelist_image"
    SUBMIT_GLOBAL = "submit_global"


def encode_custom_id(action: ReviewAction, detection_id: int) -> str:
    """Build the ``om:v1:<action>:<detection_id>`` component custom id."""
    return f"{CUSTOM_ID_PREFIX}:{action.value}:{detection_id}"


@dataclass(frozen=True, slots=True)
class ParsedCustomId:
    """A decoded review button interaction id."""

    action: ReviewAction
    detection_id: int


def decode_custom_id(custom_id: str) -> ParsedCustomId | None:
    """Parse a review ``custom_id``; return ``None`` if it is not one of ours."""
    parts = custom_id.split(":")
    if len(parts) != 4 or f"{parts[0]}:{parts[1]}" != CUSTOM_ID_PREFIX:
        return None
    try:
        action = ReviewAction(parts[2])
        detection_id = int(parts[3])
    except (ValueError, KeyError):
        return None
    return ParsedCustomId(action=action, detection_id=detection_id)


@dataclass(frozen=True, slots=True)
class ReportData:
    """The facts rendered into a moderator report embed."""

    detection_id: int
    guild_id: int
    channel_id: int
    message_id: int
    uploader_id: int
    verdict: str
    confidence: float
    action_taken: str
    matched_hash_id: str | None = None
    swarm_guilds: int | None = None
    evidence_url: str | None = None
    locale: str = "en"


def report_title(data: ReportData) -> str:
    """A short, localized title for the report."""
    return translate(
        "report.title", data.locale, detection_id=data.detection_id, verdict=data.verdict.upper()
    )


def report_fields(data: ReportData) -> list[tuple[str, str]]:
    """The ordered (localized name, value) field pairs for the report embed."""
    loc = data.locale
    fields: list[tuple[str, str]] = [
        (translate("report.field_uploader", loc), f"<@{data.uploader_id}>"),
        (translate("report.field_channel", loc), f"<#{data.channel_id}>"),
        (translate("report.field_message", loc), str(data.message_id)),
        (translate("report.field_confidence", loc), f"{data.confidence:.2f}"),
        (translate("report.field_action", loc), data.action_taken),
    ]
    if data.matched_hash_id:
        fields.append((translate("report.field_matched_hash", loc), data.matched_hash_id))
    if data.swarm_guilds:
        fields.append(
            (
                translate("report.field_swarm", loc),
                translate("report.field_swarm_value", loc, count=data.swarm_guilds),
            )
        )
    if data.evidence_url:
        fields.append((translate("report.field_evidence", loc), data.evidence_url))
    return fields


#: The buttons shown on a report, in display order.
REVIEW_BUTTONS: tuple[ReviewAction, ...] = (
    ReviewAction.CONFIRM_SCAM,
    ReviewAction.FALSE_POSITIVE,
    ReviewAction.BAN_UPLOADER,
    ReviewAction.UNBAN,
    ReviewAction.WHITELIST_IMAGE,
    ReviewAction.SUBMIT_GLOBAL,
)

BUTTON_LABELS: dict[ReviewAction, str] = {
    ReviewAction.CONFIRM_SCAM: "Confirm scam",
    ReviewAction.FALSE_POSITIVE: "False positive",
    ReviewAction.BAN_UPLOADER: "Ban uploader",
    ReviewAction.UNBAN: "Unban",
    ReviewAction.WHITELIST_IMAGE: "Whitelist image",
    ReviewAction.SUBMIT_GLOBAL: "Submit to global",
}


def build_embed(data: ReportData) -> object:
    """Build a hikari embed for ``data`` (imported lazily to keep this testable)."""
    import hikari

    embed = hikari.Embed(title=report_title(data))
    for name, value in report_fields(data):
        embed.add_field(name=name, value=value, inline=True)
    return embed


def build_action_rows(detection_id: int) -> list[object]:
    """Build hikari message action rows with the review buttons."""
    import hikari

    rows: list[object] = []
    row = hikari.impl.MessageActionRowBuilder()
    buttons_in_row = 0
    for action in REVIEW_BUTTONS:
        style = (
            hikari.ButtonStyle.SUCCESS
            if action is ReviewAction.CONFIRM_SCAM
            else hikari.ButtonStyle.DANGER
            if action in (ReviewAction.BAN_UPLOADER, ReviewAction.FALSE_POSITIVE)
            else hikari.ButtonStyle.SECONDARY
        )
        # Discord allows up to 5 buttons per row; start a fresh row when full.
        if buttons_in_row == 5:
            rows.append(row)
            row = hikari.impl.MessageActionRowBuilder()
            buttons_in_row = 0
        row.add_interactive_button(
            cast("Any", style),
            encode_custom_id(action, detection_id),
            label=BUTTON_LABELS[action],
        )
        buttons_in_row += 1
    if buttons_in_row:
        rows.append(row)
    return rows


class ChannelProvisioner(Protocol):
    """REST surface needed to auto-provision a private mod-review channel."""

    async def create_review_channel(
        self, guild_id: int, *, name: str, mod_role_ids: list[int]
    ) -> int: ...
