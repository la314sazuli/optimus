"""hikari component (button) row builders for the appeal and safe-mode flows.

These complement the report buttons built in
:mod:`optimus.services.moderation.review`. Every button's ``custom_id`` is
encoded with :func:`~optimus.services.interactions.logic.encode_component_id`
so it round-trips back to a :class:`ComponentAction` in the service layer.
"""

from __future__ import annotations

from optimus.i18n import translate
from optimus.services.interactions.logic import ComponentAction, encode_component_id


def build_appeal_dm_row(detection_id: int, locale: str = "en") -> object:
    """Build the single-button action row attached to a DM warning."""
    import hikari

    row = hikari.impl.MessageActionRowBuilder()
    row.add_interactive_button(
        hikari.ButtonStyle.SECONDARY,
        encode_component_id(ComponentAction.APPEAL_OPEN, detection_id),
        label=translate("dm.appeal_button", locale),
    )
    return row


def build_appeal_review_row(appeal_id: int, locale: str = "en") -> object:
    """Build the approve/deny action row posted to the mod-review channel."""
    import hikari

    row = hikari.impl.MessageActionRowBuilder()
    row.add_interactive_button(
        hikari.ButtonStyle.SUCCESS,
        encode_component_id(ComponentAction.APPEAL_APPROVE, appeal_id),
        label=translate("report.appeal_status", locale),
    )
    row.add_interactive_button(
        hikari.ButtonStyle.DANGER,
        encode_component_id(ComponentAction.APPEAL_DENY, appeal_id),
        label=translate("dm.appeal_denied", locale),
    )
    return row


def build_safe_mode_row(guild_id: int, locale: str = "en") -> object:
    """Build the safe-mode resume action row."""
    import hikari

    row = hikari.impl.MessageActionRowBuilder()
    row.add_interactive_button(
        hikari.ButtonStyle.PRIMARY,
        encode_component_id(ComponentAction.SAFE_MODE_RESUME, guild_id),
        label=translate("button.safe_mode_resumed", locale),
    )
    return row


def build_delete_confirm_row(guild_id: int, locale: str = "en") -> object:
    """Build the destructive ``/delete_server_data`` confirmation row."""
    import hikari

    row = hikari.impl.MessageActionRowBuilder()
    row.add_interactive_button(
        hikari.ButtonStyle.DANGER,
        encode_component_id(ComponentAction.DELETE_SERVER_CONFIRM, guild_id),
        label=translate("command.delete_server_button", locale),
    )
    return row
