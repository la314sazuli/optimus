"""Tests for the Discord interaction response lifecycle."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import hikari
import pytest

from optimus.services.interactions import service as interaction_service
from optimus.services.interactions.handlers import InteractionContext
from optimus.services.interactions.logic import (
    CommandError,
    InteractionRejected,
    ModAction,
    ParsedModId,
)

RESPONSE_MESSAGE = "Configuration updated."


@pytest.mark.parametrize(
    "interaction_type", [hikari.CommandInteraction, hikari.ComponentInteraction]
)
async def test_interaction_is_deferred_before_dispatch_then_edited(
    monkeypatch: pytest.MonkeyPatch, interaction_type: type[object]
) -> None:
    events: list[str] = []
    interaction = MagicMock(spec=interaction_type)
    interaction.create_initial_response = AsyncMock(
        side_effect=lambda *args, **kwargs: events.append("defer")
    )
    interaction.edit_initial_response = AsyncMock(
        side_effect=lambda *args, **kwargs: events.append("edit")
    )

    async def run_interaction(
        service: object, received_interaction: object
    ) -> tuple[str, int | None, list[object]]:
        assert events == ["defer"]
        assert received_interaction is interaction
        events.append("dispatch")
        return RESPONSE_MESSAGE, None, []

    monkeypatch.setattr(interaction_service, "run_interaction", run_interaction)

    await interaction_service.respond_to_interaction(MagicMock(), interaction)

    assert events == ["defer", "dispatch", "edit"]
    interaction.create_initial_response.assert_awaited_once_with(
        hikari.ResponseType.DEFERRED_MESSAGE_CREATE,
        flags=hikari.MessageFlag.EPHEMERAL,
    )
    interaction.edit_initial_response.assert_awaited_once_with(RESPONSE_MESSAGE, components=None)


@pytest.mark.asyncio
async def test_add_scam_button_rejects_a_message_from_another_guild() -> None:
    message = MagicMock()
    message.guild_id = 999
    rest = MagicMock()
    rest.fetch_message = AsyncMock(return_value=message)
    ctx = InteractionContext(
        guild_id=100,
        user_id=200,
        member_permissions=0x20,
        command="",
    )

    with pytest.raises(InteractionRejected) as exc:
        await interaction_service._resolve_mod_button_message(
            ctx, ParsedModId(ModAction.ADD_SCAM, 123, 456), rest
        )

    assert exc.value.reason is CommandError.MESSAGE_NOT_FOUND
