"""Adapter mapping the executor's ``RestActions`` protocol onto hikari's client.

The moderation executor speaks the small :class:`~optimus.services.moderation.actions.RestActions`
protocol (``ban_member(guild_id, user_id, reason)`` and friends). hikari's real
REST client has different names and keyword-only parameters — ``ban_user(guild,
user, *, reason=...)``, no ``timeout_member``, no ``send_dm`` — so handing the
raw client to the executor made every punitive action raise ``TypeError`` or
``AttributeError`` at runtime, while protocol-shaped test doubles kept the suite
green. This adapter is the one place where the protocol meets hikari's true
signatures, and its tests bind against ``create_autospec(RESTClientImpl)`` so a
signature drift fails the suite instead of production.

Attributes outside the protocol (``fetch_member``, ``fetch_guild``,
``fetch_roles``, ``create_message``, ...) delegate to the wrapped client, so the
coordinator's target resolver and report poster keep working against the same
object.
"""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime, timedelta
from typing import Any

import hikari


class HikariRestActions:
    """:class:`RestActions` implemented against a real hikari REST client."""

    def __init__(self, rest: Any) -> None:
        self._rest = rest

    def __getattr__(self, name: str) -> Any:
        # Non-protocol surface passes straight through to hikari.
        return getattr(self._rest, name)

    async def delete_message(self, channel_id: int, message_id: int) -> None:
        """Delete a message; an already-deleted message counts as success.

        The goal is "message gone", so a 404 is not a failure — this also makes
        a manual re-run of ``reviewmsg`` on an already-removed message idempotent
        instead of erroring out before the punitive step.
        """
        with contextlib.suppress(hikari.NotFoundError):
            await self._rest.delete_message(channel_id, message_id)

    async def timeout_member(self, guild_id: int, user_id: int, seconds: int) -> None:
        until = datetime.now(UTC) + timedelta(seconds=seconds)
        await self._rest.edit_member(guild_id, user_id, communication_disabled_until=until)

    async def kick_member(self, guild_id: int, user_id: int, reason: str) -> None:
        await self._rest.kick_user(guild_id, user_id, reason=reason)

    async def ban_member(self, guild_id: int, user_id: int, reason: str) -> None:
        await self._rest.ban_user(guild_id, user_id, reason=reason)

    async def unban_member(self, guild_id: int, user_id: int, reason: str) -> None:
        await self._rest.unban_user(guild_id, user_id, reason=reason)

    async def send_dm(self, user_id: int, content: str) -> None:
        channel = await self._rest.create_dm_channel(user_id)
        await self._rest.create_message(channel.id, content)
