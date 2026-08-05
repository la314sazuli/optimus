"""The live Discord edges for simple mode: one gateway bot, both listeners.

Distributed mode runs the gateway and interactions as two separate processes,
each with its own :class:`hikari.GatewayBot`. Simple mode instead runs **one**
gateway connection and hangs both listeners off it:

* message/guild-join events drive the :class:`~optimus.services.gateway.bot.GatewayService`
  (publishing ``message_image.v1`` / ``guild_joined.v1`` onto the in-process bus);
* interaction events drive the :class:`~optimus.services.interactions.service.InteractionService`
  (slash commands and review buttons, answered ephemerally).

Both services read the same SQLite engine (via the app's session scope) and the
same in-memory store, so there is no second datastore to provision. This module
is pure Discord glue — the service logic it drives is unchanged from distributed
mode.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from optimus.core.config import Settings
from optimus.core.guild_config import GuildConfigCache
from optimus.core.logging import get_logger
from optimus.core.ratelimit import InMemoryRateLimiter
from optimus.services.gateway.bot import GATEWAY_INTENTS, GatewayService, shard_start_kwargs
from optimus.services.interactions.service import InteractionService, respond_to_interaction

if TYPE_CHECKING:
    from optimus.app.simple import SimpleApp

_log = get_logger(__name__)


async def run_discord_edges(  # pragma: no cover - requires a live gateway
    app: SimpleApp, settings: Settings, *, rest: object
) -> None:
    """Connect one gateway bot wired to both the gateway and interactions edges.

    Blocks until the gateway disconnects (e.g. on interrupt). The caller owns the
    surrounding lifecycle (``app.aclose``, health/REST shutdown).
    """
    import hikari

    config_cache = GuildConfigCache(app.store, app._scope)
    gateway = GatewayService(settings, app.bus, config_cache, app.health)
    interactions = InteractionService(
        app._scope, InMemoryRateLimiter(), settings, detection=app.detection
    )

    bot = hikari.GatewayBot(token=settings.discord_token, intents=GATEWAY_INTENTS)

    @bot.listen(hikari.GuildMessageCreateEvent)
    async def _on_message(event: hikari.GuildMessageCreateEvent) -> None:
        await gateway.on_message(event)

    @bot.listen(hikari.GuildJoinEvent)
    async def _on_guild_join(event: hikari.GuildJoinEvent) -> None:
        await gateway.on_guild_join(event)

    @bot.listen(hikari.InteractionCreateEvent)
    async def _on_interaction(event: hikari.InteractionCreateEvent) -> None:
        interaction = event.interaction
        if not isinstance(interaction, hikari.CommandInteraction | hikari.ComponentInteraction):
            return
        await respond_to_interaction(interactions, interaction)

    try:
        await bot.start(**shard_start_kwargs(settings))  # type: ignore[arg-type]
        await bot.join()
    finally:
        await gateway.drain()
        with contextlib.suppress(Exception):
            await bot.close()
