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
from optimus.core.readiness import shards_check
from optimus.services.gateway.bot import GATEWAY_INTENTS, GatewayService, shard_start_kwargs
from optimus.services.gateway.watchdog import GatewayWatchdog
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

    # Readiness should track the gateway, not just the DB: a wedged gateway
    # session (the "commands all time out" incident) previously left /readyz
    # green because only the database check was registered in simple mode.
    app.health.add_readiness_check(shards_check(bot), name="shards")
    watchdog = GatewayWatchdog(
        bot,
        app.health,
        interval_seconds=settings.gateway_watchdog_interval_seconds,
        stale_exit_seconds=settings.gateway_stale_exit_seconds,
    )

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
        watchdog.start()
        await bot.join()
    finally:
        await watchdog.stop()
        await gateway.drain()
        with contextlib.suppress(Exception):
            await bot.close()
