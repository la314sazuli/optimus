"""hikari gateway wiring: least-privilege intents, publish-only, stateless.

The gateway subscribes to guild message creation, applies per-guild scan
filters (ignored channels/roles, trusted users, bot/webhook opt-in) using a
Redis-cached guild config, and publishes one ``message_image.v1`` event per
inspectable image. It holds no state beyond in-flight publishes, which are
drained on shutdown.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

import hikari

from optimus.bus.nats import EventBus
from optimus.contracts.events import SUBJECT_MESSAGE_IMAGE
from optimus.core.config import Settings, get_settings
from optimus.core.guild_config import GuildConfig, GuildConfigCache
from optimus.core.health import HealthServer
from optimus.core.logging import configure_logging, correlation_context, get_logger
from optimus.services.gateway.extract import (
    Attachment,
    IncomingMessage,
    build_events,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

_log = get_logger(__name__)

# Least-privilege: read guild structure, message events, and message content.
GATEWAY_INTENTS = (
    hikari.Intents.GUILDS | hikari.Intents.GUILD_MESSAGES | hikari.Intents.MESSAGE_CONTENT
)


def _embed_image_urls(embeds: Iterable[hikari.Embed]) -> tuple[str, ...]:
    urls: list[str] = []
    for embed in embeds:
        if embed.image is not None and embed.image.url:
            urls.append(embed.image.url)
        if embed.thumbnail is not None and embed.thumbnail.url:
            urls.append(embed.thumbnail.url)
    return tuple(urls)


def to_incoming(event: hikari.GuildMessageCreateEvent) -> IncomingMessage:
    """Adapt a hikari message-create event into a plain :class:`IncomingMessage`."""
    msg = event.message
    author = event.author
    attachments = tuple(
        Attachment(
            id=int(a.id),
            url=str(a.url),
            filename=a.filename,
            content_type=a.media_type,
        )
        for a in msg.attachments
    )
    member = msg.member
    role_ids = frozenset(int(r) for r in member.role_ids) if member is not None else frozenset()
    return IncomingMessage(
        guild_id=int(event.guild_id),
        channel_id=int(event.channel_id),
        message_id=int(msg.id),
        author_id=int(author.id),
        content=msg.content or "",
        attachments=attachments,
        embed_image_urls=_embed_image_urls(msg.embeds),
        is_bot=bool(author.is_bot),
        is_webhook=msg.webhook_id is not None,
        author_role_ids=role_ids,
    )


class GatewayService:
    """Owns the hikari bot, the event bus, and the health server."""

    def __init__(
        self,
        settings: Settings,
        bus: EventBus,
        config_cache: GuildConfigCache,
        health: HealthServer,
    ) -> None:
        self._settings = settings
        self._bus = bus
        self._config = config_cache
        self._health = health
        self._inflight: set[asyncio.Task[None]] = set()

    async def on_message(self, event: hikari.GuildMessageCreateEvent) -> None:
        """Filter and publish image events for one message."""
        msg = to_incoming(event)
        config = await self._config.get(msg.guild_id)
        if not self._should_scan(config, msg):
            return
        with correlation_context() as cid:
            events = build_events(msg, correlation_id=cid)
            for image_event in events:
                await self._bus.publish(SUBJECT_MESSAGE_IMAGE, image_event)
            if events:
                _log.info(
                    "gateway_published",
                    guild_id=msg.guild_id,
                    message_id=msg.message_id,
                    images=len(events),
                )

    @staticmethod
    def _should_scan(config: GuildConfig, msg: IncomingMessage) -> bool:
        return config.should_scan(
            channel_id=msg.channel_id,
            uploader_id=msg.author_id,
            author_role_ids=msg.author_role_ids,
            is_bot=msg.is_bot,
            is_webhook=msg.is_webhook,
        )

    def track(self, task: asyncio.Task[None]) -> None:
        """Track an in-flight publish task so shutdown can drain it."""
        self._inflight.add(task)
        task.add_done_callback(self._inflight.discard)

    async def drain(self) -> None:
        """Await all in-flight publish tasks during graceful shutdown."""
        if self._inflight:
            await asyncio.gather(*self._inflight, return_exceptions=True)


async def _amain() -> None:
    settings = get_settings()
    configure_logging(level=settings.log_level, service_name="optimus-gateway")

    bus, nc = await EventBus.connect(settings.nats_url)
    await bus.ensure_stream()

    redis = _open_redis(settings)
    from optimus.db.engine import create_engine, create_session_factory, session_scope

    engine = create_engine()
    factory = create_session_factory(engine)

    def loader() -> object:
        return session_scope(factory)

    config_cache = GuildConfigCache(redis, loader)

    health = HealthServer(host=settings.health_host, port=settings.health_port)
    await health.start()

    bot = hikari.GatewayBot(token=settings.discord_token, intents=GATEWAY_INTENTS)
    service = GatewayService(settings, bus, config_cache, health)

    @bot.listen(hikari.GuildMessageCreateEvent)
    async def _on_message(event: hikari.GuildMessageCreateEvent) -> None:
        await service.on_message(event)

    try:
        await bot.start()
        await bot.join()
    finally:
        health.set_live(False)
        await service.drain()
        with contextlib.suppress(Exception):
            await bot.close()
        with contextlib.suppress(Exception):
            await nc.drain()
        await health.stop()
        await engine.dispose()


def _open_redis(settings: Settings) -> object | None:
    try:
        import redis.asyncio as aioredis

        return aioredis.from_url(settings.redis_url, decode_responses=True)
    except Exception:  # pragma: no cover - redis optional at boot
        _log.warning("redis_unavailable_gateway")
        return None


def main() -> None:
    """Console entrypoint: ``python -m optimus.services.gateway``."""
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
