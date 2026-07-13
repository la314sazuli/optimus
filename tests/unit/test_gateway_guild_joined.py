"""Unit test: the gateway publishes guild_joined.v1 on a guild-join event."""

from __future__ import annotations

from dataclasses import dataclass

from optimus.contracts.events import SUBJECT_GUILD_JOINED, GuildJoinedEvent
from optimus.core.guild_config import GuildConfigCache
from optimus.services.gateway.bot import GatewayService


class _FakeBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, object]] = []

    async def publish(self, subject: str, event: object) -> None:
        self.published.append((subject, event))


@dataclass
class _Guild:
    name: str
    owner_id: int


@dataclass
class _JoinEvent:
    guild_id: int
    guild: _Guild | None


async def test_on_guild_join_publishes_event() -> None:
    bus = _FakeBus()
    service = GatewayService(
        settings=object(),  # type: ignore[arg-type]
        bus=bus,  # type: ignore[arg-type]
        config_cache=GuildConfigCache(None, lambda guild_id=None: None),
        health=object(),  # type: ignore[arg-type]
    )
    event = _JoinEvent(guild_id=123, guild=_Guild(name="My Server", owner_id=7))
    await service.on_guild_join(event)  # type: ignore[arg-type]

    assert len(bus.published) == 1
    subject, published = bus.published[0]
    assert subject == SUBJECT_GUILD_JOINED
    assert isinstance(published, GuildJoinedEvent)
    assert published.guild_id == 123
    assert published.guild_name == "My Server"
    assert published.owner_id == 7


async def test_on_guild_join_handles_missing_guild() -> None:
    bus = _FakeBus()
    service = GatewayService(
        settings=object(),  # type: ignore[arg-type]
        bus=bus,  # type: ignore[arg-type]
        config_cache=GuildConfigCache(None, lambda guild_id=None: None),
        health=object(),  # type: ignore[arg-type]
    )
    await service.on_guild_join(_JoinEvent(guild_id=99, guild=None))  # type: ignore[arg-type]
    _, published = bus.published[0]
    assert isinstance(published, GuildJoinedEvent)
    assert published.guild_name is None
    assert published.owner_id is None
