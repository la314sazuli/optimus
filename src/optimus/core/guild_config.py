"""Hot-path guild configuration: Redis cache with a Postgres fallback.

The gateway and detection workers need a guild's scan policy on every message
without hammering Postgres. This module exposes an immutable snapshot loaded
from the database and cached in Redis as JSON with a short TTL. A cache miss
falls back to the database and repopulates the cache.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from optimus.core.config import Sensitivity
from optimus.db.models import (
    Guild,
    GuildChannelIgnored,
    GuildRoleIgnored,
    GuildTrustedUser,
)

_CACHE_PREFIX = "optimus:guildcfg"
_DEFAULT_TTL = 300


@dataclass(frozen=True, slots=True)
class GuildConfig:
    """An immutable snapshot of a guild's scanning policy."""

    guild_id: int
    sensitivity: Sensitivity = Sensitivity.BALANCED
    scan_bots: bool = False
    safe_mode: bool = False
    ignored_channels: frozenset[int] = field(default_factory=frozenset)
    ignored_roles: frozenset[int] = field(default_factory=frozenset)
    trusted_users: frozenset[int] = field(default_factory=frozenset)

    def should_scan(
        self,
        *,
        channel_id: int,
        uploader_id: int,
        author_role_ids: frozenset[int],
        is_bot: bool,
        is_webhook: bool,
    ) -> bool:
        """Whether a message from this author/channel should be scanned."""
        if channel_id in self.ignored_channels:
            return False
        if uploader_id in self.trusted_users:
            return False
        if author_role_ids & self.ignored_roles:
            return False
        return not ((is_bot or is_webhook) and not self.scan_bots)

    def to_json(self) -> str:
        """Serialize to a compact JSON string for caching."""
        return json.dumps(
            {
                "guild_id": self.guild_id,
                "sensitivity": self.sensitivity.value,
                "scan_bots": self.scan_bots,
                "safe_mode": self.safe_mode,
                "ignored_channels": sorted(self.ignored_channels),
                "ignored_roles": sorted(self.ignored_roles),
                "trusted_users": sorted(self.trusted_users),
            },
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, raw: str | bytes) -> GuildConfig:
        """Deserialize a cached JSON snapshot."""
        data = json.loads(raw)
        return cls(
            guild_id=int(data["guild_id"]),
            sensitivity=Sensitivity(data["sensitivity"]),
            scan_bots=bool(data["scan_bots"]),
            safe_mode=bool(data["safe_mode"]),
            ignored_channels=frozenset(int(x) for x in data["ignored_channels"]),
            ignored_roles=frozenset(int(x) for x in data["ignored_roles"]),
            trusted_users=frozenset(int(x) for x in data["trusted_users"]),
        )

    @classmethod
    def default(cls, guild_id: int) -> GuildConfig:
        """A safe default snapshot for an unconfigured guild (scan everything human)."""
        return cls(guild_id=guild_id)


async def load_from_db(session: AsyncSession, guild_id: int) -> GuildConfig:
    """Load a guild's scan policy directly from Postgres."""
    guild = await session.get(Guild, guild_id)
    if guild is None:
        return GuildConfig.default(guild_id)

    channels = (
        (
            await session.execute(
                select(GuildChannelIgnored.channel_id).where(
                    GuildChannelIgnored.guild_id == guild_id
                )
            )
        )
        .scalars()
        .all()
    )
    roles = (
        (
            await session.execute(
                select(GuildRoleIgnored.role_id).where(GuildRoleIgnored.guild_id == guild_id)
            )
        )
        .scalars()
        .all()
    )
    users = (
        (
            await session.execute(
                select(GuildTrustedUser.user_id).where(GuildTrustedUser.guild_id == guild_id)
            )
        )
        .scalars()
        .all()
    )

    return GuildConfig(
        guild_id=guild_id,
        sensitivity=Sensitivity(guild.sensitivity),
        scan_bots=guild.optin_scan_bots,
        safe_mode=guild.safe_mode,
        ignored_channels=frozenset(channels),
        ignored_roles=frozenset(roles),
        trusted_users=frozenset(users),
    )


class GuildConfigCache:
    """Redis-cached guild config with a Postgres fallback.

    ``redis`` may be ``None`` (no cache available), in which case every lookup
    hits the database. ``session_provider`` yields an :class:`AsyncSession` as an
    async context manager.
    """

    def __init__(
        self,
        redis: object | None,
        loader: object,
        *,
        prefix: str = _CACHE_PREFIX,
        ttl_seconds: int = _DEFAULT_TTL,
    ) -> None:
        self._redis = redis
        self._loader = loader
        self._prefix = prefix
        self._ttl = ttl_seconds

    def _key(self, guild_id: int) -> str:
        return f"{self._prefix}:{guild_id}"

    async def get(self, guild_id: int) -> GuildConfig:
        """Return the guild config, preferring the Redis cache."""
        if self._redis is not None:
            cached = await self._redis.get(self._key(guild_id))  # type: ignore[attr-defined]
            if cached is not None:
                return GuildConfig.from_json(cached)
        config = await self._load(guild_id)
        if self._redis is not None:
            await self._redis.set(  # type: ignore[attr-defined]
                self._key(guild_id), config.to_json(), ex=self._ttl
            )
        return config

    async def invalidate(self, guild_id: int) -> None:
        """Drop a guild's cached config (call after a config change)."""
        if self._redis is not None:
            await self._redis.delete(self._key(guild_id))  # type: ignore[attr-defined]

    async def _load(self, guild_id: int) -> GuildConfig:
        async with self._loader() as session:  # type: ignore[operator]
            return await load_from_db(session, guild_id)
