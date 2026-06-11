"""Repository classes enforcing ``guild_id`` scoping on every query.

These repositories are the only sanctioned path to per-guild data. Every read
and write is filtered by ``guild_id`` as application-level defense in depth on
top of Postgres RLS (multi-tenant mode).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from sqlalchemy import CursorResult, delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from optimus.db.models import (
    Detection,
    GlobalHash,
    Guild,
    GuildHash,
    GuildWhitelist,
)


class GuildRepository:
    """CRUD for guild configuration rows."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, guild_id: int) -> Guild | None:
        """Return the guild config, or ``None``."""
        return await self._session.get(Guild, guild_id)

    async def upsert(self, guild: Guild) -> Guild:
        """Insert or update a guild config row."""
        merged = await self._session.merge(guild)
        await self._session.flush()
        return merged

    async def set_safe_mode(self, guild_id: int, enabled: bool) -> None:
        """Toggle a guild's safe mode."""
        guild = await self.get(guild_id)
        if guild is None:
            raise KeyError(guild_id)
        guild.safe_mode = enabled
        await self._session.flush()


class GuildHashRepository:
    """Guild-scoped access to known scam hashes."""

    def __init__(self, session: AsyncSession, guild_id: int) -> None:
        self._session = session
        self._guild_id = guild_id

    async def add(self, gh: GuildHash) -> GuildHash:
        """Add a hash, forcing it into this repository's guild scope."""
        gh.guild_id = self._guild_id
        self._session.add(gh)
        await self._session.flush()
        return gh

    async def get(self, hash_id: str) -> GuildHash | None:
        """Return a hash by id, scoped to this guild."""
        stmt = select(GuildHash).where(
            GuildHash.guild_id == self._guild_id, GuildHash.hash_id == hash_id
        )
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def list_active(self) -> Sequence[GuildHash]:
        """Return all active hashes for this guild."""
        stmt = select(GuildHash).where(
            GuildHash.guild_id == self._guild_id, GuildHash.status == "active"
        )
        return (await self._session.execute(stmt)).scalars().all()

    async def remove(self, hash_id: str) -> int:
        """Delete a hash by id within this guild; return rows affected."""
        stmt = delete(GuildHash).where(
            GuildHash.guild_id == self._guild_id, GuildHash.hash_id == hash_id
        )
        result = await self._session.execute(stmt)
        await self._session.flush()
        return cast("CursorResult[Any]", result).rowcount or 0


class WhitelistRepository:
    """Guild-scoped whitelist access (always overrides global matches)."""

    def __init__(self, session: AsyncSession, guild_id: int) -> None:
        self._session = session
        self._guild_id = guild_id

    async def add(self, entry: GuildWhitelist) -> GuildWhitelist:
        """Add a whitelist entry within this guild scope."""
        entry.guild_id = self._guild_id
        self._session.add(entry)
        await self._session.flush()
        return entry

    async def list(self) -> Sequence[GuildWhitelist]:
        """Return all whitelist entries for this guild."""
        stmt = select(GuildWhitelist).where(GuildWhitelist.guild_id == self._guild_id)
        return (await self._session.execute(stmt)).scalars().all()


class GlobalHashRepository:
    """Read access to globally-shared promoted scam hashes (not guild-scoped)."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list_promoted(self) -> Sequence[GlobalHash]:
        """Return all promoted global hashes (the set every worker indexes)."""
        stmt = select(GlobalHash).where(GlobalHash.status == "promoted")
        return (await self._session.execute(stmt)).scalars().all()


class DetectionRepository:
    """Guild-scoped detection records with idempotent insertion."""

    def __init__(self, session: AsyncSession, guild_id: int) -> None:
        self._session = session
        self._guild_id = guild_id

    async def record(self, detection: Detection) -> Detection:
        """Persist a detection, forcing this repository's guild scope."""
        detection.guild_id = self._guild_id
        self._session.add(detection)
        await self._session.flush()
        return detection

    async def get_by_idempotency_key(self, key: str) -> Detection | None:
        """Return a detection by idempotency key, scoped to this guild."""
        stmt = select(Detection).where(
            Detection.guild_id == self._guild_id, Detection.idempotency_key == key
        )
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def list_recent(self, limit: int = 100) -> Sequence[Detection]:
        """Return the most recent detections for this guild."""
        stmt = (
            select(Detection)
            .where(Detection.guild_id == self._guild_id)
            .order_by(Detection.created_at.desc())
            .limit(limit)
        )
        return (await self._session.execute(stmt)).scalars().all()
