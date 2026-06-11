"""Repository classes enforcing ``guild_id`` scoping on every query.

These repositories are the only sanctioned path to per-guild data. Every read
and write is filtered by ``guild_id`` as application-level defense in depth on
top of Postgres RLS (multi-tenant mode).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any, cast

from sqlalchemy import CursorResult, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from optimus.db.models import (
    Appeal,
    Detection,
    Evidence,
    GlobalHash,
    GlobalHashApproval,
    GlobalSubmitter,
    Guild,
    GuildChannelIgnored,
    GuildHash,
    GuildRoleIgnored,
    GuildTrustedUser,
    GuildWhitelist,
    ModAction,
    StatsRollup,
    UserOptout,
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
    """Access to globally-shared scam hashes (candidate/promoted/revoked)."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list_promoted(self) -> Sequence[GlobalHash]:
        """Return all promoted global hashes (the set every worker indexes)."""
        stmt = select(GlobalHash).where(GlobalHash.status == "promoted")
        return (await self._session.execute(stmt)).scalars().all()

    async def get(self, hash_id: str) -> GlobalHash | None:
        """Return a global hash by id, regardless of status."""
        return await self._session.get(GlobalHash, hash_id)

    async def submit_candidate(
        self,
        *,
        hash_id: str,
        phash: int,
        dhash: int,
        whash: int,
        submitter_user_id: int,
        submitter_guild_id: int,
    ) -> GlobalHash:
        """Insert a new candidate, or return the existing row for ``hash_id``."""
        existing = await self.get(hash_id)
        if existing is not None:
            return existing
        row = GlobalHash(
            hash_id=hash_id,
            phash=phash,
            dhash=dhash,
            whash=whash,
            status="candidate",
            submitter_user_id=submitter_user_id,
            submitter_guild_id=submitter_guild_id,
        )
        self._session.add(row)
        await self._session.flush()
        return row

    async def add_approval(
        self, *, hash_id: str, approver_user_id: int, approver_guild_id: int
    ) -> Sequence[GlobalHashApproval]:
        """Record an approval (idempotent per user) and return all approvals."""
        stmt = select(GlobalHashApproval).where(
            GlobalHashApproval.hash_id == hash_id,
            GlobalHashApproval.approver_user_id == approver_user_id,
        )
        if (await self._session.execute(stmt)).scalar_one_or_none() is None:
            self._session.add(
                GlobalHashApproval(
                    hash_id=hash_id,
                    approver_user_id=approver_user_id,
                    approver_guild_id=approver_guild_id,
                )
            )
            await self._session.flush()
        return await self.list_approvals(hash_id)

    async def list_approvals(self, hash_id: str) -> Sequence[GlobalHashApproval]:
        """Return all approvals recorded for ``hash_id``."""
        stmt = select(GlobalHashApproval).where(GlobalHashApproval.hash_id == hash_id)
        return (await self._session.execute(stmt)).scalars().all()

    async def promote(self, hash_id: str, *, signature: str) -> None:
        """Mark a candidate promoted and attach its Ed25519 signature."""
        row = await self.get(hash_id)
        if row is None:
            raise KeyError(hash_id)
        row.status = "promoted"
        row.signature = signature
        await self._session.flush()

    async def revoke(self, hash_id: str) -> None:
        """Mark a hash revoked so consumers stop trusting it."""
        row = await self.get(hash_id)
        if row is None:
            raise KeyError(hash_id)
        row.status = "revoked"
        await self._session.flush()


class GlobalSubmitterRepository:
    """Submitter reputation for the global hash database (not guild-scoped)."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_or_create(self, user_id: int) -> GlobalSubmitter:
        """Return the submitter row for ``user_id``, creating it at zero."""
        row = await self._session.get(GlobalSubmitter, user_id)
        if row is None:
            row = GlobalSubmitter(user_id=user_id)
            self._session.add(row)
            await self._session.flush()
        return row

    async def record_submission(self, user_id: int) -> GlobalSubmitter:
        """Increment a submitter's submission counter."""
        row = await self.get_or_create(user_id)
        row.submitted += 1
        await self._session.flush()
        return row

    async def adjust_reputation(
        self, user_id: int, *, confirmed: int = 0, rejected: int = 0
    ) -> GlobalSubmitter:
        """Apply confirm/reject reputation deltas to a submitter."""
        from optimus.globaldb.promotion import reputation_after

        row = await self.get_or_create(user_id)
        row.confirmed += confirmed
        row.rejected += rejected
        row.reputation = reputation_after(
            row.reputation, confirmed=confirmed, rejected=rejected
        )
        await self._session.flush()
        return row


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

    async def set_action_taken(self, detection_id: int, action: str) -> int:
        """Record the action applied to a detection; return rows affected."""
        from sqlalchemy import update

        stmt = (
            update(Detection)
            .where(Detection.guild_id == self._guild_id, Detection.id == detection_id)
            .values(action_taken=action)
        )
        result = await self._session.execute(stmt)
        await self._session.flush()
        return cast("CursorResult[Any]", result).rowcount or 0

    async def count_in_window(self, start: datetime, end: datetime) -> int:
        """Count detections created in ``[start, end)`` for this guild."""
        stmt = select(func.count()).where(
            Detection.guild_id == self._guild_id,
            Detection.created_at >= start,
            Detection.created_at < end,
        )
        return int((await self._session.execute(stmt)).scalar_one())

    async def delete_older_than(self, cutoff: datetime) -> int:
        """Delete detections older than ``cutoff``; return rows deleted."""
        stmt = delete(Detection).where(
            Detection.guild_id == self._guild_id, Detection.created_at < cutoff
        )
        result = await self._session.execute(stmt)
        await self._session.flush()
        return cast("CursorResult[Any]", result).rowcount or 0


class ModActionRepository:
    """Append-only audit log of administrative actions, scoped per guild."""

    def __init__(self, session: AsyncSession, guild_id: int) -> None:
        self._session = session
        self._guild_id = guild_id

    async def record(
        self,
        *,
        actor_id: int,
        action: str,
        target: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> ModAction:
        """Append an audit row for this guild."""
        row = ModAction(
            guild_id=self._guild_id,
            actor_id=actor_id,
            action=action,
            target=target,
            payload=payload or {},
        )
        self._session.add(row)
        await self._session.flush()
        return row

    async def list_recent(self, limit: int = 100) -> Sequence[ModAction]:
        """Return the most recent audit rows for this guild."""
        stmt = (
            select(ModAction)
            .where(ModAction.guild_id == self._guild_id)
            .order_by(ModAction.created_at.desc())
            .limit(limit)
        )
        return (await self._session.execute(stmt)).scalars().all()

    async def delete_older_than(self, cutoff: datetime) -> int:
        """Delete audit rows older than ``cutoff``; return rows deleted."""
        stmt = delete(ModAction).where(
            ModAction.guild_id == self._guild_id, ModAction.created_at < cutoff
        )
        result = await self._session.execute(stmt)
        await self._session.flush()
        return cast("CursorResult[Any]", result).rowcount or 0


class AppealRepository:
    """Guild-scoped user appeals against detections."""

    def __init__(self, session: AsyncSession, guild_id: int) -> None:
        self._session = session
        self._guild_id = guild_id

    async def open(self, *, detection_id: int, user_id: int) -> Appeal:
        """Open a new appeal for a detection."""
        row = Appeal(guild_id=self._guild_id, detection_id=detection_id, user_id=user_id)
        self._session.add(row)
        await self._session.flush()
        return row

    async def delete_older_than(self, cutoff: datetime) -> int:
        """Delete appeals older than ``cutoff``; return rows deleted."""
        stmt = delete(Appeal).where(
            Appeal.guild_id == self._guild_id, Appeal.created_at < cutoff
        )
        result = await self._session.execute(stmt)
        await self._session.flush()
        return cast("CursorResult[Any]", result).rowcount or 0


class StatsRollupRepository:
    """Per-guild hourly statistics rollups (upserted by bucket)."""

    def __init__(self, session: AsyncSession, guild_id: int) -> None:
        self._session = session
        self._guild_id = guild_id

    async def upsert(
        self,
        *,
        bucket_start: datetime,
        detections: int,
        false_positives: int,
        actions: dict[str, Any],
    ) -> StatsRollup:
        """Insert or replace the rollup for ``bucket_start``."""
        stmt = select(StatsRollup).where(
            StatsRollup.guild_id == self._guild_id,
            StatsRollup.bucket_start == bucket_start,
        )
        existing = (await self._session.execute(stmt)).scalar_one_or_none()
        if existing is None:
            existing = StatsRollup(guild_id=self._guild_id, bucket_start=bucket_start)
            self._session.add(existing)
        existing.detections = detections
        existing.false_positives = false_positives
        existing.actions = actions
        await self._session.flush()
        return existing


class EvidenceRepository:
    """Tracks stored evidence objects and their expiry (not guild-scoped)."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def record(self, *, detection_id: int, object_key: str, expires_at: datetime) -> Evidence:
        """Persist a reference to a stored evidence object."""
        row = Evidence(detection_id=detection_id, object_key=object_key, expires_at=expires_at)
        self._session.add(row)
        await self._session.flush()
        return row

    async def list_expired(self, now: datetime, limit: int = 500) -> Sequence[Evidence]:
        """Return evidence rows whose TTL has elapsed."""
        stmt = select(Evidence).where(Evidence.expires_at < now).limit(limit)
        return (await self._session.execute(stmt)).scalars().all()

    async def delete(self, evidence_id: int) -> int:
        """Delete an evidence row by id; return rows deleted."""
        stmt = delete(Evidence).where(Evidence.id == evidence_id)
        result = await self._session.execute(stmt)
        await self._session.flush()
        return cast("CursorResult[Any]", result).rowcount or 0


class GuildListRepository:
    """Account-wide guild enumeration (not guild-scoped) for sweep jobs."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def all_ids(self) -> Sequence[int]:
        """Return every configured guild id."""
        stmt = select(Guild.guild_id)
        return (await self._session.execute(stmt)).scalars().all()

    async def retention_days(self, guild_id: int) -> int | None:
        """Return a guild's configured retention window in days."""
        stmt = select(Guild.retention_days).where(Guild.guild_id == guild_id)
        return (await self._session.execute(stmt)).scalar_one_or_none()


class UserOptoutRepository:
    """Tracks users who have exercised their right to erasure (not guild-scoped)."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def is_opted_out(self, user_id: int) -> bool:
        """Return whether ``user_id`` has opted out of processing."""
        return (await self._session.get(UserOptout, user_id)) is not None

    async def opt_out(self, user_id: int) -> UserOptout:
        """Record an opt-out for ``user_id`` (idempotent)."""
        row = await self._session.get(UserOptout, user_id)
        if row is None:
            row = UserOptout(user_id=user_id)
            self._session.add(row)
            await self._session.flush()
        return row

    async def purge_user(self, user_id: int) -> int:
        """Delete a user's rows across every guild-scoped table; return rows deleted.

        Detections (and their cascading appeals/evidence) plus any trusted-user
        entries are removed. The opt-out tombstone itself is left in place so the
        user remains excluded from future processing.
        """
        total = 0
        for stmt in (
            delete(Detection).where(Detection.uploader_id == user_id),
            delete(Appeal).where(Appeal.user_id == user_id),
            delete(GuildTrustedUser).where(GuildTrustedUser.user_id == user_id),
        ):
            result = await self._session.execute(stmt)
            total += cast("CursorResult[Any]", result).rowcount or 0
        await self._session.flush()
        return total


class GuildPurgeRepository:
    """Full per-guild data erasure for the ``/delete_server_data`` GDPR flow."""

    def __init__(self, session: AsyncSession, guild_id: int) -> None:
        self._session = session
        self._guild_id = guild_id

    async def purge(self) -> int:
        """Delete every row owned by this guild; return total rows deleted.

        The ``guilds`` config row is removed last so its ``ON DELETE CASCADE``
        children (ignored channels/roles, trusted users, guild hashes, whitelist)
        are torn down with it.
        """
        gid = self._guild_id
        total = 0
        for stmt in (
            delete(Appeal).where(Appeal.guild_id == gid),
            delete(Detection).where(Detection.guild_id == gid),
            delete(ModAction).where(ModAction.guild_id == gid),
            delete(StatsRollup).where(StatsRollup.guild_id == gid),
            delete(GuildHash).where(GuildHash.guild_id == gid),
            delete(GuildWhitelist).where(GuildWhitelist.guild_id == gid),
            delete(GuildChannelIgnored).where(GuildChannelIgnored.guild_id == gid),
            delete(GuildRoleIgnored).where(GuildRoleIgnored.guild_id == gid),
            delete(GuildTrustedUser).where(GuildTrustedUser.guild_id == gid),
            delete(Guild).where(Guild.guild_id == gid),
        ):
            result = await self._session.execute(stmt)
            total += cast("CursorResult[Any]", result).rowcount or 0
        await self._session.flush()
        return total
