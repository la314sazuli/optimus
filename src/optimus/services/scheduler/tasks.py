"""The maintenance jobs the scheduler runs, written against repositories.

Each job is an ``async`` function taking a session scope (and any clients it
needs) so it can be exercised against an aiosqlite database in tests. The
periodic loop machinery and jitter live in :mod:`service`.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from optimus.db.engine import SessionScope
from optimus.db.repositories import (
    AppealRepository,
    DetectionRepository,
    EvidenceRepository,
    GuildListRepository,
    ModActionRepository,
    StatsRollupRepository,
)


def hour_bucket(moment: datetime) -> datetime:
    """Truncate ``moment`` to the start of its UTC hour."""
    return moment.astimezone(UTC).replace(minute=0, second=0, microsecond=0)


async def enforce_retention(
    scope: SessionScope, *, default_days: int, now: datetime | None = None
) -> int:
    """Delete detections/appeals/mod-actions past each guild's retention window.

    Returns the total number of rows deleted across all guilds.
    """
    moment = now or datetime.now(UTC)
    deleted = 0
    async with scope() as session:
        guild_ids = list(await GuildListRepository(session).all_ids())
        retention = {
            gid: (await GuildListRepository(session).retention_days(gid) or default_days)
            for gid in guild_ids
        }
    for gid in guild_ids:
        cutoff = moment - timedelta(days=retention[gid])
        async with scope() as session:
            deleted += await DetectionRepository(session, gid).delete_older_than(cutoff)
            deleted += await AppealRepository(session, gid).delete_older_than(cutoff)
            deleted += await ModActionRepository(session, gid).delete_older_than(cutoff)
    return deleted


async def cleanup_evidence(
    scope: SessionScope,
    *,
    delete_object: object,
    now: datetime | None = None,
) -> int:
    """Delete expired evidence objects and their tracking rows.

    ``delete_object`` is an async callable ``(object_key) -> None``. Returns the
    number of evidence rows removed.
    """
    moment = now or datetime.now(UTC)
    removed = 0
    async with scope() as session:
        repo = EvidenceRepository(session)
        expired = list(await repo.list_expired(moment))
        for row in expired:
            await delete_object(row.object_key)  # type: ignore[operator]
            await repo.delete(row.id)
            removed += 1
    return removed


async def roll_up_stats(scope: SessionScope, *, now: datetime | None = None) -> int:
    """Upsert the previous hour's detection counts per guild.

    Returns the number of guild rollups written.
    """
    moment = now or datetime.now(UTC)
    bucket = hour_bucket(moment) - timedelta(hours=1)
    bucket_end = bucket + timedelta(hours=1)
    written = 0
    async with scope() as session:
        guild_ids = list(await GuildListRepository(session).all_ids())
    for gid in guild_ids:
        async with scope() as session:
            count = await DetectionRepository(session, gid).count_in_window(bucket, bucket_end)
            await StatsRollupRepository(session, gid).upsert(
                bucket_start=bucket,
                detections=count,
                false_positives=0,
                actions={},
            )
            written += 1
    return written
