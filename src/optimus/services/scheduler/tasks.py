"""The maintenance jobs the scheduler runs, written against repositories.

Each job is an ``async`` function taking a session scope (and any clients it
needs) so it can be exercised against an aiosqlite database in tests. The
periodic loop machinery and jitter live in :mod:`service`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

from optimus.db.engine import SessionScope
from optimus.db.repositories import (
    AppealRepository,
    DetectionRepository,
    EvidenceRepository,
    GuildListRepository,
    ModActionRepository,
    StatsRollupRepository,
    delete_appeals_before,
    delete_detections_before,
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


async def purge_old_data(
    scope: SessionScope,
    *,
    retention_days: int | None,
    batch_size: int,
    pause_seconds: float = 0.0,
    now: datetime | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> int:
    """Deployment-wide batched purge of detections/appeals past ``retention_days``.

    Disabled (returns 0 without touching the DB) when ``retention_days`` is
    ``None`` so self-hosters keep everything by default. Otherwise rows older
    than the cutoff are deleted in ``batch_size`` chunks, each in its own
    transaction, sleeping ``pause_seconds`` between batches to avoid long locks
    and yield to foreground traffic. FK order is respected: appeals are purged
    before detections (and appeals under purged detections cascade away).

    Returns the total number of rows deleted across both tables.
    """
    if retention_days is None:
        return 0
    moment = now or datetime.now(UTC)
    cutoff = moment - timedelta(days=retention_days)
    deleted = 0
    for delete_batch in (delete_appeals_before, delete_detections_before):
        while True:
            async with scope() as session:
                removed = await delete_batch(session, cutoff, limit=batch_size)
            deleted += removed
            if removed < batch_size:
                break
            if pause_seconds > 0:
                await sleep(pause_seconds)
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

    Deliberately split into three phases -- list, then delete-from-object-
    storage, then delete-rows -- rather than looping the network
    ``delete_object`` call inside one open transaction. ``delete_object`` is a
    real network call once wired to actual object storage (this job is a
    no-op today only because simple mode passes a no-op ``delete_object``);
    holding a SQLite write transaction open across N sequential network calls
    would block every other writer in the process for as long as that loop
    takes, exactly the anti-pattern fixed for the reviewmsg attachment path
    (see ``compute_attachment_hashes``/``store_attachment_hash`` in
    ``services/interactions``). Object-storage deletes here run with no
    transaction open at all; each row's DB delete is its own short
    transaction, so a mid-loop failure only leaves that one object
    undeleted-but-already-removed-from-storage rather than rolling back
    unrelated rows or holding a lock for the whole batch.
    """
    moment = now or datetime.now(UTC)
    async with scope() as session:
        expired = list(await EvidenceRepository(session).list_expired(moment))

    removed = 0
    for row in expired:
        await delete_object(row.object_key)  # type: ignore[operator]
        async with scope() as session:
            removed += await EvidenceRepository(session).delete(row.id)
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
