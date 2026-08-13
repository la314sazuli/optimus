"""Regression: reviewmsg must not deadlock against its own open transaction.

The production incident behind PRs #17-#20: ``/scamhash reviewmsg`` (and the
"Review as scam" context menu) failed with ``database is locked`` on every
attempt, on an otherwise idle bot. The cause was never external contention --
the command deadlocked against **itself**:

1. ``store_attachment_hash`` flushes an INSERT through the interaction's
   session, so that transaction now holds SQLite's single file-level write
   lock until commit (at session-scope exit);
2. still inside that transaction, ``submit_confirmed_scam`` used to call
   ``DetectionService.submit_confirmed_match``, which opens a **second**
   session (a second pooled connection) and INSERTs the detection row;
3. that second INSERT blocks on the write lock held by step 1 -- which cannot
   release until step 2 returns -- so it always burns the full
   ``busy_timeout`` and raises ``database is locked``. Every retry re-entered
   the identical deadlock, so the whole retry budget failed too.

This test drives the real ``InteractionService._run`` against a real
file-backed SQLite database (two pooled connections would be involved, exactly
like production; ``:memory:`` cannot reproduce a cross-connection file lock)
with a deliberately short busy timeout, and asserts the sequence completes.
Against the pre-fix code this exact test fails with ``database is locked``.
"""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from optimus.app.memory import MemoryStore
from optimus.bus.inprocess import InProcessBus
from optimus.contracts.events import SUBJECT_VERDICT, VerdictEvent
from optimus.core.config import Settings
from optimus.core.ratelimit import InMemoryRateLimiter
from optimus.db.engine import create_engine, create_session_factory, session_scope
from optimus.db.models import Base, Guild, GuildHash
from optimus.db.repositories import DetectionRepository
from optimus.services.detection.service import build_service as build_detection
from optimus.services.interactions.service import InteractionResponse, InteractionService

pytestmark = pytest.mark.asyncio

GUILD_ID = 4242


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        simple_database_url=f"sqlite+aiosqlite:///{tmp_path / 'optimus.db'}",
        # Short enough that the old self-deadlock would fail fast, long enough
        # to be unambiguous that a lock (not a race) is what would be reported.
        sqlite_busy_timeout_ms=250,
    )


async def test_reviewmsg_write_sequence_survives_its_own_transaction(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    engine = create_engine(settings.simple_database_url, settings=settings)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = create_session_factory(engine)

    def scope() -> AbstractAsyncContextManager[AsyncSession]:
        return session_scope(factory)

    async with scope() as session:
        session.add(Guild(guild_id=GUILD_ID, action_policy="delete_ban"))

    bus = InProcessBus()
    detection = build_detection(
        settings, bus, MemoryStore(), session_scope_factory=scope, enable_swarm=False
    )
    service = InteractionService(scope, InMemoryRateLimiter(), settings, detection=detection)

    published: list[VerdictEvent] = []
    original_publish = detection.publish_confirmed_match

    async def record_publish(verdict: VerdictEvent) -> None:
        published.append(verdict)
        await original_publish(verdict)

    detection.publish_confirmed_match = record_publish  # type: ignore[method-assign]

    # The exact write sequence _review_message performs per attachment, run
    # through the real _run session scope: a flushed guild-hash INSERT (which
    # takes the transaction's write lock) followed by the confirmed-scam
    # persistence. Pre-fix, the second step opened a second connection and
    # deadlocked on the first step's lock.
    async def call(deps: object) -> InteractionResponse:
        stored = await deps.add_guild_hash(  # type: ignore[attr-defined]
            GUILD_ID,
            GuildHash(hash_id=f"{7:016x}", phash=7, dhash=7, whash=7, source="local"),
        )
        assert len(published) == 0, "verdict published before the transaction committed"
        await deps.submit_confirmed_scam(  # type: ignore[attr-defined]
            GUILD_ID,
            channel_id=111,
            message_id=222,
            attachment_id=333,
            uploader_id=444,
            matched_hash_id=stored.hash_id,
        )
        assert len(published) == 0, "verdict published before the transaction committed"
        return InteractionResponse("ok")

    try:
        response = await service._run(call)

        assert response.i18n_key == "ok"
        # The detection row committed in the same transaction as the hash...
        async with scope() as session:
            row = await DetectionRepository(session, GUILD_ID).get_by_idempotency_key(
                f"reviewmsg:{GUILD_ID}:222:333"
            )
            assert row is not None
            assert row.verdict == "scam"
        # ...and the verdict reached the bus exactly once, only after commit.
        assert [v.idempotency_key for v in published] == [f"reviewmsg:{GUILD_ID}:222:333"]
        assert published[0].correlation_id  # sanity: a real, well-formed event
        assert SUBJECT_VERDICT  # imported for documentation of the publish subject
    finally:
        await engine.dispose()
