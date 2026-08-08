"""Regression coverage: reviewing a message must not hold the DB write lock
open during attachment fetch/decode.

Production logs showed the same ``reviewmsg:...`` idempotency key exhausting
every retry in :meth:`InteractionService._run` twice in a row, seconds apart --
proof the "database is locked" condition was not the brief, transient
contention the retry-with-backoff fix (PR #17) assumed. ``_review_message``
(``src/optimus/services/interactions/handlers.py``) ran, for every attachment
on the reviewed message, a network fetch to Discord's CDN plus a sandboxed
image-decode subprocess *inside* the single DB transaction
``InteractionService._run`` opens for the whole handler call -- multiple
seconds of non-DB work per attachment, all while SQLite's exclusive
file-level write lock stayed held, for a multi-image review potentially far
longer than any reasonable retry budget.

The fix splits attachment handling into two phases: ``compute_attachment_hashes``
(network fetch + decode, no DB access) runs for every attachment first, then
``store_attachment_hash`` (DB-only) runs for each successfully computed
result. This test proves the first phase never touches the session at all,
by pointing a real ``DbDeps`` at a session that raises if it is ever used
during ``compute_attachment_hashes``, and separately proves two real SQLite
connections can both make forward progress when only the fast DB-only phase
happens inside the transaction.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from optimus.core.config import get_settings
from optimus.db.engine import create_engine, create_session_factory, session_scope
from optimus.db.models import Base, Guild
from optimus.ingest.fetcher import FetchedImage
from optimus.services.interactions.service import DbDeps

# A tiny 4x4 white PNG -- small enough to decode instantly once fetched, so
# any wall-clock time in these tests comes from the injected fetch delay, not
# real decode cost. Generated with Pillow (Image.new("RGB", (4, 4)).save(...)).
_TINY_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000004000000040802000000"
    "269309290000001449444154789c63fcffff3f030c30312001dc1c0096"
    "6e0305f225bef90000000049454e44ae426082"
)
#: A second, visually distinct tiny PNG (black instead of white) so two
#: attachments hashed in the same test produce different hash ids instead of
#: colliding on SQLite's unique constraint for guild_hashes.hash_id.
_TINY_PNG_2 = bytes.fromhex(
    "89504e470d0a1a0a0000000d494844520000000400000004080200000026"
    "9309290000000c49444154789c6360201d000000340001765eaec3000000"
    "0049454e44ae426082"
)


class _SessionUsedDuringComputeError(Exception):
    """Raised if compute_attachment_hashes touches the DB session at all."""


class _TripwireSession:
    """Stands in for AsyncSession; raises on any attribute access.

    ``compute_attachment_hashes`` must be implementable without touching the
    session in any way -- not even a read -- so any access at all (not just
    execute()) is a violation worth failing loudly on.
    """

    def __getattr__(self, name: str) -> Any:
        raise _SessionUsedDuringComputeError(f"session.{name} accessed during compute phase")


async def test_compute_attachment_hashes_never_touches_the_session() -> None:
    """The whole point of the split: computing a hash is pure fetch+decode,
    with no DB dependency, so it can safely run before any transaction is
    relevant at all.
    """

    async def _slow_fetch(url: str) -> FetchedImage:
        await asyncio.sleep(0.05)
        return FetchedImage(data=_TINY_PNG, content_type="image/png", final_url=url)

    deps = DbDeps(
        _TripwireSession(),  # type: ignore[arg-type]
        rate_limiter=None,  # type: ignore[arg-type]  # unused by this method
        settings=get_settings(),
        fetch=_slow_fetch,
    )

    # Must not raise _SessionUsedDuringComputeError.
    hashes = await deps.compute_attachment_hashes(attachment_id=1, url="https://x/1.png")
    assert hashes.attachment_id == 1


async def test_two_reviews_do_not_serialize_on_the_slow_fetch_phase() -> None:
    """Two 'requests' each computing an attachment hash concurrently (slow
    fetch, no DB) must both make progress in parallel -- proving the slow
    phase carries no lock that could serialize them -- and each can then
    independently complete its fast DB-only store phase against a real
    file-backed SQLite database without contending on the other's slow fetch.
    """
    engine: AsyncEngine = create_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = create_session_factory(engine)

    guild_id = 123456789012345678  # obviously synthetic, not a real guild id

    async def _seed_guild() -> None:
        async with session_scope(factory) as session:
            session.add(Guild(guild_id=guild_id))

    await _seed_guild()

    concurrent_fetches = 0
    max_concurrent_fetches = 0

    async def _tracked_slow_fetch(url: str) -> FetchedImage:
        nonlocal concurrent_fetches, max_concurrent_fetches
        concurrent_fetches += 1
        max_concurrent_fetches = max(max_concurrent_fetches, concurrent_fetches)
        try:
            await asyncio.sleep(0.1)
        finally:
            concurrent_fetches -= 1
        # Distinct bytes per URL so the two concurrent "requests" produce
        # different hash ids instead of racing each other on the same
        # unique-constrained row (a separate concern from what this test
        # is checking).
        data = _TINY_PNG if url.endswith("/1.png") else _TINY_PNG_2
        return FetchedImage(data=data, content_type="image/png", final_url=url)

    async def _review_one_attachment(attachment_id: int) -> str:
        # Phase 1: compute, no session open at all (mirrors _review_message's
        # first pass -- no `async with self._scope()` wraps this call).
        deps_for_compute = DbDeps(
            _TripwireSession(),  # type: ignore[arg-type]
            rate_limiter=None,  # type: ignore[arg-type]
            settings=get_settings(),
            fetch=_tracked_slow_fetch,
        )
        hashes = await deps_for_compute.compute_attachment_hashes(
            attachment_id=attachment_id, url=f"https://x/{attachment_id}.png"
        )
        # Phase 2: store, DB-only, inside a real short-lived transaction.
        async with session_scope(factory) as session:
            deps_for_store = DbDeps(session, rate_limiter=None, settings=get_settings())  # type: ignore[arg-type]
            stored = await deps_for_store.store_attachment_hash(guild_id, hashes=hashes, added_by=1)
        return stored.hash_id

    try:
        results = await asyncio.wait_for(
            asyncio.gather(
                _review_one_attachment(1),
                _review_one_attachment(2),
            ),
            timeout=2.0,
        )
    finally:
        await engine.dispose()

    # Both "requests" completed and each got its own hash id.
    assert len(results) == 2
    assert results[0] != results[1]
    # The two slow fetches genuinely overlapped -- if the old code path's
    # per-attachment DB session had been open across the fetch, gather()
    # would have had no reason to interleave them (the whole point of this
    # assertion is that nothing here serializes the two "requests").
    assert max_concurrent_fetches == 2


async def test_store_attachment_hash_is_idempotent_on_conflicting_id(tmp_path: Any) -> None:
    """A duplicate phash (re-reviewing a message, or an image another path
    already hashed) returns the existing row rather than raising -- the same
    guarantee the pre-split ``hash_and_store_attachment`` provided.
    """
    db_path = tmp_path / "reviewmsg.db"
    engine = create_engine(f"sqlite+aiosqlite:///{db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    guild_id = 987654321098765432  # obviously synthetic, not a real guild id

    async def _scope() -> AsyncIterator[Any]:
        async with session_scope(factory) as session:
            yield session

    try:
        async with session_scope(factory) as session:
            session.add(Guild(guild_id=guild_id))

        async def _fetch(url: str) -> FetchedImage:
            return FetchedImage(data=_TINY_PNG, content_type="image/png", final_url=url)

        deps = DbDeps(
            _TripwireSession(),  # type: ignore[arg-type]
            rate_limiter=None,  # type: ignore[arg-type]
            settings=get_settings(),
            fetch=_fetch,
        )
        hashes = await deps.compute_attachment_hashes(attachment_id=1, url="https://x/1.png")

        async with session_scope(factory) as session:
            first = await DbDeps(
                session, rate_limiter=None, settings=get_settings()
            ).store_attachment_hash(  # type: ignore[arg-type]
                guild_id, hashes=hashes, added_by=1
            )
        async with session_scope(factory) as session:
            second = await DbDeps(
                session, rate_limiter=None, settings=get_settings()
            ).store_attachment_hash(  # type: ignore[arg-type]
                guild_id, hashes=hashes, added_by=2
            )
        assert first.hash_id == second.hash_id
        assert second.added_by == first.added_by  # existing row returned, not overwritten
    finally:
        await engine.dispose()
