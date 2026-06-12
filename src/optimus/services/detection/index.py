"""In-memory hash indexes (BK-trees) for guild-local and global scam hashes.

Each guild gets a lazily-built phash BK-tree keyed to its rows; a single global
BK-tree holds promoted cross-guild hashes. The full hash set for each matched
payload is retained so the ensemble can vote across all four hashes after the
BK-tree narrows candidates by phash. Indexes rebuild from Postgres on boot and
on demand when an invalidation arrives.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

from prometheus_client import Counter

from optimus.core.logging import get_logger
from optimus.db.engine import SessionScope
from optimus.db.repositories import GlobalHashRepository, GuildHashRepository
from optimus.hashing.bktree import BKTree

_log = get_logger(__name__)

GUILD_INDEX_EVICTED = Counter(
    "optimus_detection_guild_index_evicted_total",
    "Per-guild hash indexes evicted from the LRU cache.",
)


@dataclass(frozen=True, slots=True)
class KnownHash:
    """A known scam hash set with provenance for matching."""

    hash_id: str
    phash: int
    dhash: int
    whash: int
    ahash: int
    source: str  # "guild" | "global"
    campaign_id: str | None = None

    def as_dict(self) -> dict[str, int]:
        """Return the four-hash mapping consumed by the ensemble."""
        return {
            "phash": self.phash,
            "dhash": self.dhash,
            "whash": self.whash,
            "ahash": self.ahash,
        }


class HashIndex:
    """A phash BK-tree plus a payload table mapping hash_id -> :class:`KnownHash`."""

    def __init__(self, entries: Sequence[KnownHash]) -> None:
        self._tree = BKTree()
        self._by_id: dict[str, KnownHash] = {}
        for entry in entries:
            # Collisions on hash_id keep the last; phash collisions are fine.
            self._by_id[entry.hash_id] = entry
            self._tree.add(entry.phash, entry.hash_id)

    def __len__(self) -> int:
        return len(self._by_id)

    def candidates(self, phash: int, radius: int) -> list[KnownHash]:
        """Return known hashes whose phash is within ``radius`` of ``phash``."""
        out: list[KnownHash] = []
        for match in self._tree.query(phash, radius):
            if match.payload is not None and match.payload in self._by_id:
                out.append(self._by_id[match.payload])
        return out


class IndexManager:
    """Owns per-guild and global indexes and rebuilds them from Postgres.

    ``session_scope`` is an async-context-manager factory yielding an
    :class:`AsyncSession` (e.g. :func:`optimus.db.engine.session_scope` bound to a
    factory). Indexes are cached until invalidated.

    The per-guild cache is an LRU bounded by ``max_guilds`` (``None`` = unbounded):
    once the cap is exceeded the least-recently-used guild index is dropped and
    rebuilt on demand the next time it is queried. The cache is touched on every
    read, so the most active guilds stay resident. Eviction runs synchronously
    *after* a freshly built index is stored, and the just-stored guild is moved to
    the most-recent end first, so it can never evict the index a caller is about to
    return — the only await inside :meth:`guild_index` is the build, and within the
    single event loop the store-and-evict sequence after it is indivisible.
    """

    def __init__(self, session_scope: SessionScope, *, max_guilds: int | None = None) -> None:
        if max_guilds is not None and max_guilds < 1:
            raise ValueError("max_guilds must be >= 1 or None")
        self._scope = session_scope
        self._guilds: OrderedDict[int, HashIndex] = OrderedDict()
        self._max_guilds = max_guilds
        self._global: HashIndex | None = None

    async def guild_index(self, guild_id: int) -> HashIndex:
        """Return (building if needed) the index for ``guild_id``."""
        index = self._guilds.get(guild_id)
        if index is None:
            index = await self._build_guild(guild_id)
            self._store_guild(guild_id, index)
        else:
            self._guilds.move_to_end(guild_id)
        return index

    def _store_guild(self, guild_id: int, index: HashIndex) -> None:
        """Insert/refresh ``guild_id`` as most-recent, then evict over-cap LRUs."""
        self._guilds[guild_id] = index
        self._guilds.move_to_end(guild_id)
        if self._max_guilds is None:
            return
        while len(self._guilds) > self._max_guilds:
            evicted, _ = self._guilds.popitem(last=False)
            GUILD_INDEX_EVICTED.inc()
            _log.info("guild_index_evicted", guild_id=evicted, cache_size=len(self._guilds))

    def cached_guilds(self) -> list[int]:
        """Return resident guild ids in LRU order (oldest first)."""
        return list(self._guilds)

    async def global_index(self) -> HashIndex:
        """Return (building if needed) the global promoted-hash index."""
        if self._global is None:
            self._global = await self._build_global()
        return self._global

    async def invalidate(self, guild_id: int | None) -> None:
        """Reload the given guild's index, or the global index when ``None``."""
        if guild_id is None:
            self._global = await self._build_global()
        else:
            self._store_guild(guild_id, await self._build_guild(guild_id))

    async def _build_guild(self, guild_id: int) -> HashIndex:
        async with self._scope() as session:
            repo = GuildHashRepository(session, guild_id)
            rows = await repo.list_active()
            entries = [
                KnownHash(
                    hash_id=r.hash_id,
                    phash=r.phash,
                    dhash=r.dhash,
                    whash=r.whash,
                    ahash=r.ahash,
                    source="guild",
                )
                for r in rows
            ]
        return HashIndex(entries)

    async def _build_global(self) -> HashIndex:
        async with self._scope() as session:
            repo = GlobalHashRepository(session)
            rows = await repo.list_promoted()
            entries = [
                KnownHash(
                    hash_id=r.hash_id,
                    phash=r.phash,
                    dhash=r.dhash,
                    whash=r.whash,
                    ahash=0,
                    source="global",
                    campaign_id=r.campaign_id,
                )
                for r in rows
            ]
        return HashIndex(entries)
