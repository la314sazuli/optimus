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

# Internal BK-tree node-key suffix for a source's mirror sibling. The NUL byte
# cannot occur in a real hash_id, so the suffixed key never collides.
_MIRROR_SUFFIX = "\x00mirror"


def _mirror_dict(
    mphash: int | None, mdhash: int | None, mwhash: int | None, mahash: int | None
) -> dict[str, int] | None:
    """Assemble a mirror hash-set from stored columns, or ``None`` if absent.

    The mirror is indexed only when its phash/dhash/whash are all present (rows
    added before flip support, or via a hex/import path with no image, leave them
    NULL and simply contribute no mirror entry).
    """
    if mphash is None or mdhash is None or mwhash is None:
        return None
    return {"phash": mphash, "dhash": mdhash, "whash": mwhash, "ahash": mahash or 0}


GUILD_INDEX_EVICTED = Counter(
    "optimus_detection_guild_index_evicted_total",
    "Per-guild hash indexes evicted from the LRU cache.",
)


@dataclass(frozen=True, slots=True)
class KnownHash:
    """A known scam hash set with provenance for matching.

    ``mirror`` optionally carries the hash set of the horizontally-flipped source
    image. When present it is indexed as a sibling entry under the *same*
    ``hash_id`` and ``campaign_id``, so a mirrored re-share matches back to the
    same source detection (dedup/ownership preserved). Mirror hashes are the
    actual hashes of the flipped pixels, not a permutation of the originals, so
    the ensemble scores a flipped upload at ~zero distance.
    """

    hash_id: str
    phash: int
    dhash: int
    whash: int
    ahash: int
    source: str  # "guild" | "global"
    campaign_id: str | None = None
    mirror: dict[str, int] | None = None

    def as_dict(self) -> dict[str, int]:
        """Return the four-hash mapping consumed by the ensemble."""
        return {
            "phash": self.phash,
            "dhash": self.dhash,
            "whash": self.whash,
            "ahash": self.ahash,
        }

    def mirror_entry(self) -> KnownHash | None:
        """Return the mirrored sibling :class:`KnownHash`, or ``None``.

        Shares this entry's ``hash_id`` and ``campaign_id`` so a match against
        the mirror resolves to the same source. Its primary hashes are the
        mirror hashes, and it carries no further mirror of its own.
        """
        if self.mirror is None:
            return None
        return KnownHash(
            hash_id=self.hash_id,
            phash=self.mirror["phash"],
            dhash=self.mirror["dhash"],
            whash=self.mirror["whash"],
            ahash=self.mirror.get("ahash", 0),
            source=self.source,
            campaign_id=self.campaign_id,
            mirror=None,
        )


class HashIndex:
    """A phash BK-tree plus a payload table mapping hash_id -> :class:`KnownHash`."""

    def __init__(self, entries: Sequence[KnownHash]) -> None:
        self._tree = BKTree()
        # Tree payloads are internal node keys -> KnownHash. An entry's mirror is
        # stored under a suffixed key so its (flipped) hashes are what the
        # ensemble scores, while the returned KnownHash keeps the original
        # hash_id/campaign so a mirrored match resolves to the same source.
        self._by_node: dict[str, KnownHash] = {}
        seen: set[str] = set()
        for entry in entries:
            # Collisions on hash_id keep the last; phash collisions are fine.
            seen.add(entry.hash_id)
            self._by_node[entry.hash_id] = entry
            self._tree.add(entry.phash, entry.hash_id)
            mirror = entry.mirror_entry()
            if mirror is not None:
                node_key = f"{entry.hash_id}{_MIRROR_SUFFIX}"
                self._by_node[node_key] = mirror
                self._tree.add(mirror.phash, node_key)
        self._ids = seen

    def __len__(self) -> int:
        return len(self._ids)

    def candidates(self, phash: int, radius: int) -> list[KnownHash]:
        """Return known hashes whose phash is within ``radius`` of ``phash``.

        Both originals and mirror siblings are eligible; a mirror match yields a
        :class:`KnownHash` carrying the source ``hash_id``/``campaign_id`` but the
        flipped hashes, so the ensemble scores the flipped upload correctly.
        """
        out: list[KnownHash] = []
        for match in self._tree.query(phash, radius):
            if match.payload is not None and match.payload in self._by_node:
                out.append(self._by_node[match.payload])
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
                    mirror=_mirror_dict(r.mphash, r.mdhash, r.mwhash, r.mahash),
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
                    mirror=_mirror_dict(r.mphash, r.mdhash, r.mwhash, 0),
                )
                for r in rows
            ]
        return HashIndex(entries)
